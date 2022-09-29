using Flux
using Flux.Losses
using Zygote
using Plots

export SupportProposalNet, FPCRNet, CDFNet, FPCRLearner

Base.@kwdef struct SupportProposalNet
    net
    scale
    loc
    N::Int
end

function SupportProposalNet(emb_size::Int, N::Int, n_actions::Int)
    net = Chain(
        Dense(emb_size => n_actions * N),
    )
    scale = Dense(emb_size => n_actions)
    loc = Dense(emb_size => n_actions)
    return SupportProposalNet(net, scale, loc, N)
end

Flux.@functor SupportProposalNet

function (net::SupportProposalNet)(state_emb)
    log_p = reshape(net.net(state_emb), :, net.N, size(state_emb, 2))
    log_p = logsoftmax(log_p, dims=2)
    support_01 = cumsum(exp.(log_p), dims=2)
    log_scale = Flux.unsqueeze(net.scale(state_emb), dims=2)
    scale = exp.(log_scale)
    loc = Flux.unsqueeze(net.loc(state_emb), dims=2)

    # support = support_01
    # @show size()
    support = (support_01 .+ loc) .* scale
    entropies = sum(-(support_01 .* log_p .+ log_scale), dims=(1, 2))

    # support = reshape(support, :, net.N, size(state_emb, 2))
    return support, entropies
end

Base.@kwdef struct CDFNet{A}
    net::A
    N::Int
end

Flux.@functor CDFNet

function (net::CDFNet)(state_emb, support)
    ξ = net.net(state_emb)
    loc, log_scale = chunk(ξ, 2, dims=1)
    loc = reshape(loc, net.N, :, size(loc, 2))
    log_scale = reshape(log_scale, net.N, :, size(log_scale, 2))
    scale = softplus.(log_scale)

    support = Flux.unsqueeze(support, 1)
    loc = Flux.unsqueeze(loc, 3)
    scale = Flux.unsqueeze(scale, 3)

    mix = sigmoid.((support .- loc) ./ scale)
    cdf = dropdims(sum(mix, dims=1), dims=1) ./ size(mix, 1)
    deriv = mix .* (1 .- mix) ./ scale
    pdf = dropdims(sum(deriv, dims=1), dims=1) ./ size(mix, 1)

    # @show cdf[1,:,1]
    # @show support

    return cdf, pdf
end

Base.@kwdef struct FPCRNet{A}
    base::A
    support::SupportProposalNet
    cdf::CDFNet
end

Flux.@functor FPCRNet

function (net::FPCRNet)(s; only_mid=true)
    state_emb = net.base(s)  # (n_feature, batch_size)
    support, entropies = net.support(state_emb) # (n_actions, N, batch_size)

    return net(state_emb, support, entropies; only_mid=only_mid)
end

function (net::FPCRNet)(state_emb, support, entropies; only_mid=true)
    support_mid = (support[:, 1:(end-1), :] .+ support[:, 2:end, :]) ./ 2  # (n_action, N-1, batch_size)
    support_diff = support[:, 2:end, :] .- support[:, 1:(end-1), :]  # (n_action, N-1, batch_size)

    cdf_mid, pdf_mid = net.cdf(state_emb, support_mid)
    cdf_mid = reshape(cdf_mid, size(support_mid, 1), :, size(state_emb, 2))  # (n_action, N-1, batch_size)
    pdf_mid = reshape(pdf_mid, size(support_mid, 1), :, size(state_emb, 2))  # (n_action, N-1, batch_size)

    q_value = dropdims(sum(support_diff .* support_mid .* pdf_mid, dims=2), dims=2)
    if only_mid
        return cdf_mid, pdf_mid, support_mid, q_value
    end

    cdf, pdf = Zygote.@ignore net.cdf(state_emb, support)
    cdf = reshape(cdf, size(support, 1), :, size(state_emb, 2))  # (n_action, N, batch_size)
    pdf = reshape(pdf, size(support, 1), :, size(state_emb, 2))  # (n_action, N, batch_size)

    cdf_left = zeros(size(cdf, 1), 1, size(cdf, 3))
    cdf_right = ones(size(cdf, 1), 1, size(cdf, 3))

    cdf_ext = cat(cdf_left, cdf_mid, cdf_right, dims=2)
    grad = Zygote.@ignore 2cdf .- cdf_ext[:, 1:end-1, :] .- cdf_ext[:, 2:end, :]
    l = grad .* support

    return cdf_mid, pdf_mid, support_mid, q_value, cdf, pdf, support, l, entropies
end

# function energy_distance(x, y)
#     n = size(x, 2)
#     m = size(y, 2)

#     x_ = Flux.unsqueeze(x, dims=2)
#     _x = Flux.unsqueeze(x, dims=3)
#     _y = Flux.unsqueeze(y, dims=3)
#     d_xy = dropdims(sum((x_ .- _y) .^ 2, dims=(2,3)), dims=(2,3))
#     d_xx = dropdims(sum((x_ .- _x) .^ 2, dims=(2,3)), dims=(2,3))
#     ε = 2 / (n * m) .* d_xy .- 1 / n^2 .* d_xx
#     return ε
# end


Base.@kwdef mutable struct FPCRLearner{A<:Approximator{<:TwinNetwork}} <: AbstractLearner
    approximator::A
    support_optimiser
    ent_coef::Float32 = 0.001f0
    γ::Float32 = 0.99f0
    κ::Float32 = 1.0f0
    rng::AbstractRNG = GLOBAL_RNG
    device_rng::AbstractRNG = rng
    # for logging
    logging_params = DefaultDict(0.0)
end

@functor FPCRLearner (approximator, device_rng)


function (learner::FPCRLearner)(s::AbstractArray)
    _, _, _, q_value = learner.approximator(s)
    return q_value
end

function (L::FPCRLearner)(env::AbstractEnv)
    s = env |> state |> send_to_device(L)
    q = s |> unsqueeze(dims=ndims(s) + 1) |> L |> vec
    q |> send_to_host
end

function RLBase.optimise!(learner::FPCRLearner, batch::NamedTuple)
    A = learner.approximator
    Z = A.model.source
    Zₜ = A.model.target
    κ = learner.κ
    D = device(Z)
    # s, s′, a, r, t = map(x -> batch[x], SS′ART)
    s = send_to_device(D, collect(batch.state))
    r = send_to_device(D, batch.reward)
    t = send_to_device(D, batch.terminal)
    s′ = send_to_device(D, collect(batch.next_state))
    a = send_to_device(D, batch.action)

    cdf_mid, _, _, q, cdf, _, support, _, entropies = Zₜ(s′; only_mid=false)

    batch_size = length(t)
    # τ′ = rand(learner.device_rng, Float32, N′, batch_size)  # TODO: support β distribution
    # τₑₘ′ = embed(τ′, Nₑₘ)
    # zₜ = Zₜ(s′, τₑₘ′)
    # avg_zₜ = mean(zₜ, dims=2)


    if haskey(batch, :next_legal_actions_mask)
        masked_value = similar(batch.next_legal_actions_mask, Float32)
        masked_value = fill!(masked_value, typemin(Float32))
        masked_value[batch.next_legal_actions_mask] .= 0
        q .+= masked_value
    end

    aₜ = argmax(Flux.unsqueeze(q, 2), dims=1)
    # aₜ = Flux.unsqueeze(aₜ, 2)

    # @show size(aₜ)
    # @show size(cdf_mid)

    aₜ = aₜ .+ typeof(aₜ)(CartesianIndices((0:0, 0:(size(cdf_mid, 2)-1), 0:0)))
    cdfₜ = reshape(cdf[aₜ], :, batch_size)
    # cdfₜ = cdf[aₜ]
    # @show size(cdfₜ)
    # target = reshape(r, 1, batch_size) .+ learner.γ * reshape(1 .- t, 1, batch_size) .* qₜ  # reshape to allow broadcast

    # τ = rand(learner.device_rng, Float32, N, batch_size)
    # τₑₘ = embed(τ, Nₑₘ)
    N = size(cdf_mid, 2)
    a = CartesianIndex.(repeat(a, inner=N), 1:(N*batch_size))
    # p = scatter(support[1,:,1], cdfₜ[:,1])
    # display(p)

    gs_cdf = gradient(Flux.params(A)) do
        state_emb = Z.base(s)
        r = reshape(r, 1, 1, :)
        t = reshape(1 .- t, 1, 1, :)

        support = (support .* t .- r) ./ learner.γ
        cdf_mid, _, _, _ = Z(state_emb, support, entropies)
        cdf_mid = reshape(cdf_mid, size(cdf_mid)[1:end-2]..., :)
        cdfₐ = reshape(cdf_mid[a], :, batch_size)

        loss = Flux.huber_loss(cdfₜ, cdfₐ)

        return loss
    end

    gs = gradient(Flux.params(Z.support)) do
        _, _, _, _, _, _, _, l, entropies = Z(s; only_mid=false)
        return mean(sum(l, dims=2) .+ learner.ent_coef .* entropies)
    end

    optimise!(A, gs_cdf)
    Flux.Optimise.update!(learner.support_optimiser, Flux.params(Z.support), gs)
end

