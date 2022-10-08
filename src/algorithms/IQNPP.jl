export IQNPPLearner, ImplicitQuantileNetPP

using Functors: @functor
using Flux: params, unsqueeze
using Random: AbstractRNG, GLOBAL_RNG
using StatsBase: mean
using StatsFuns
using Zygote: gradient
using ChainRulesCore: ignore_derivatives
using RLExp

"""
    ImplicitQuantileNet(;ψ, ϕ, header)
```
        quantiles (n_action, n_quantiles, batch_size)
           ↑
         header
           ↑
feature ↱  ⨀   ↰ transformed embedding
       ψ       ϕ
       ↑       ↑
       s        τ
```
"""
Base.@kwdef struct ImplicitQuantileNetPP{A,B,C}
    ψ::A
    ϕ::B
    header::C
end

@functor ImplicitQuantileNetPP

function (net::ImplicitQuantileNetPP)(s, emb)
    features = net.ψ(s)  # (n_feature, batch_size)
    emb_aligned = net.ϕ(emb)  # (n_feature, N * batch_size)
    merged = unsqueeze(features, dims=2) .* reshape(emb_aligned, size(features, 1), :, size(features, 2))  # (n_feature, N, batch_size)
    quantiles = net.header(reshape(merged, size(merged)[1:end-2]..., :)) # flattern last two dimension first
    reshape(quantiles, :, size(merged, 2), size(merged, 3))  # (n_action, N, batch_size)
end

Base.@kwdef mutable struct IQNPPLearner{A<:Approximator{<:TwinNetwork}} <: AbstractLearner
    approximator::A
    γ::Float32 = 0.99f0
    N::Int = 32
    N′::Int = 32
    Nₑₘ::Int = 64
    K::Int = 32
    rng::AbstractRNG = GLOBAL_RNG
    device_rng::AbstractRNG = rng
    # for logging
    loss::Float32 = 0.0f0
    logging_params = DefaultDict(0.0)
end

@functor IQNPPLearner (approximator, device_rng)

embed(x, Nₑₘ) = cos.(Float32(π) .* (1:Nₑₘ) .* reshape(x, 1, :))

# the last dimension is batch_size
function (learner::IQNPPLearner)(s::AbstractArray)
    batch_size = size(s)[end]
    τ = rand(learner.device_rng, Float32, learner.K, batch_size)

    τₑₘ = embed(τ, learner.Nₑₘ)
    quantiles = learner.approximator(s, τₑₘ)
    dropdims(mean(quantiles; dims=2); dims=2)
end

function (L::IQNPPLearner)(env::AbstractEnv)
    s = env |> state |> send_to_device(L)
    q = s |> unsqueeze(dims=ndims(s) + 1) |> L |> vec
    q |> send_to_host
end

function huber_norm(td, κ)
    abs_error = abs.(td)
    quadratic = min.(abs_error, κ)
    linear = abs_error .- quadratic
    return 0.5f0 .* quadratic .* quadratic .+ κ .* linear
end

function l2_norm(td)
    return td .^ 2
end

function l1_norm(td)
    return abs.(td)
end

function ludde_norm(td)
    d = abs.(td)
    return (d .- 1) .* d ./ log.(d .+ 1f8)
end

cauchy_norm(td) = log.(1 .+ abs.(td) .^ 2) 
function lol_norm(td)
    d = abs.(td)
    return d .* log.(1 .+ d)
end

function lol2_norm(td)
    # d = abs.(td)
    return sqrt.(1 .+ td .^ 2)
end

function mix_norm(td)
    d = reshape(td .^ 2, 1, :)
    h = gpu(1 ./ reshape(collect(1:10), :, 1))
    # s = zero(td)
    # reshape(td, )
    # for l=1:10
    #    s = s + exp.(-d ./ l) 
    # end
    reshape(sum(h * d, dims=1), size(td)...)
end

hhh = gpu(1 ./ reshape([0.01, 0.1, 0.5, 1, 5, 10, 20, 50], :, 1))

function rqn(td)
    d = reshape(td .^ 2, 1, :)
    # h = gpu(1 ./ reshape([0.01, 0.1, 0.5, 1, 5, 10, 20, 50], :, 1))
    a = 0.5
    return -1 ./ sqrt.(1 .+ hhh * d)
end

function energy_distance(x, y)
    # n = size(x, 2)
    # m = size(y, 2)
    # τₑₘ_ = Flux.unsqueeze(x, dims=2)
    # _τₑₘ = Flux.unsqueeze(x, dims=3)
    # _τₑₘ′ = Flux.unsqueeze(x, dims=3)

    # ϵ1 = Zygote.@ignore (τₑₘ_ .- _τₑₘ) .^ 2
    # ϵ2 = Zygote.@ignore (τₑₘ_ .- _τₑₘ′) .^ 2

    x_ = Flux.unsqueeze(x, dims=2)
    _x = Flux.unsqueeze(x, dims=3)
    _y = Flux.unsqueeze(y, dims=3)
    # y_ = Flux.unsqueeze(y, dims=2)
    # d_xy = dropdims(sum(lol_norm(x_ .- _y), dims=(2, 3)), dims=(2, 3))
    d_xy = rqn(x_ .- _y)
    # d_xx = dropdims(sum(lol_norm(x_ .- _x), dims=(2, 3)), dims=(2, 3))
    d_xx = rqn(x_ .- _x)
    # d_yy = mix_norm(y_ .- _y)
    # d_xx = dropdims(sum(l1_norm(y_ .- _y), dims=(2, 3)), dims=(2, 3))
    ε = 2mean(d_xy, dims=(2,3)) .- mean(d_xx, dims=(2,3)) #.- mean(d_yy, dims=(2,3))
    return ε
end

# function energy_distance_fast(x, y; κ=1.0f0)
#     n = size(x, 2)
#     m = size(y, 2)

#     x_ = permutedims(x, (2, 1, 3))
#     y_ = permutedims(y, (2, 1, 3))
#     x_ = Flux.unsqueeze(x, dims=1)
#     _x = Flux.unsqueeze(x, dims=2)
#     _y = Flux.unsqueeze(y, dims=2)
#     # d_xy = dropdims(sum(l2_norm(x_ .- _y, κ), dims=(2, 3)), dims=(2, 3))
#     # d_xx = dropdims(sum(l2_norm(x_ .- _x, κ), dims=(2, 3)), dims=(2, 3))
#     ε = dropdims(batched_mul(x_, _x)) + 2batched_mul(x_, _y)
#     # ε = 2 / (n * m) .* d_xy .- 1 / n^2 .* d_xx
#     return ε
# end

function RLBase.optimise!(learner::IQNPPLearner, batch::NamedTuple)
    A = learner.approximator
    Z = A.model.source
    Zₜ = A.model.target
    N = learner.N
    N′ = learner.N′
    Nₑₘ = learner.Nₑₘ
    D = device(Z)
    # s, s′, a, r, t = map(x -> batch[x], SS′ART)
    s = DenseCuArray(batch.state)
    # s = send_to_device(D, collect(batch.state))
    r = send_to_device(D, batch.reward)
    t = send_to_device(D, batch.terminal)
    s′ = DenseCuArray(batch.next_state)
    # s′ = send_to_device(D, collect(batch.next_state))
    a = send_to_device(D, batch.action)


    batch_size = length(t)
    τ′ = rand(learner.device_rng, Float32, N′, batch_size)  # TODO: support β distribution
    τₑₘ′ = embed(τ′, Nₑₘ)
    zₜ = Zₜ(s′, τₑₘ′)
    avg_zₜ = mean(zₜ, dims=2)

    if haskey(batch, :next_legal_actions_mask)
        masked_value = similar(batch.next_legal_actions_mask, Float32)
        masked_value = fill!(masked_value, typemin(Float32))
        masked_value[batch.next_legal_actions_mask] .= 0
        avg_zₜ .+= masked_value
    end

    aₜ = argmax(avg_zₜ, dims=1)
    aₜ = aₜ .+ typeof(aₜ)(CartesianIndices((0:0, 0:N′-1, 0:0)))
    qₜ = reshape(zₜ[aₜ], :, batch_size)
    target = reshape(r, 1, batch_size) .+ learner.γ * reshape(1 .- t, 1, batch_size) .* qₜ  # reshape to allow broadcast

    τ = rand(learner.device_rng, Float32, N, batch_size)
    τₑₘ = embed(τ, Nₑₘ)
    a = CartesianIndex.(repeat(a, inner=N), 1:(N*batch_size))

    gs = gradient(params(A)) do
        z_raw = Z(s, τₑₘ)
        # z_raw = Z(s, τₑₘ′)
        z = reshape(z_raw, size(z_raw)[1:end-2]..., :)
        q = z[a]

        target = reshape(target, 1, N′, batch_size)
        q = reshape(q, 1, N, batch_size)
        loss_batch = energy_distance(q, target)
        # loss = loss_batch .* τₑₘ
        loss = mean(loss_batch)
        ignore_derivatives() do
            learner.loss = loss
            learner.logging_params["𝐿"] = loss
        end
        loss
    end

    optimise!(A, gs)
end