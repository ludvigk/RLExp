export QQFLOWLearner, FlowNetwork, FlowNet
import ReinforcementLearning.RLBase.optimise!

using DataStructures: DefaultDict
using Distributions: Uniform, Product
using StatsBase: sample
using Flux.Losses
using CUDA: randn
using MLUtils
using SpecialFunctions
using StatsFuns
import Statistics.mean

const erf1 = Float32(erf(1))
const erfm1 = Float32(erf(-1))
const erfratio = Float32(sqrt(2π) * erf(1/sqrt(2)) / (sqrt(2π) * erf(1/sqrt(2)) + 2exp(-1/2)))

function v(x, b, c, d)
    ϵ = 1f-6
    # b = 4tanh.(b)
    # c = 4tanh.(c)
    # d = 4tanh.(d)
    xd = x .- d
    axd = abs.(xd)
    sb = softplus.(b)
    inner = logexpm1.(axd .+ sb)
    out = sign.(xd) .* (inner .- b) .- c
    # eb = exp.(b) .+ 1
    # leb = softplus.(b)
    # d = axd .+ leb .- log.(eb .* exp.(axd) .- 1)
    d = 0f0
    return out, d
end

function v⁻¹(x, b, c, d)
    # ϵ = 1f-6
    # b = 4tanh.(b)
    # c = 4tanh.(c)
    # d = 4tanh.(d)
    inner = abs.(x .+ c) .+ b
    out = sign.(x .+ c) .* (softplus.(inner) .- softplus.(b)) .+ d
    acb = abs.(c .+ x) .+ b
    # d = sigmoid.(acb)
    # d_inner = exp.(acb)
    # d = d_inner ./ (d_inner .+ 1 .+ ϵ)
    return out, logsigmoid.(acb)
end

function v2(x, b, c, d)
    xc = x .- c
    axc = abs.(xc)
    u = max.(axc, b)
    excu = exp.(axc .- u)
    exbu = exp.(b .- u)
    exu = exp.(-u)
    s = excu .+ exbu .- exu
    r = u .+ log.(s) .- b
    out = sign.(xc) .* r .+ d
    # dupper = excu
    # dlower = s
    out, log.(excu ./ s)
end

v2⁻¹(x, b, c, d) = v2(x, -b, d, c)

# function v2⁻¹(x, b, c, d)
#     xc = x .- c
#     axc = abs.(xc)
#     u = max.(axc, b)
#     s = exp.(axc .- u) .+ exp.(b .- u) .- exp.(-u)
#     r = u .+ log.(s) .- b
#     out = sign.(xc) .* r .+ d
#     out, 0f0
# end

Base.@kwdef struct FlowNet{P}
    net::P
end

Flux.@functor FlowNet

# function (m::FlowNet)(state::AbstractMatrix)
#     na = m.n_actions
#     p = m.net(state)
#     μ = p[1:na,:]
#     ρ = p[(na+1):(2na),:]
#     σ = softplus.(ρ)
#     σ = clamp.(σ, 1f-4, 1000)
#     z = Zygote.@ignore randn!(similar(μ))
#     lz = Zygote.@ignore fill!(similar(z), 0f0)
#     for i=(2na + 1):(3na):(size(p,3) - 3na + 1)
#         b, c, d = MLUtils.chunk(p[i:(i+3na-1), :], 3, dims=1)
#         z, lz_ = v(z, b, c, d)
#         lz = lz .+ lz_
#     end
#     z = μ .+ z .* σ
#     z, lz
# end

function (m::FlowNet)(state::AbstractArray, num_samples::Int, na::Int)
    p = m.net(state)
    μ = @inbounds p[1:na,:]
    ρ = @inbounds p[(na+1):(2na),:]
    σ = softplus.(ρ) 
    σ = clamp.(σ, 1f-4, 1000)
    
    z = Zygote.@ignore randn!(similar(μ, size(μ)..., num_samples))
    # μcpu = cpu(μ)
    # r = Zygote.@ignore rand!(similar(μcpu, size(μ)..., num_samples))
    # tn = Zygote.@ignore rand!(TruncatedNormal(0,1,-1,1), similar(μcpu, size(μ)..., num_samples))
    # lap = Zygote.@ignore rand!(Exponential(), similar(μcpu, size(μ)..., num_samples))
    # sig = Zygote.@ignore sign.(rand!(similar(μcpu, size(μ)..., num_samples)) .- 0.5)
    # z = Zygote.@ignore (r .< erfratio) .* tn .+ (r .> erfratio) .* (lap .+ 1) .* sig
    # z = gpu(z)

    lz = Zygote.@ignore fill!(similar(z), 0f0)
    
    μ = reshape(μ, size(μ)..., 1)
    σ = reshape(σ, size(σ)..., 1)
    
    for i=(2na + 1):(3na):(size(p,1) - 3na + 1)
        # b, c, d = MLUtils.chunk(p[i:(i+3na-1), :], 3, dims=1)
        b = @inbounds p[i:(i + na - 1), :]
        b = 4tanh.(b)
        c = @inbounds p[(i+na):(i + 2na - 1), :]
        d = @inbounds p[(i+2na):(i + 3na - 1), :]
        z, lz_ = v2(z, b, c, d)
        lz = lz .+ lz_
    end
    z = μ .+ z .* σ
    z, lz
end

function (m::FlowNet)(samples::AbstractArray, state::AbstractArray, na::Int)
    p = m.net(state)
    μ = @inbounds p[1:na,:]
    ρ = @inbounds p[(na+1):(2na),:]
    σ = softplus.(ρ)
    σ = clamp.(σ, 1f-4, 1000)

    z = samples
    lz = Zygote.@ignore fill!(similar(z), 0f0)

    μ = reshape(μ, size(μ)..., 1)
    σ = reshape(σ, size(σ)..., 1)
    
    z = (z .- μ) ./ σ
    for i=(size(p,1) - 3na + 1):(-3na):(2na + 1)
        b = @inbounds p[i:(i + na - 1), :]
        b = 4tanh.(b)
        c = @inbounds p[(i+na):(i + 2na - 1), :]
        d = @inbounds p[(i+2na):(i + 3na - 1), :]
        # b, c, d = MLUtils.chunk(p[i:(i+3na-1), :], 3, dims=1)
        z, lz_ = v2⁻¹(z, b, c, d)
        lz = lz .+ lz_
    end
    z, lz, μ, σ
end

Base.@kwdef struct FlowNetwork{P,B,F}
    base::B
    prior::P
    flow::F
end

Flux.@functor FlowNetwork

function (m::FlowNetwork)(state::AbstractMatrix; inv::Bool=true)
    h = m.base(state)
    p = m.prior(h)
    μ, ρ = MLUtils.chunk(p, 2, dims=1)
    σ = softplus.(ρ)
    σ = clamp.(σ, 1f-4, 1000)
    # samples = μ .+ randn!(similar(μ)) .* σ
    samples = Zygote.@ignore randn!(similar(μ))
    # samples = Zygote.@ignore randn!(similar(μ, 1, size(μ, 2), num_samples))
    # samples = Zygote.@ignore repeat(samples, size(μ, 1), 1, 1)
    # x1 = rand!(similar(μ))
    # x2 = rand!(similar(μ))
    # samples = μ .+ log.(x1 ./ x2) .* σ
    if !inv
        samples = (samples .- μ) ./ σ
        preds, sldj = inverse(m.flow, samples, h)
        # preds = μ .+ preds .* σ
    else
        # samples = (samples .- μ) ./ σ
        preds, sldj = m.flow(samples, h)
        preds = μ .+ preds .* σ
    end
    return preds, sldj
end

function (m::FlowNetwork)(state::AbstractArray, num_samples::Int; inv::Bool=true)
    h = m.base(state)
    p = m.prior(h)
    μ, ρ = MLUtils.chunk(p, 2, dims=1)
    σ = softplus.(ρ)
    σ = clamp.(σ, 1f-4, 1000)
    # samples = μ .+ randn!(similar(μ, size(μ)..., num_samples)) .* σ
    samples = Zygote.@ignore randn!(similar(μ, size(μ)..., num_samples))
    # samples = Zygote.@ignore randn!(similar(μ, 1, size(μ, 2), num_samples))
    # samples = Zygote.@ignore repeat(samples, size(μ, 1), 1, 1)
    # x1 = rand!(similar(μ, size(μ)..., num_samples))
    # x2 = rand!(similar(μ, size(μ)..., num_samples))
    # samples = μ .+ log.(x1 ./ x2) .* σ
    if !inv
        samples = (samples .- μ) ./ σ
        preds, sldj = inverse(m.flow, samples, h)
        # preds = μ .+ preds .* σ
    else
        # samples = (samples .- μ) ./ σ
        preds, sldj = m.flow(samples, h)
        preds = μ .+ preds .* σ
    end
    return preds, sldj
end

function (m::FlowNetwork)(samples::AbstractArray, state::AbstractArray)
    h = m.base(state)
    p = m.prior(h)
    μ, ρ = MLUtils.chunk(p, 2, dims=1)
    σ = softplus.(ρ)
    σ = clamp.(σ, 1f-4, 1000)
    μ = reshape(μ, size(μ)..., 1)
    σ = reshape(σ, size(σ)..., 1)
    samples = (samples .- μ) ./ σ
    preds, sldj = inverse(m.flow, samples, h)
    return preds, sldj, μ, σ
end

mutable struct QQFLOWLearner{A<:AbstractApproximator} <: AbstractLearner
    approximator::A
    n_actions::Int
    min_replay_history::Int
    n_samples_act::Int
    n_samples_target::Int
    sampler::NStepBatchSampler
    rng::AbstractRNG
    is_enable_double_DQN::Bool
    training::Bool
    logging_params
end

function QQFLOWLearner(;
    approximator::A,
    n_actions::Int,
    stack_size::Union{Int,Nothing}=nothing,
    γ::Real=0.99f0,
    batch_size::Int=32,
    update_horizon::Int=1,
    min_replay_history::Int=100,
    traces=SARTS,
    n_samples_act::Int=30,
    n_samples_target::Int=30,
    is_enable_double_DQN::Bool=false,
    training::Bool=true,
    rng=Random.GLOBAL_RNG
) where {A}
    sampler = NStepBatchSampler{traces}(;
        γ=Float32(γ),
        n=update_horizon,
        stack_size=stack_size,
        batch_size=batch_size
    )
    return QQFLOWLearner(
        approximator,
        n_actions,
        min_replay_history,
        n_samples_act,
        n_samples_target,
        sampler,
        rng,
        is_enable_double_DQN,
        training,
        DefaultDict(0.0),
    )
end

Flux.@functor QQFLOWLearner (approximator,)

function (learner::QQFLOWLearner)(env)
    s = send_to_device(device(learner.approximator), state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    q = dropdims(mean(learner.approximator(s, learner.n_samples_act)[1], dims=3), dims=3)
    vec(q) |> send_to_host
end

# function RLBase.update!(learner::QQFLOWLearner, t::AbstractTrajectory)
#     length(t[:terminal]) - learner.sampler.n <= learner.min_replay_history && return nothing

#     learner.update_step += 1

#     learner.update_step % learner.B_update_freq == 0 || learner.update_step % learner.Q_update_freq == 0 || return nothing

#     if learner.update_step % learner.B_update_freq == 0
#         _, batch = sample(learner.rng, t, learner.sampler)
#         update!(learner, batch)
#     end
#     if learner.update_step % learner.Q_update_freq == 0
#         η = learner.Q_lr
#         B = learner.B_approximator
#         Q = learner.Q_approximator
#         Bp = Flux.params(B)
#         Qp = Flux.params(Q)
#         if η == 1
#             Flux.loadparams!(Q, Bp)
#         else
#             p = Qp .- η .* (Qp .- Bp)
#             Flux.loadparams!(Q, p)
#         end
#     end
# end

function RLBase.optimise!(learner::QQFLOWLearner, batch::NamedTuple)
    A = learner.approximator
    Z = A.model.source
    Zₜ = A.model.target
    n_actions = learner.n_actions
    n_samples_target = learner.n_samples_target
    γ = learner.sampler.γ
    n = learner.sampler.n
    lp = learner.logging_params
    
    D = device(Q)
    states = send_to_device(D, batch.state)
    rewards = send_to_device(D, batch.reward)
    terminals = send_to_device(D, batch.terminal)
    next_states = send_to_device(D, batch.next_state)

    batch_size = length(terminals)
    actions = CartesianIndex.(batch.action, 1:batch_size)

    if learner.is_enable_double_DQN
        q_values = Z(next_states, n_samples_target, n_actions)[1]
    else
        q_values = Zₜ(next_states, n_samples_target, n_actions)[1]
    end

    mean_q = dropdims(mean(q_values, dims=3), dims=3)

    if haskey(batch, :next_legal_actions_mask)
        l′ = send_to_device(D, batch[:next_legal_actions_mask])
        q_values .+= ifelse.(l′, 0.0f0, typemin(Float32))
    end

    selected_actions = dropdims(argmax(mean_q; dims=1); dims=1)
    if learner.is_enable_double_DQN
        q_values = Zₜ(next_states, n_samples_target, n_actions)[1]
    end
    next_q = @inbounds q_values[selected_actions, :]

    target_distribution =
        Flux.unsqueeze(rewards, 2) .+
        Flux.unsqueeze(γ^n .* (1 .- terminals), 2) .* next_q
    target_distribution = repeat(Flux.unsqueeze(target_distribution, 1),
                                 n_actions, 1, 1)

    gs = gradient(params(B)) do
        preds, sldj, μ, σ = Z(target_distribution, states, n_actions)

        nll = preds[actions, :] .^ 2 ./ 2
        
        # abs_error = abs.(TD_error)
        # quadratic = min.(abs_error, 1)
        # linear = abs_error .- quadratic
        # nll = 0.5f0 .* quadratic .* quadratic .+ 1 .* linear

        sldj = sldj[actions, :]
        loss = (sum(nll) - sum(sldj)) / n_samples_target + sum(log.(σ[a, :]))
        loss = loss / batch_size

        Zygote.ignore() do
            lp["loss"] = loss
            lp["nll"] = sum(nll) / (batch_size * n_samples_target)
            lp["sldj"] = sum(sldj) / (batch_size * n_samples_target)
            lp["Qₜ"] = sum(target_distribution) / length(target_distribution)
            lp["QA"] = sum(selected_actions)[1] / length(selected_actions)
            lp["mu"] = sum(μ) / length(μ)
            lp["sigma"] = sum(σ[a,:]) / length(σ[a,:])
            lp["max_weight"] = maximum(maximum.(Flux.params(Z)))
            lp["min_weight"] = minimum(minimum.(Flux.params(Z)))
            lp["max_pred"] = maximum(preds)
            lp["min_pred"] = minimum(preds)
            for i = 1:n_actions
                lp["Q$i"] = sum(G[i,:]) / batch_size
            end
        end

        return 𝐿
    end
    optimise!(B, gs)
end
