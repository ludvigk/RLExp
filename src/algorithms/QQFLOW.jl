export QQFLOWLearner, FlowNetwork
import ReinforcementLearning.RLBase.update!

using DataStructures: DefaultDict
using Distributions: Uniform, Product
using StatsBase: sample
using Flux.Losses
using CUDA: randn
using MLUtils
import Statistics.mean

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
    samples = Zygote.@ignore randn!(similar(μ, 1, size(μ, 2), size(μ,3)))
    samples = Zygote.@ignore repeat(samples, size(μ, 1), 1, 1)
    # x1 = rand!(similar(μ))
    # x2 = rand!(similar(μ))
    # samples = μ .+ log.(x1 ./ x2) .* σ
    if inv
        preds, sldj = inverse(m.flow, samples, h)
        preds = μ .+ preds .* σ
    else
        samples = (samples .- μ) ./ σ
        preds, sldj = m.flow(samples, h)
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
    samples = randn!(similar(μ, size(μ)..., num_samples))
    # x1 = rand!(similar(μ, size(μ)..., num_samples))
    # x2 = rand!(similar(μ, size(μ)..., num_samples))
    # samples = μ .+ log.(x1 ./ x2) .* σ
    if inv
        preds, sldj = inverse(m.flow, samples, h)
        preds = μ .+ preds .* σ
    else
        samples = (samples .- μ) ./ σ
        preds, sldj = m.flow(samples, h)
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
    preds, sldj = m.flow(samples, h)
    return preds, sldj, μ, σ
end

mutable struct QQFLOWLearner{
    Tq<:AbstractApproximator,
    Tt<:AbstractApproximator,
    R<:AbstractRNG,
} <: AbstractLearner
    B_approximator::Tq
    Q_approximator::Tt
    num_actions::Int
    Q_lr::Float32
    min_replay_history::Int
    B_update_freq::Int
    Q_update_freq::Int
    update_step::Int
    n_samples_act::Int
    n_samples_target::Int
    sampler::NStepBatchSampler
    rng::R
    is_enable_double_DQN::Bool
    training::Bool
    logging_params
end

function QQFLOWLearner(;
    B_approximator::Tq,
    Q_approximator::Tt,
    num_actions::Int,
    Q_lr::Real=0.01f0,
    stack_size::Union{Int,Nothing}=nothing,
    γ::Real=0.99f0,
    batch_size::Int=32,
    update_horizon::Int=1,
    min_replay_history::Int=100,
    B_update_freq::Int=1,
    Q_update_freq::Int=1,
    traces=SARTS,
    update_step::Int=0,
    n_samples_act::Int=30,
    n_samples_target::Int=30,
    is_enable_double_DQN::Bool=false,
    training::Bool=true,
    rng=Random.GLOBAL_RNG
) where {Tq,Tt}
    sampler = NStepBatchSampler{traces}(;
        γ=Float32(γ),
        n=update_horizon,
        stack_size=stack_size,
        batch_size=batch_size
    )
    return QQFLOWLearner(
        B_approximator,
        Q_approximator,
        num_actions,
        Float32(Q_lr),
        min_replay_history,
        B_update_freq,
        Q_update_freq,
        update_step,
        n_samples_act,
        n_samples_target,
        sampler,
        rng,
        is_enable_double_DQN,
        training,
        DefaultDict(0.0),
    )
end

Flux.functor(x::QQFLOWLearner) = (B=x.B_approximator, Q=x.Q_approximator),
y -> begin
    x = @set x.B_approximator = y.B
    x = @set x.Q_approximator = y.Q
    x
end

function (learner::QQFLOWLearner)(env)
    s = send_to_device(device(learner.B_approximator), state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    q = dropdims(mean(learner.B_approximator(s, learner.n_samples_act)[1], dims=3), dims=3)
    vec(q) |> send_to_host
end

function RLBase.update!(learner::QQFLOWLearner, t::AbstractTrajectory)
    length(t[:terminal]) - learner.sampler.n <= learner.min_replay_history && return nothing

    learner.update_step += 1

    learner.update_step % learner.B_update_freq == 0 || learner.update_step % learner.Q_update_freq == 0 || return nothing

    if learner.update_step % learner.B_update_freq == 0
        _, batch = sample(learner.rng, t, learner.sampler)
        update!(learner, batch)
    end
    if learner.update_step % learner.Q_update_freq == 0
        η = learner.Q_lr
        B = learner.B_approximator
        Q = learner.Q_approximator
        Bp = Flux.params(B)
        Qp = Flux.params(Q)
        if η == 1
            Flux.loadparams!(Q, Bp)
        else
            p = Qp .- η .* (Qp .- Bp)
            Flux.loadparams!(Q, p)
        end
    end
end

function RLBase.update!(learner::QQFLOWLearner, batch::NamedTuple)
    B = learner.B_approximator
    Q = learner.Q_approximator
    num_actions = learner.num_actions
    n_samples_target = learner.n_samples_target
    γ = learner.sampler.γ
    n = learner.sampler.n
    batch_size = learner.sampler.batch_size
    D = device(Q)

    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    a = CartesianIndex.(a, 1:batch_size)

    if learner.is_enable_double_DQN
        q_values = B(s′, n_samples_target)[1]
    else
        q_values = Q(s′, n_samples_target)[1]
    end

    mean_q = dropdims(mean(q_values, dims=3), dims=3)


    if haskey(batch, :next_legal_actions_mask)
        l′ = send_to_device(D, batch[:next_legal_actions_mask])
        q_values .+= ifelse.(l′, 0.0f0, typemin(Float32))
    end

    selected_actions = dropdims(argmax(mean_q; dims=1); dims=1)
    if learner.is_enable_double_DQN
        q_values = Q(s′, n_samples_target)[1]
    end
    q′ = @inbounds q_values[selected_actions, :]

    G = Flux.unsqueeze(r, 2) .+ Flux.unsqueeze(γ^n .* (1 .- t), 2) .* q′
    G = repeat(Flux.unsqueeze(G, 1), num_actions, 1, 1)
    # G_in = similar(G, num_actions, size(G)...)
    # fill!(G_in, 0)
    # G_in[selected_actions] = G
    # G = G_in

    gs = gradient(params(B)) do
        preds, sldj, μ, σ = B(G, s)
        # σ = clamp.(σ, 1f-2, 1f4)
        # p = (preds .- μ) .^ 2 ./ (2 .* σ .^ 2 .+ 1f-6)
        ll = preds[a, :] .^ 2 ./ 2
        # p = ((preds .- μ) ./ σ)[a,:]
        # ll = min.(abs.(p), p .^ 2)
        # ll = p[a,:]
        # ll = min.(p .^ 2, p)
        sldj = sldj[a, :]
        𝐿 = (sum(ll) - sum(sldj)) / n_samples_target + sum(log.(σ[a, :]))

        sqnorm(x) = sum(abs2, x)
        l2norm = sum(sqnorm, Flux.params(B))
        
        𝐿 = 𝐿 / batch_size #+ 1f-5 * l2norm

        Zygote.ignore() do
            learner.logging_params["𝐿"] = 𝐿
            learner.logging_params["nll"] = sum(ll) / (batch_size * n_samples_target)
            learner.logging_params["sldj"] = sum(sldj) / (batch_size * n_samples_target)
            learner.logging_params["Qₜ"] = sum(G) / length(G)
            learner.logging_params["QA"] = sum(selected_actions)[1] / length(selected_actions)
            learner.logging_params["mu"] = sum(μ) / length(μ)
            learner.logging_params["sigma"] = sum(σ[a,:]) / length(σ[a,:])
            learner.logging_params["l2norm"] = l2norm
            learner.logging_params["max_weight"] = maximum(maximum.(Flux.params(B)))
            learner.logging_params["min_weight"] = minimum(minimum.(Flux.params(B)))
            learner.logging_params["max_pred"] = maximum(preds)
            learner.logging_params["min_pred"] = minimum(preds)
            for i = 1:learner.num_actions
                learner.logging_params["Q$i"] = sum(G[i,:]) / batch_size
            end
        end

        return 𝐿
    end
    update!(B, gs)
end
