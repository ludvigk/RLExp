export QQFLOWLearner, FlowNetwork
import ReinforcementLearning.RLBase.update!

using DataStructures: DefaultDict
using Distributions: Uniform, Product
using StatsBase: sample
using Flux.Losses
import Statistics.mean

Base.@kwdef struct FlowNetwork{B,F}
    base::B
    flow::F
end

Flux.@functor FlowNetwork

function (m::FlowNetwork)(samples::AbstractMatrix, state::AbstractMatrix; action=nothing, reverse::Bool=true)
    h = m.base(state)
    return m.flow(samples, h; action, reverse)
end

function (m::FlowNetwork)(samples::AbstractArray{T,3}, state::AbstractMatrix; action=nothing, reverse::Bool=true) where {T}
    h = m.base(state)
    return m.flow(samples, h; action, reverse)
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
    norm_samples = send_to_device(device(learner.B_approximator), randn(learner.num_actions, size(s, 2), learner.n_samples_act))
    q = dropdims(mean(learner.B_approximator(norm_samples, s; reverse=true), dims=3), dims=3)
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
    is_enable_double_DQN = learner.is_enable_double_DQN
    D = device(Q)

    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    a = CartesianIndex.(a, 1:batch_size)

    norm_samples = send_to_device(device(B), randn(num_actions, batch_size, n_samples_target))
    if is_enable_double_DQN
        q_values = dropdims(mean(B(norm_samples, s′; reverse=true), dims=3), dims=3)
    else
        q_values = dropdims(mean(Q(norm_samples, s′; reverse=true), dims=3), dims=3)
    end

    if haskey(batch, :next_legal_actions_mask)
        l′ = send_to_device(D, batch[:next_legal_actions_mask])
        q_values .+= ifelse.(l′, 0.0f0, typemin(Float32))
    end

    if is_enable_double_DQN
        selected_actions = dropdims(argmax(q_values; dims=1); dims=1)
        q′ = dropdims(mean(Q(norm_samples, s′; reverse=true), dims=3), dims=3)
        # q′ = @inbounds q′[]
    else
        q′ = dropdims(maximum(q_values; dims=1); dims=1)
        q′ = q_values
    end
    println(size(q′), size(r))
    G = Flux.unsqueeze(r .+ γ^n .* (1 .- t), 1) .* q′

    gs = gradient(params(B)) do
        preds, sldj = B(G, s; action=selected_actions, reverse=false)
        ll = preds[selected_actions] .^ 2 ./ 2
        𝐿 = sum(ll) - sum(sldj)
        𝐿 = 𝐿 / batch_size

        Zygote.ignore() do
            learner.logging_params["𝐿"] = 𝐿
            learner.logging_params["Qₜ"] = sum(G) / length(G)
        end

        return 𝐿
    end
    update!(B, gs)
end
