export QFLOWLearner
import ReinforcementLearning.RLBase.update!

using DataStructures: DefaultDict
using Distributions: Uniform, Product
using StatsBase: sample
using Flux.Losses
import Statistics.mean

mutable struct QFLOWLearner{
    Tq<:AbstractApproximator,
    Tt<:AbstractApproximator,
    R<:AbstractRNG,
} <: AbstractLearner
    B_approximator::Tq
    Q_approximator::Tt
    flow
    Q_lr::Float32
    min_replay_history::Int
    B_update_freq::Int
    Q_update_freq::Int
    update_step::Int
    sampler::NStepBatchSampler
    rng::R
    is_enable_double_DQN::Bool
    training::Bool
    logging_params
end

function QFLOWLearner(;
    B_approximator::Tq,
    Q_approximator::Tt,
    flow,
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
    return QFLOWLearner(
        B_approximator,
        Q_approximator,
        flow,
        Float32(Q_lr),
        min_replay_history,
        B_update_freq,
        Q_update_freq,
        update_step,
        sampler,
        rng,
        is_enable_double_DQN,
        training,
        DefaultDict(0.0),
    )
end

Flux.functor(x::QFLOWLearner) = (B=x.B_approximator, Q=x.Q_approximator),
y -> begin
    x = @set x.B_approximator = y.B
    x = @set x.Q_approximator = y.Q
    x
end

function (learner::QFLOWLearner)(env)
    s = send_to_device(device(learner.B_approximator), state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    q = learner.B_approximator(s)
    vec(q) |> send_to_host
end

function RLBase.update!(learner::QFLOWLearner, t::AbstractTrajectory)
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

function RLBase.update!(learner::QFLOWLearner, batch::NamedTuple)
    B = learner.B_approximator
    Q = learner.Q_approximator
    γ = learner.sampler.γ
    n = learner.sampler.n
    flow = learner.flow
    batch_size = learner.sampler.batch_size
    is_enable_double_DQN = learner.is_enable_double_DQN
    D = device(Q)

    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    a = CartesianIndex.(a, 1:batch_size)

    if is_enable_double_DQN
        q_values = B(s′)
    else
        q_values = Q(s′)
    end

    if haskey(batch, :next_legal_actions_mask)
        l′ = send_to_device(D, batch[:next_legal_actions_mask])
        q_values .+= ifelse.(l′, 0.0f0, typemin(Float32))
    end

    if is_enable_double_DQN
        selected_actions = dropdims(argmax(q_values; dims=1); dims=1)
        q′ = Q(s′)
        q′ = @inbounds q′[selected_actions]
    else
        q′ = dropdims(maximum(q_values; dims=1); dims=1)
    end
    G = r .+ γ^n .* (1 .- t) .* q′

    gs = gradient(params(B)) do
        b_all, h_all = B(s) ## SLOW
        h_all = reshape(h_all, size(b_all, 1), :, size(b_all, 2))
        h_all = permutedims(h_all, (1, 3, 2))
        h = @inbounds reshape(h_all, size(b_all, 1), size(b_all, 2), :)[a, :]
        h = permutedims(h, (2, 1))
        b = @inbounds b_all[a]
        # ss = @inbounds s_all[a]
        # preds, sldj = flow(Flux.unsqueeze(G, 1), h)
        ll = huber_loss(b, G)
        # ll = (b .- G) .^ 2 ./ 2
        # ll = (b .- preds) .^ 2 ./ 2
        # 𝐿 = sum(ll) - sum(sldj) + sum((preds .- G) .^ 2) / 2
        𝐿 = sum(ll) #- sum(sldj) + sum((preds .- G) .^ 2) / 2
        𝐿 = 𝐿 / batch_size

        Zygote.ignore() do
            # learner.logging_params["s"] = sum(ss) / length(ss)
            learner.logging_params["𝐿"] = 𝐿
            learner.logging_params["Qₜ"] = sum(G) / length(G)
            # learner.logging_params["J"] = sum(sldj) / batch_size
        end

        return 𝐿
    end
    update!(B, gs)
end
