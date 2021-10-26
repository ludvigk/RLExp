export BDQNLearner
import ReinforcementLearning.RLBase.update!

using CUDA: randn!
using Random: randn!
using StatsBase: sample

mutable struct BDQNLearner{
    Tq<:AbstractApproximator,
    Tt<:AbstractApproximator,
    Tv<:AbstractApproximator,
    Tf,
    R<:AbstractRNG,
} <: AbstractLearner
    approximator::Tq
    target_approximator::Tt
    variance_approximator::Tv
    loss_func::Tf
    min_replay_history::Int
    update_freq::Int
    update_step::Int
    target_update_freq::Int
    sampler::NStepBatchSampler
    rng::R
    sse::SpectralSteinEstimator
    injected_noise::Float32
    n_samples::Int
    is_enable_double_DQN::Bool
    # for logging
    loss::Float32
    kl::Float32
    q_var::Float32
    nll::Float32
    σ::Float32
end

function BDQNLearner(;
    approximator::Tq,
    target_approximator::Tt,
    variance_approximator::Tv,
    loss_func::Tf,
    stack_size::Union{Int,Nothing} = nothing,
    γ::Float32 = 0.99f0,
    batch_size::Int = 32,
    update_horizon::Int = 1,
    min_replay_history::Int = 32,
    update_freq::Int = 1,
    target_update_freq::Int = 100,
    traces = SARTS,
    update_step::Int = 0,
    injected_noise::Float32 = 0.01f0,
    n_samples::Int = 100,
    η::Float32 = 0.05f0,
    n_eigen_threshold::Float32 = 0.99f0,
    rng = Random.GLOBAL_RNG,
    is_enable_double_DQN::Bool = true,
) where {Tq,Tt,Tv,Tf}
    copyto!(approximator, target_approximator)
    sampler = NStepBatchSampler{traces}(;
        γ = γ,
        n = update_horizon,
        stack_size = stack_size,
        batch_size = batch_size,
    )
    return BDQNLearner(
        approximator,
        target_approximator,
        variance_approximator,
        loss_func,
        min_replay_history,
        update_freq,
        update_step,
        target_update_freq,
        sampler,
        rng,
        SpectralSteinEstimator(η, nothing, n_eigen_threshold),
        injected_noise,
        n_samples,
        is_enable_double_DQN,
        0.0f0,
        0.0f0,
        0.0f0,
        0.0f0,
        0.0f0,
    )
end

Flux.functor(x::BDQNLearner) = (Q = x.approximator, Qₜ = x.target_approximator),
y -> begin
    x = @set x.approximator = y.Q
    x = @set x.target_approximator = y.Qₜ
    x
end

function (learner::BDQNLearner)(env)
    env |>
    state |>
    x ->
        Flux.unsqueeze(x, ndims(x) + 1) |>
        x ->
            send_to_device(device(learner), x) |>
            learner.approximator |>
            vec |>
            send_to_host
end

function RLBase.update!(learner::BDQNLearner, t::AbstractTrajectory)
    length(t[:terminal]) - learner.sampler.n <= learner.min_replay_history && return nothing

    learner.update_step += 1

    if learner.update_step % learner.target_update_freq == 0
        copyto!(learner.target_approximator, learner.approximator)
    end

    learner.update_step % learner.update_freq == 0 || return nothing

    inds, batch = sample(learner.rng, t, learner.sampler)

    if t isa PrioritizedTrajectory
        priorities = update!(learner, batch)
        t[:priority][inds] .= priorities
    else
        update!(learner, batch)
    end
end

function RLBase.update!(learner::BDQNLearner, batch::NamedTuple)
    Q = learner.approximator
    Qₜ = learner.target_approximator
    Σ = learner.variance_approximator
    γ = learner.sampler.γ
    n = learner.sampler.n
    batch_size = learner.sampler.batch_size
    is_enable_double_DQN = learner.is_enable_double_DQN
    D = device(Q)

    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    a = CartesianIndex.(a, 1:batch_size)

    if is_enable_double_DQN
        q_values = Q(s′)
    else
        q_values = Qₜ(s′)
    end

    if haskey(batch, :next_legal_actions_mask)
        l′ = send_to_device(D, batch[:next_legal_actions_mask])
        q_values .+= ifelse.(l′, 0.0f0, typemin(Float32))
    end

    if is_enable_double_DQN
        selected_actions = dropdims(argmax(q_values; dims = 1); dims = 1)
        q′ = Qₜ(s′)[selected_actions]
    else
        q′ = dropdims(maximum(q_values; dims = 1); dims = 1)
    end

    G = r .+ γ^n .* (1 .- t) .* q′

    noise = rand!(similar(s))[:,:,:,1:5]

    gs = gradient(params([Q, Σ])) do
        noise_q = Q(noise)
        q = Q(s)[a, :]
        G_ = repeat(G, 1, 100)
        σ = Σ(s)[a, :]
        # σ = 0.001f0
        nll = sum(prod(size(G)) .* σ .+ sum((G_ .- q) .^ 2 ./ (2 .* exp.(σ).^2)))
        nll = nll ./ (100 .* learner.sampler.batch_size)

        q = reshape(q, :, 100)
        noise_q = reshape(noise_q, :, 100)
        noisy_q = cat(q, noise_q; dims = 1)
        noisy_q = noisy_q + learner.injected_noise * randn!(similar(noisy_q))
        ent = entropy_surrogate(learner.sse, permutedims(noisy_q, (2, 1)))
        const_term = size(noisy_q, 2) * log(2π * 5 ^ 2) / 2
        ce = const_term .+ sum(noisy_q .^ 2 ./ (size(noisy_q, 2) * 2 * 5.0f0 .^ 2))
        kl = -ent + ce

        Zygote.ignore() do
            learner.loss = nll .+ kl / learner.sampler.batch_size
            learner.q_var = mean(var(cpu(q); dims = 2))
            learner.nll = nll
            learner.kl = kl / learner.sampler.batch_size
            learner.σ = mean(cpu(exp.(σ)))
        end
        return nll + kl / learner.sampler.batch_size
    end
    update!(Σ, gs)
    return update!(Q, gs)
end
