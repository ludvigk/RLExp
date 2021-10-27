export GDQNLearner
import ReinforcementLearning.RLBase.update!

using CUDA: randn!
using Random: randn!
using StatsBase: sample

struct NoisyDense2
    w_μ::AbstractMatrix
    w_ρ::AbstractMatrix
    b_μ::AbstractVector
    b_ρ::AbstractVector
    f::Any
end

function NoisyDense2(
    in, out, f=identity; init_μ=glorot_uniform(), init_σ=(dims...) -> fill(0.0017f0, dims)
)
    return NoisyDense2(
        init_μ(out, in),
        log.(exp.(init_σ(out, in)) .- 1),
        init_μ(out),
        log.(exp.(init_σ(out)) .- 1),
        f,
    )
end

Flux.@functor NoisyDense2

function (l::NoisyDense2)(x)
    x = ndims(x) == 2 ? unsqueeze(x, 3) : x
    tmp_x = reshape(x, size(x, 1), :)
    μ = l.w_μ * tmp_x .+ l.b_μ
    σ² = softplus.(l.w_ρ) * tmp_x .^ 2 .+ softplus.(l.b_ρ)
    ϵ = Zygote.@ignore randn!(similar(μ, size(μ, 1), 1, 100))
    μ = reshape(μ, size(μ, 1), size(x, 2), :)
    σ² = reshape(σ², size(μ, 1), size(x, 2), :)
    return y = l.f.(μ .+ ϵ .* sqrt.(σ²))
end


mutable struct GDQNLearner{
    Tq<:AbstractApproximator,
    Tt<:AbstractApproximator,
    Tf,
    R<:AbstractRNG,
} <: AbstractLearner
    approximator::Tq
    target_approximator::Tt
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

function GDQNLearner(;
    approximator::Tq,
    target_approximator::Tt,
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
    return GDQNLearner(
        approximator,
        target_approximator,
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

Flux.functor(x::GDQNLearner) = (Q = x.approximator, Qₜ = x.target_approximator),
y -> begin
    x = @set x.approximator = y.Q
    x = @set x.target_approximator = y.Qₜ
    x
end

function (learner::GDQNLearner)(env)
    s = send_to_device(device(learner), state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    return vec(mean(reshape(learner.approximator(s), :, 100), dims=2)) |> send_to_host
end

function RLBase.update!(learner::GDQNLearner, t::AbstractTrajectory)
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

function RLBase.update!(learner::GDQNLearner, batch::NamedTuple)
    Q = learner.approximator
    Qₜ = learner.target_approximator
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

    q_values = reshape(q_values, :, 100)
    G = r .+ γ^n .* (1 .- t) .* q′
    G = reshape(G, :, 100)

    gs = gradient(params([Q, Σ])) do
        q_ = Q(s) 
        q = q_[a, :]
        nll = cross_entropy_surrogate(learner.sse, permutedims(q, (2,1)), permutedims(G, (2,1)))

        noisy_q = reshape(q_, :, 100) + learner.injected_noise * randn!(similar(noisy_q))
        ent = entropy_surrogate(learner.sse, permutedims(noisy_q, (2, 1)))

        Zygote.ignore() do
            learner.loss = nll - ent / learner.sampler.batch_size
            learner.q_var = mean(var(cpu(q); dims = 2))
            learner.nll = nll
            learner.ent = ent / learner.sampler.batch_size
            learner.σ = mean(cpu(exp.(σ)))
        end
        return nll - ent / learner.sampler.batch_size
    end
    update!(Σ, gs)
    return update!(Q, gs)
end
