export DUQNLearner, FlatPrior, GeneralPrior, GaussianPrior
import ReinforcementLearning.RLBase.update!

using DataStructures: DefaultDict
using Distributions: Uniform, Product
using StatsBase: sample
import Statistics.mean

Statistics.mean(a::CuArray) = sum(a) / length(a)
Statistics.mean(a::CuArray, dims) = sum(a, dims) / prod(size(a, dims))


abstract type AbstractPrior end

struct FlatPrior <: AbstractPrior end
(p::FlatPrior)(s::AbstractArray, t::AbstractArray) = zero(eltype(t))

struct GeneralPrior{F} <: AbstractPrior
    f::F
end
(p::GeneralPrior)(s::AbstractArray, t::AbstractArray) = p.f(s, t)

struct GaussianPrior <: AbstractPrior
    μ
    σ
end
(p::GaussianPrior)(_, t::AbstractArray) = sum((t .- p.μ) .^ 2 ./ (2p.σ .^ 2))

abstract type AbstractMeasure end

mutable struct DUQNLearner{
    Tq<:AbstractApproximator,
    Tt<:AbstractApproximator,
    P<:AbstractPrior,
    M<:Union{AbstractMeasure, Nothing},
    R<:AbstractRNG,
} <: AbstractLearner
    B_approximator::Tq
    Q_approximator::Tt
    Q_lr::Float32
    prior::P
    λ::Union{Float32, Nothing}
    min_replay_history::Int
    B_update_freq::Int
    Q_update_freq::Int
    updates_per_step::Int
    update_step::Int
    sampler::NStepBatchSampler
    rng::R
    sse::SpectralSteinEstimator
    injected_noise::Float32
    n_samples::Int
    is_enable_double_DQN::Bool
    measure::M
    training::Bool
    logging_params
end

function DUQNLearner(;
    B_approximator::Tq,
    Q_approximator::Tt,
    Q_lr::Real = 0.01f0,
    prior::AbstractPrior = FlatPrior(),
    λ::Union{Real, Nothing} = 1.0,
    stack_size::Union{Int, Nothing} = nothing,
    γ::Real = 0.99f0,
    batch_size::Int = 32,
    update_horizon::Int = 1,
    min_replay_history::Int = 100,
    B_update_freq::Int = 1,
    Q_update_freq::Int = 1,
    updates_per_step::Int = 1,
    traces = SARTS,
    update_step::Int = 0,
    injected_noise::Real = 0.01f0,
    n_samples::Int = 100,
    η::Real = 0.05f0,
    nev::Int = 10,
    is_enable_double_DQN::Bool = false,
    measure::Union{M, Nothing} = nothing,
    training::Bool = true,
    rng = Random.GLOBAL_RNG,
) where {Tq,Tt,M}
    sampler = NStepBatchSampler{traces}(;
        γ = Float32(γ),
        n = update_horizon,
        stack_size = stack_size,
        batch_size = batch_size,
    )
    return DUQNLearner(
        B_approximator,
        Q_approximator,
        Float32(Q_lr),
        prior,
        Float32(λ),
        min_replay_history,
        B_update_freq,
        Q_update_freq,
        updates_per_step,
        update_step,
        sampler,
        rng,
        SpectralSteinEstimator(Float32(η), nev, nothing),
        Float32(injected_noise),
        n_samples,
        is_enable_double_DQN,
        measure,
        training,
        DefaultDict(0.0),
    )
end

Flux.functor(x::DUQNLearner) = (B = x.B_approximator, Q = x.Q_approximator),
y -> begin
    x = @set x.B_approximator = y.B
    x = @set x.Q_approximator = y.Q
    x
end

function (learner::DUQNLearner)(env)
    s = send_to_device(device(learner.B_approximator), state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    q = learner.B_approximator(s)
    vec(q) |> send_to_host
end

function RLBase.update!(learner::DUQNLearner, t::AbstractTrajectory)
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
        p = Qp .- η .* (Qp .- Bp)
        # for _=1:(learner.updates_per_step-1)
        #     p = p .- η .* (p .- Bp)
        # end
        Flux.loadparams!(Q, p)
    end
end

function RLBase.update!(learner::DUQNLearner, batch::NamedTuple)
    B = learner.B_approximator
    Q = learner.Q_approximator
    sse = learner.sse
    γ = learner.sampler.γ
    n = learner.sampler.n
    n_samples = learner.n_samples
    batch_size = learner.sampler.batch_size
    is_enable_double_DQN = learner.is_enable_double_DQN
    D = device(Q)

    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    a = CartesianIndex.(a, 1:batch_size)

    seed = hash(rand())
    rng_B = Random.MersenneTwister(seed)
    rng_Q = Random.MersenneTwister(seed + 1)

    if is_enable_double_DQN
        # q_values = B(s′, n_samples, rng = rng_B)
        q_values = B(s′, n_samples)
        rng_B = Random.MersenneTwister(seed)
    else
        q_values = Q(s′, n_samples, rng = rng_Q)
    end

    if haskey(batch, :next_legal_actions_mask)
        l′ = send_to_device(D, batch[:next_legal_actions_mask])
        q_values .+= ifelse.(l′, 0.0f0, typemin(Float32))
    end

    if is_enable_double_DQN
        selected_actions = dropdims(argmax(q_values; dims = 1); dims = 1)
        q′ = Q(s′, n_samples; rng = rng_Q)[selected_actions, :]
        q′ = dropdims(q′, dims=ndims(q′))
    else
        q′ = dropdims(maximum(q_values; dims = 1); dims = 1)
    end
    G = r .+ γ^n .* (1 .- t) .* q′

    gs = gradient(params(B)) do
        b_all = B(s, n_samples, rng = rng_B) ## SLOW
        b = b_all[a, :]
        B̂ = dropdims(mean(b, dims=ndims(b)), dims=ndims(b))
        λ = learner.λ
        𝐿 = sum((b .- G) .^ 2) / n_samples

        b_rand = reshape(b_all, :, n_samples) ## SLOW

        S = entropy_surrogate(sse, permutedims(b_rand, (2, 1)))
        H = learner.prior(s, b_rand) ./ n_samples

        KL = H - S

        Zygote.ignore() do
            learner.logging_params["KL"] = KL
            learner.logging_params["H"] = H
            learner.logging_params["S"] = S
            learner.logging_params["𝐿"] = 𝐿
            learner.logging_params["Q"] = mean(B̂)
            learner.logging_params["Qₜ"] = mean(G)
            learner.logging_params["B_var"] = mean(var(b, dims=ndims(b)))
            learner.logging_params["QA"] = mean(getindex.(a, 1))
        end

        return 𝐿 + λ * KL / batch_size
    end
    update!(B, gs)
end
