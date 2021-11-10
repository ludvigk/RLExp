export DUQNLearner
import ReinforcementLearning.RLBase.update!

using DataStructures: DefaultDict
using Distributions: Uniform, Product
using Random: randn!
using StatsBase: sample
import Statistics.mean

Statistics.mean(a::CuArray) = sum(a) / length(a)
Statistics.mean(a::CuArray, dims) = sum(a, dims) / prod(size(a, dims))


mutable struct DUQNLearner{
    Tq<:AbstractApproximator,
    Tt<:AbstractApproximator,
    R<:AbstractRNG,
} <: AbstractLearner
    B_approximator::Tq
    Q_approximator::Tt
    min_replay_history::Int
    B_update_freq::Int
    Q_update_freq::Int
    update_step::Int
    sampler::NStepBatchSampler
    rng::R
    sse::SpectralSteinEstimator
    injected_noise::Float32
    n_samples::Int
    # for logging
    logging_params
end

function DUQNLearner(;
    B_approximator::Tq,
    Q_approximator::Tt,
    stack_size::Union{Int,Nothing} = nothing,
    γ::Float32 = 0.99f0,
    batch_size::Int = 32,
    update_horizon::Int = 1,
    min_replay_history::Int = 32,
    B_update_freq::Int = 1,
    Q_update_freq::Int = 1,
    traces = SARTS,
    update_step::Int = 0,
    injected_noise::Float32 = 0.01f0,
    n_samples::Int = 100,
    η::Float32 = 0.05f0,
    n_eigen_threshold::Float32 = 0.99f0,
    rng = Random.GLOBAL_RNG,
) where {Tq,Tt}
    sampler = NStepBatchSampler{traces}(;
        γ = γ,
        n = update_horizon,
        stack_size = stack_size,
        batch_size = batch_size,
    )
    return DUQNLearner(
        B_approximator,
        Q_approximator,
        min_replay_history,
        B_update_freq,
        Q_update_freq,
        update_step,
        sampler,
        rng,
        SpectralSteinEstimator(η, nothing, n_eigen_threshold),
        injected_noise,
        n_samples,
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

    for _=1:20
    _, batch = sample(learner.rng, t, learner.sampler)
    update!(learner, batch)
    end
end

function RLBase.update!(learner::DUQNLearner, batch::NamedTuple)
    B = learner.B_approximator
    Q = learner.Q_approximator
    sse =learner.sse
    γ = learner.sampler.γ
    n = learner.sampler.n
    n_samples = learner.n_samples
    batch_size = learner.sampler.batch_size
    D = device(Q)

    s, a, r, t, s′ = (send_to_device(D, batch[x]) for x in SARTS)
    a = CartesianIndex.(a, 1:batch_size)

    if learner.update_step % learner.B_update_freq == 0
        q_values = Q(s′)
        
        if haskey(batch, :next_legal_actions_mask)
            l′ = send_to_device(D, batch[:next_legal_actions_mask])
            q_values .+= ifelse.(l′, 0.0f0, typemin(Float32))
        end
        
        q′ = dropdims(maximum(q_values; dims = 1); dims = 1)
        G = r .+ γ^n .* (1 .- t) .* q′

        # u = Product([Uniform(-2.4f0, 2.4f0),
        #              Uniform(-10.0f0, 10.0f0),
        #              Uniform(-0.418f0, 0.418f0),
        #              Uniform(-10.0f0, 10.0f0)])
        # samples = rand(u, 5)


        # samples_s = cat(repeat(s, 2, 1, n_samples), repeat(samples, 2, 1, n_samples); dims = 1)
        # samples_s = reshape(samples_s, :, n_samples)
        rs = repeat(s, 1, 2, 100)
        
        gs = gradient(params(B)) do
            b_all = B(s, n_samples)
            b = b_all[1][a, :]
            B̂ = dropdims(mean(b, dims=ndims(b)), dims=ndims(b))
            σ = b_all[2][a, :]
            Σ = dropdims(mean(σ, dims=ndims(σ)), dims=ndims(σ))
            # Σ = 0.01

            𝐿 = sum(log.(Σ) .+ (G .- B̂) .^ 2 ./ 2Σ .^ 2)
            
            b_rand = reshape(b_all[1], :, n_samples)
            S = entropy_surrogate(sse, permutedims(b_rand, (2, 1)))
            # pr = [-1 0 1] .* s[2,:]
            # pr = reshape(repeat(pr, 1,  100), :, 100)
            # H = sum((b_rand .- 100 .* pr) .^ 2 ./ (2 * 100.0f0 .^ 2)) ./ (size(b_rand, 2) .* batch_size)
            H = 0
            KL = H - S

            Zygote.ignore() do
                learner.logging_params["KL"] = KL
                learner.logging_params["H"] = H
                learner.logging_params["S"] = S
                learner.logging_params["𝐿"] = 𝐿
                learner.logging_params["Q"] = mean(B̂)
                learner.logging_params["Σ"] = mean(Σ)
            end

            return 𝐿 + KL / batch_size
        end
        
        update!(B, gs)
    end
    if learner.update_step % learner.Q_update_freq == 0
        b = B(s, n_samples)[a, :]
        B̂ = dropdims(mean(b, dims=ndims(b)), dims=ndims(b))
        
        gs = gradient(params(Q)) do
            q = Q(s)[a]
            𝐿 = sum((q .- B̂) .^ 2)
            Zygote.ignore() do
                learner.logging_params["mse"] = 𝐿
            end
            return 𝐿
        end
        
        update!(Q, gs)
    end
end