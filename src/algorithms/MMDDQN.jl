export MMDDQNLearner

using ChainRulesCore: ignore_derivatives
using Random: GLOBAL_RNG, AbstractRNG
using StatsBase: mean
using Functors: @functor

Base.@kwdef mutable struct MMDDQNLearner{A<:Approximator{<:TwinNetwork}} <: AbstractLearner
    approximator::A
    n_quantile::Int
    γ::Float32 = 0.99f0
    rng::AbstractRNG = GLOBAL_RNG
    # for recording
    loss::Float32 = 0.0f0
    logging_params = DefaultDict(0.0)
end

@functor MMDDQNLearner (approximator,)

function (learner::MMDDQNLearner)(s::AbstractArray)
    batch_size = size(s)[end]
    # τ = rand(learner.device_rng, Float32, learner.K, batch_size)

    # τₑₘ = embed(τ, learner.Nₑₘ)
    quantiles = learner.approximator(s)
    quantiles = reshape(quantiles, :, learner.n_quantile, batch_size)
    dropdims(mean(quantiles; dims=2); dims=2)
end

function (L::MMDDQNLearner)(env::AbstractEnv)
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

function l2_norm(td, κ)
    return td .^ 2
end

function energy_distance(x, y; κ=1f0)
    n = size(x, 2)
    m = size(y, 2)

    x_ = Flux.unsqueeze(x, dims=1)
    _x = Flux.unsqueeze(x, dims=2)
    _y = Flux.unsqueeze(y, dims=2)
    d_xy = dropdims(sum(l2_norm(x_ .- _y, κ), dims=(1,2)), dims=(1,2))
    d_xx = dropdims(sum(l2_norm(x_ .- _x, κ), dims=(1,2)), dims=(1,2))
    ε = 2 / (n * m) .* d_xy .- 1 / n^2 .* d_xx
    return ε
end

function RLBase.optimise!(learner::MMDDQNLearner, batch::NamedTuple)
    A = learner.approximator
    Q = A.model.source
    Qₜ = A.model.target
    γ = learner.γ
    N = learner.n_quantile
    # loss_func = learner.loss_func
    D = device(Q)

    s, s′, a, r, t = map(x -> batch[x], SS′ART)
    s = send_to_device(D, collect(batch.state))
    r = send_to_device(D, batch.reward)
    t = send_to_device(D, batch.terminal)
    s′ = send_to_device(D, collect(batch.next_state))
    a = send_to_device(D, batch.action)

    batch_size = length(r)
    a = CartesianIndex.(a, 1:batch_size)

    target_quantiles = reshape(Qₜ(s′), N, :, batch_size)
    qₜ = dropdims(mean(target_quantiles; dims=1); dims=1)
    aₜ = dropdims(argmax(qₜ, dims=1); dims=1)
    @views target_quantile_aₜ = target_quantiles[:, aₜ]
    y = reshape(r, 1, batch_size) .+ γ .* reshape(1 .- t, 1, batch_size) .* target_quantile_aₜ

    gs = gradient(params(A)) do
        q = reshape(Q(s), N, :, batch_size)
        @views ŷ = q[:, a]

        loss = mean(energy_distance(ŷ, y))

        ignore_derivatives() do
            learner.loss = loss
        end
        loss
    end

    optimise!(A, gs)
end