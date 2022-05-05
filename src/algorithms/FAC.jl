export FACPolicy

"""
Vanilla Policy Gradient
FACPolicy(;kwargs)
# Keyword arguments
- `approximator`,
- `baseline`,
- `dist`, distribution function of the action
- `γ`, discount factor
- `α_θ`, step size of policy parameter
- `α_w`, step size of baseline parameter
- `batch_size`,
- `rng`,
- `loss`,
- `baseline_loss`,
if the action space is continuous,
then the env should transform the action value, (such as using tanh),
in order to make sure low ≤ value ≤ high
"""
Base.@kwdef mutable struct FACPolicy{
    A<:NeuralNetworkApproximator,
    B<:Union{NeuralNetworkApproximator,Nothing},
    S,
    R<:AbstractRNG,
} <: AbstractPolicy
    approximator::A
    baseline::B = nothing
    action_space::S
    dist::Any
    γ::Float32 = 0.99f0 # discount factor
    α_θ = 1.0f0 # step size of policy
    α_w = 1.0f0 # step size of baseline
    batch_size::Int = 1024
    rng::R = Random.GLOBAL_RNG
    loss::Float32 = 0.0f0
    baseline_loss::Float32 = 0.0f0
end

"""
About continuous action space, see
* [Diagonal Gaussian Policies](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#stochastic-policies
* [Clipped Action Policy Gradient](https://arxiv.org/pdf/1802.07564.pdf)
"""

function (π::FACPolicy)(env::AbstractEnv)
    to_dev(x) = send_to_device(device(π.approximator), x)

    logits = env |> state |> to_dev |> π.approximator |> send_to_host
    if π.action_space isa AbstractVector
        println(size(logits))
        dist = logits |> softmax |> π.dist
        action = π.action_space[rand(π.rng, dist)]
    elseif π.action_space isa Interval
        dist = π.dist.(logits...)
        action = rand.(π.rng, dist)[1]
    else
        error("not implemented")
    end
    action
end

function (π::FACPolicy)(env::MultiThreadEnv)
    error("not implemented")
    # TODO: can PG support multi env? PG only get updated at the end of an episode.
end

function RLBase.update!(
    trajectory::ElasticSARTTrajectory,
    policy::FACPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    push!(trajectory[:state], state(env))
    push!(trajectory[:action], action)
end

function RLBase.update!(
    t::ElasticSARTTrajectory,
    ::FACPolicy,
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    empty!(t)
end

RLBase.update!(::FACPolicy, ::ElasticSARTTrajectory, ::AbstractEnv, ::PreActStage) = nothing

function RLBase.update!(
    π::FACPolicy,
    traj::ElasticSARTTrajectory,
    ::AbstractEnv,
    ::PostEpisodeStage,
)
    model = π.approximator
    to_dev(x) = send_to_device(device(model), x)

    states = traj[:state]
    actions = traj[:action] |> Array # need to convert ElasticArray to Array, or code will fail on gpu. `log_prob[CartesianIndex.(A, 1:length(A))`
    gains = traj[:reward] |> x -> discount_rewards(x, π.γ)

    for idx in Iterators.partition(shuffle(1:length(traj[:terminal])), π.batch_size)
        S = select_last_dim(states, idx) |> Array |> to_dev
        A = actions[idx]
        G = gains[idx] |> x -> Flux.unsqueeze(x, 1) |> to_dev
        # gains is a 1 column array, but the output of flux model is 1 row, n_batch columns array. so unsqueeze it.

        if π.baseline isa NeuralNetworkApproximator
            gs = gradient(Flux.params(π.baseline)) do
                δ = G - π.baseline(S)
                loss = mean(δ .^ 2) * π.α_w # mse
                ignore() do
                    π.baseline_loss = loss
                end
                loss
            end
            update!(π.baseline, gs)
        elseif π.baseline isa Nothing
            # Normalization. See
            # (http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/hw2_final.pdf)
            # (https://web.stanford.edu/class/cs234/assignment3/solution.pdf)
            # normalise should not be used with baseline. or the loss of the policy will be too small.
            δ = G |> x -> normalise(x; dims=2)
        end

        gs = gradient(Flux.params(model)) do
            if π.action_space isa AbstractVector
                # log_prob = S |> model |> logsoftmax
                log_prob = logsoftmax(model(S, 100), dims=3)
                log_probₐ = log_prob[CartesianIndex.(A, 1:length(A)), :]
            elseif π.action_space isa Interval
                dist = π.dist.(model(S)...) # TODO: this part does not work on GPU. See: https://github.com/JuliaStats/Distributions.jl/issues/1183 .
                log_probₐ = logpdf.(dist, A)
            end
            loss = -mean(log_probₐ .* δ) * π.α_θ

            b_rand = reshape(log_prob, :, n_samples) ## SLOW
            b_rand = Zygote.@ignore b_rand .+ 0.01f0 .* CUDA.randn(size(b_rand)...)
            S = entropy_surrogate(sse, permutedims(b_rand, (2, 1)))
            # H = learner.prior(s, b_all) ./ (n_samples)
            loss -= S

            ignore() do
                π.loss = loss
            end
            loss
        end
        update!(model, gs)
    end
end