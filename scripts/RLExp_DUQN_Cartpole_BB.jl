using Base.Iterators: tail
using Conda
using CUDA
using Distributions: Uniform, Product
using DrWatson
using Flux
using Logging
using RLExp
using Random
using ReinforcementLearning
using Setfield
using Statistics
using Wandb

applychain(::Tuple{}, x, n) = x
applychain(fs::Tuple, x, n) = applychain(tail(fs), first(fs)(x, n), n)
(c::Chain)(x, n) = applychain(c.layers, x, n)

function RL.Experiment(
    ::Val{:RLExp},
    ::Val{:DUQN},
    ::Val{:Cartpole},
    seed = 1
)
    """
    SEEDS
    """
    rng = Random.GLOBAL_RNG
    Random.seed!(rng, seed)
    device_rng = CUDA.functional() ? CUDA.CURAND.RNG() : rng
    Random.seed!(device_rng, isnothing(seed) ? nothing : hash(seed + 1))

    """
    SET UP ENVIRONMENT
    """
    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

    """
    CREATE MODEL
    """
    init = glorot_uniform(rng)


    model = Chain(
        NoisyDense(ns, 64, relu; init_μ = init),
        NoisyDense(64, 64, relu; init_μ = init),
        NoisyDense(64, na; init_μ = init),
    ) |> gpu

    model2 = Chain(
        Dense(ns, 64, relu; init = init),
        Dense(64, 64, relu; init = init),
        Dense(64, na; init = init),
    ) |> gpu

    """
    CREATE AGENT
    """
    agent = Agent(
        policy = QBasedPolicy(
            learner = DUQNLearner(
                B_approximator = NeuralNetworkApproximator(
                    model = model,
                    optimizer = Optimiser(ClipNorm(1.0), ADAM(1e-3)),
                ),
                Q_approximator = NeuralNetworkApproximator(
                    model = model2,
                    optimizer = Optimiser(ClipNorm(1.0), ADAM(1e-5)),
                ),
                γ = 0.99f0,
                update_horizon = 1,
                batch_size = 32,
                min_replay_history = 1,
                B_update_freq = 1,
                Q_update_freq = 10,
                updates_per_step = 10,
                obs_var = 0.01f0,
            ),
            explorer = GreedyExplorer(),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 1_000_000,
            state = Vector{Float32} => ns,
        ),
    )

    """
    SET UP HOOKS
    """
    step_per_episode = StepsPerEpisode()
    reward_per_episode = TotalRewardPerEpisode()
    hook = ComposedHook(
        step_per_episode,
        reward_per_episode,
    )
    stop_condition = StopAfterStep(2_000, is_show_progress=true)

    """
    RETURN EXPERIMENT
    """
    Experiment(agent, env, stop_condition, hook, "# DUQN <-> CartPole")
end