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
    ::Val{:MountainCar},
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
    env = MountainCarEnv(; T = Float32, max_steps = 200, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

    """
    SET UP LOGGING
    """
    simulation = @ntuple seed
    lg = WandbLogger(project = "RLExp", name="DUQN_MountainCar" * savename(simulation))
    save_dir = datadir("sims", "DUQN", savename(simulation, "jld2"))

    """
    CREATE MODELS
    """
    init = glorot_uniform(rng)


    model = Chain(
        NoisyDense(ns, 128, relu; init_μ = init),
        NoisyDense(128, 128, relu; init_μ = init),
        Split(NoisyDense(128, na; init_μ = init),
              NoisyDense(128, na, softplus; init_μ = init))
    ) |> gpu

    model2 = Chain(
        Dense(ns, 128, relu; init = init),
        Dense(128, 128, relu; init = init),
        Dense(128, na; init = init),
    ) |> gpu


    """
    CREATE AGENT
    """
    agent = Agent(
        policy = QBasedPolicy(
            learner = DUQNLearner(
                B_approximator = NeuralNetworkApproximator(
                    model = model,
                    optimizer = ADAM(1e-3),
                ),
                Q_approximator = NeuralNetworkApproximator(
                    model = model2,
                    optimizer = ADAM(1e-3)
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
        DoEveryNStep() do t, agent, env
            with_logger(lg) do
                p = agent.policy.learner.logging_params
                KL, MSE, H, S, L, Q, V = p["KL"], p["mse"], p["H"], p["S"], p["𝐿"], p["Q"], p["Σ"]
                @info "training" KL = KL MSE = MSE H = H S = S L = L Q = Q V = V
            end
        end,
        DoEveryNEpisode() do t, agent, env
            @info "evaluating agent at $t step..."
            p = agent.policy
            h = ComposedHook(
                TotalRewardPerEpisode(),
                StepsPerEpisode(),
            )
            s = @elapsed run(
                p,
                MountainCarEnv(; T = Float32, max_steps = 200, rng = rng),
                StopAfterEpisode(100; is_show_progress = false),
                h,
            )
            avg_score = mean(h[1].rewards[1:end-1])
            avg_length = mean(h[2].steps[1:end-1])

            @info "finished evaluating agent in $s seconds" avg_length = avg_length avg_score = avg_score
            p = agent.policy.learner.logging_params
            p["episode"] += 1
            with_logger(lg) do
                @info "evaluating" avg_length = avg_length avg_score = avg_score log_step_increment = 0
                @info "training" episode_length = step_per_episode.steps[end] reward = reward_per_episode.rewards[end] log_step_increment = 0
                @info "training" episode = p["episode"] log_step_increment = 0
            end
        end,
        CloseLogger(lg),
    )
    stop_condition = StopAfterStep(200_000, is_show_progress=true)

    """
    RETURN EXPERIMENT
    """
    Experiment(agent, env, stop_condition, hook, "# DUQN <-> MountainCar")
end