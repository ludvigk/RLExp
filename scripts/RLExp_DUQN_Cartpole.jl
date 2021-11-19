using Base.Iterators: tail
using Conda
using CUDA
using Dates: now
using Distributions: Uniform, Product
using DrWatson
using Flux
using Flux: Optimiser
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
    rng = MersenneTwister()
    Random.seed!(rng, seed)
    device_rng = CUDA.functional() ? CUDA.CURAND.RNG() : rng
    Random.seed!(device_rng, isnothing(seed) ? nothing : hash(seed + 1))

    """
    SET UP ENVIRONMENT
    """
    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

    """
    SET UP LOGGING
    """
    simulation = @ntuple seed
    lg = WandbLogger(project = "RLExp",
                     name="DUQN_CartPole_" * savename(simulation),
                     config = Dict(
                        "B_lr" => 1e-3,
                        "Q_lr" => 1e-3,
                        "B_clip_norm" => 1.0,
                        "Q_clip_norm" => 1.0,
                        "B_update_freq" => 1,
                        "Q_update_freq" => 10,
                        "gamma" => 0.99,
                        "update_horizon" => 1,
                        "batch_size" => 32,
                        "min_replay_history" => 1,
                        "updates_per_step" => 10,
                        "obs_var" => 0.01f0,
                        "traj_capacity" => 1_000_000,
                        "seed" => seed,
                     ),
    )
    save_dir = datadir("sims", "DUQN", "CartPole", "$(now())")

    """
    CREATE MODEL
    """
    init = glorot_uniform(rng)


    model = Chain(
        NoisyDense(ns, 64, relu; init_μ = init, rng = device_rng),
        NoisyDense(64, 64, relu; init_μ = init, rng = device_rng),
        NoisyDense(64, na; init_μ = init, rng = device_rng),
    ) |> gpu

    model2 = Chain(
        Dense(ns, 64, relu; init = init),
        Dense(64, 64, relu; init = init),
        Dense(64, na; init = init),
    ) |> gpu

    # u = Product([Uniform(-2.4f0, 2.4f0),
    #                  Uniform(-10.0f0, 10.0f0),
    #                  Uniform(-0.418f0, 0.418f0),
    #                  Uniform(-10.0f0, 10.0f0)])
    # sse = SpectralSteinEstimator(0.05, nothing, 0.99)

    # for _=1:1_000
    #     samples = rand(u, 32)

    #     gs = gradient(params(model)) do
    #         b_all = model(samples, 100)
    #         b_rand = reshape(b_all[1], :, 100)
    #         b_noisy = b_rand
    #         S = entropy_surrogate(sse, permutedims(b_noisy, (2, 1)))

    #         pr = [-1 1] .* samples[4,:]
    #         pr = reshape(repeat(pr, 1,  100), :, 100)
    #         H = sum((b_noisy .- 40 .* pr) .^ 2 ./ (2 * 100.0f0 .^ 2)) ./ (size(b_noisy, 2) .* 32)
    #         KL = H - S
    #     end
    #     Flux.update!(opt, params(model), gs)
        
    #     gs = gradient(params(model2)) do 
    #         b = model(samples, 100)[1]
    #         B̂ = dropdims(mean(b, dims=ndims(b)), dims=ndims(b))
    #         q = model2(samples)
    #         𝐿 = sum((q .- B̂) .^ 2)
    #         return 𝐿
    #     end
    #     Flux.update!(opt2, params(model2), gs)
    # end

    """
    CREATE AGENT
    """
    agent = Agent(
        policy = QBasedPolicy(
            learner = DUQNLearner(
                B_approximator = NeuralNetworkApproximator(
                    model = model,
                    optimizer = Optimiser(ClipNorm(1.0), ADAM(1e-2)),
                ),
                Q_approximator = NeuralNetworkApproximator(
                    model = model2,
                    optimizer = Optimiser(ClipNorm(1.0), ADAM(3e-3)),
                ),
                γ = 0.99f0,
                update_horizon = 1,
                batch_size = 32,
                min_replay_history = 1,
                B_update_freq = 1,
                Q_update_freq = 20,
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
                CartPoleEnv(; T = Float32),
                StopAfterEpisode(100; is_show_progress = false),
                h,
            )
            avg_score = mean(h[1].rewards[1:end-1])
            avg_length = mean(h[2].steps[1:end-1])

            @info "finished evaluating agent in $s seconds" avg_length = avg_length avg_score = avg_score
            with_logger(lg) do
                @info "evaluating" avg_length = avg_length avg_score = avg_score log_step_increment = 0
                @info "training" episode_length = step_per_episode.steps[end] reward = reward_per_episode.rewards[end] log_step_increment = 0
            end
        end,
        CloseLogger(lg),
    )
    stop_condition = StopAfterStep(5_000, is_show_progress=true)

    """
    RETURN EXPERIMENT
    """
    Experiment(agent, env, stop_condition, hook, "# DUQN <-> CartPole")
end