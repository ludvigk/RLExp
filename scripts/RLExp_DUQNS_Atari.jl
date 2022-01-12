using Base.Iterators: tail
using BSON: @load, @save
using Conda
using CUDA
using Dates: now
using Distributions: Uniform, Product
using DrWatson
using Flux
using Flux: Optimiser
using Images
using Logging
using RLExp
using Random
using ReinforcementLearning
using Setfield
using Statistics
using Wandb

applychain(::Tuple{}, x, n; kwargs...) = x
function applychain(fs::Tuple, x, n; kwargs...)
    if isa(first(fs), NoisyDense) || isa(first(fs), Split)
        return applychain(tail(fs), first(fs)(x, n; kwargs...), n; kwargs...)
    else
        return applychain(tail(fs), first(fs)(x), n; kwargs...)
    end
end
(c::Chain)(x, n; kwargs...) = applychain(c.layers, x, n; kwargs...)

function RL.Experiment(
    ::Val{:RLExp},
    ::Val{:DUQNS},
    ::Val{:Atari},
    name;
    restore=nothing,
   )

    """
    SET UP LOGGING
    """
    lg = WandbLogger(project = "RLExp",
                     name="DUQNS_Atari($name)",
                     config = Dict(
                        "B_lr" => 1e-4,
                        "Q_lr" => 1,
                        "B_clip_norm" => 10.0,
                        "B_update_freq" => 4,
                        "Q_update_freq" => 40_000,
                        "B_opt" => "ADAM",
                        "gamma" => 0.99,
                        "update_horizon" => 1,
                        "batch_size" => 32,
                        "min_replay_history" => 50_000,
                        "updates_per_step" => 1,
                        "λ" => 1,
                        "prior" => "GaussianPrior(0, 10)",
                        "n_samples" => 100,
                        "η" => 0.01,
                        "nev" => 20,
                        "is_enable_double_DQN" => true,
                        "traj_capacity" => 1_000_000,
                        "seed" => 1,
                     ),
    )
    save_dir = datadir("sims", "DUQNS", "Atari($name)", "$(now())")

    """
    SEEDS
    """
    seed = get_config(lg, "seed")
    rng = MersenneTwister()
    Random.seed!(rng, seed)
    device_rng = CUDA.functional() ? CUDA.CURAND.RNG() : rng
    Random.seed!(device_rng, isnothing(seed) ? nothing : hash(seed + 1))

    """
    SET UP ENVIRONMENT
    """
    N_FRAMES = 4
    STATE_SIZE = (84, 84)
    env = atari_env_factory(
        name,
        STATE_SIZE,
        N_FRAMES;
        seed = isnothing(seed) ? nothing : hash(seed + 1),
    )
    N_ACTIONS = length(action_space(env))

    """
    CREATE MODEL
    """
    initc = glorot_uniform(rng)
    init(a, b) = (2 .* rand(a, b) .- 1) .* √(3 / a)
    # init_σ(dims...) = fill(0.005f0, dims)
    init_σ(dims...) = fill(0.1f0 / Float32(sqrt(dims[end])), dims)

    if restore === nothing
        B_model = Chain(
            x -> x ./ 255,
            Conv((8, 8), N_FRAMES => 32, relu; stride = 4, pad = 2, init = initc),
            Conv((4, 4), 32 => 64, relu; stride = 2, pad = 2, init = initc),
            Conv((3, 3), 64 => 64, relu; stride = 1, pad = 1, init = initc),
            x -> reshape(x, :, size(x)[end]),
            NoisyDense(11 * 11 * 64, 512, relu; init_μ = init, init_σ = init_σ),
            Split(
                NoisyDense(512, N_ACTIONS; init_μ = init, init_σ = init_σ),
                NoisyDense(512, N_ACTIONS; init_μ = init, init_σ = init_σ),
            )
        ) |> gpu

        Q_model = Chain(
            x -> x ./ 255,
            Conv((8, 8), N_FRAMES => 32, relu; stride = 4, pad = 2, init = initc),
            Conv((4, 4), 32 => 64, relu; stride = 2, pad = 2, init = initc),
            Conv((3, 3), 64 => 64, relu; stride = 1, pad = 1, init = initc),
            x -> reshape(x, :, size(x)[end]),
            NoisyDense(11 * 11 * 64, 512, relu; init_μ = init),
            Split(
                NoisyDense(512, N_ACTIONS; init_μ = init),
                NoisyDense(512, N_ACTIONS; init_μ = init),
            )
        ) |> gpu

    else
        @load restore B_approximator Q_approximator
    end


    """
    CREATE AGENT
    """
    B_opt = eval(Meta.parse(get_config(lg, "B_opt")))
    # B_opt = ADAM(6.25e-5, (0.4, 0.5))
    prior = eval(Meta.parse(get_config(lg, "prior")))

    agent = Agent(
        policy = QBasedPolicy(
            learner = DUQNSLearner(
                B_approximator = NeuralNetworkApproximator(
                    model = B_model,
                    optimizer = Optimiser(ClipNorm(get_config(lg, "B_clip_norm")), B_opt(get_config(lg, "B_lr"))),
                ),
                Q_approximator = NeuralNetworkApproximator(
                    model = Q_model
                ),
                Q_lr = get_config(lg, "Q_lr"),
                γ = get_config(lg, "gamma"),
                update_horizon = get_config(lg, "update_horizon"),
                batch_size = get_config(lg, "batch_size"),
                min_replay_history = get_config(lg, "min_replay_history"),
                B_update_freq = get_config(lg, "B_update_freq"),
                Q_update_freq = get_config(lg, "Q_update_freq"),
                updates_per_step = get_config(lg, "updates_per_step"),
                λ = get_config(lg, "λ"),
                n_samples = get_config(lg, "n_samples"),
                η = get_config(lg, "η"),
                nev = get_config(lg, "nev"),
                is_enable_double_DQN = get_config(lg, "is_enable_double_DQN"),
                prior = prior,
                stack_size = N_FRAMES,
            ),
            explorer = GreedyExplorer(),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = get_config(lg, "traj_capacity"),
            state = Matrix{Float32} => STATE_SIZE,
        ),
    )

    """
    SET UP EVALUATION
    """
    EVALUATION_FREQ = 250_000
    STEP_LOG_FREQ = 1_000
    EPISODE_LOG_FREQ = 100
    MAX_EPISODE_STEPS_EVAL = 27_000

    screens = []

    """
    SET UP HOOKS
    """
    step_per_episode = StepsPerEpisode()
    reward_per_episode = TotalRewardPerEpisode()
    hook = ComposedHook(
        step_per_episode,
        reward_per_episode,
        DoEveryNStep(;n=STEP_LOG_FREQ) do t, agent, env
            try
                with_logger(lg) do
                    p = agent.policy.learner.logging_params
                    KL, MSE, H, S, L, Q = p["KL"], p["mse"], p["H"], p["S"], p["𝐿"], p["Q"]
                    @info "training" KL = KL MSE = MSE H = H S = S L = L Q = Q log_step_increment = STEP_LOG_FREQ
                end
            catch
                close(lg)
                stop("Program most likely terminated through WandB interface.")
            end
        end,
        DoEveryNEpisode(;n=EPISODE_LOG_FREQ) do t, agent, env
            with_logger(lg) do
                @info "training" episode = t log_step_increment = 0
                @info "training" episode_length = step_per_episode.steps[end] reward = reward_per_episode.rewards[end] log_step_increment = 0
            end
        end,
        DoEveryNStep(;n=EVALUATION_FREQ) do t, agent, env
            @info "evaluating agent at $t step..."
            p = agent.policy

            h = ComposedHook(
                TotalOriginalRewardPerEpisode(),
                StepsPerEpisode(),
            )
            s = @elapsed run(
                p,
                atari_env_factory(
                    name,
                    STATE_SIZE,
                    N_FRAMES,
                    MAX_EPISODE_STEPS_EVAL;
                    seed = isnothing(seed) ? nothing : hash(seed + t),
                ),
                StopAfterStep(25_000; is_show_progress = false),
                h,
            )

            run(
                p,
                atari_env_factory(
                    name,
                    STATE_SIZE,
                    N_FRAMES,
                    MAX_EPISODE_STEPS_EVAL;
                    seed = isnothing(seed) ? nothing : hash(seed + t),
                ),
                StopAfterEpisode(1; is_show_progress = false),
                ComposedHook(
                DoEveryNStep() do tt, agent, env
                    push!(screens, get_screen(env))
                end,
                DoEveryNEpisode() do tt, agent, env
                    Images.save(joinpath(save_dir, "$(t).gif"), cat(screens..., dims=3), fps=30)
                    Wandb.log(lg, Dict(
                        "evaluating" => Wandb.Video(joinpath(save_dir, "$(t).gif"))
                    ); step = lg.global_step)
                    screens = []    
                end,
                ),
            )
            avg_score = mean(h[1].rewards[1:end-1])
            avg_length = mean(h[2].steps[1:end-1])

            @info "finished evaluating agent in $(round(s, digits=2)) seconds" avg_length = avg_length avg_score = avg_score
            try
                with_logger(lg) do
                    @info "evaluating" avg_length = avg_length avg_score = avg_score log_step_increment = 0
                    @info "training" episode_length = step_per_episode.steps[end] reward = reward_per_episode.rewards[end] log_step_increment = 0
                end
            catch
                close(lg)
                stop("Program most likely terminated through WandB interface.")
            end
        end,
        CloseLogger(lg),
    )
    stop_condition = StopAfterStep(50_000_000, is_show_progress=true)

    """
    RETURN EXPERIMENT
    """
    Experiment(agent, env, stop_condition, hook, "# DUQNS <-> Atari($name)")
end