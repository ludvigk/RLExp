using Base.Iterators: tail
using BSON: @load, @save
using Conda
using CUDA
using Distributions: Uniform, Product
using DrWatson
using Flux
using Images
using Logging
using RLExp
using Random
using ReinforcementLearning
using Setfield
using Statistics
using Wandb

applychain(::Tuple{}, x, n) = x
function applychain(fs::Tuple, x, n)
    f = first(fs)
    if isa(f, NoisyDense) || isa(f, NoisyConv) || isa(f, Split{Tuple{NoisyDense, NoisyDense}})
        y = applychain(tail(fs), f(x, n), n)
    else
        y = applychain(tail(fs), f(x), n)
    end
    y
end
(c::Chain)(x, n) = applychain(c.layers, x, n)

function RL.Experiment(
    ::Val{:RLExp},
    ::Val{:DUQN},
    ::Val{:Atari},
    name::AbstractString;
    restore=nothing,
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
    SET UP LOGGING
    """
    simulation = @ntuple seed
    lg = WandbLogger(project = "RLExp", name="DUQN_Atari_" * savename(simulation))
    save_dir = datadir("sims", "DUQN", "CartPole", $(now()))

    """
    CREATE MODELS
    """
    init = glorot_uniform(rng)

    if restore === nothing
        B_model = Chain(
            x -> x ./ 255,
            CrossCor((8, 8), N_FRAMES => 32, relu; stride = 4, pad = 2, init = init),
            CrossCor((4, 4), 32 => 64, relu; stride = 2, pad = 2, init = init),
            CrossCor((3, 3), 64 => 64, relu; stride = 1, pad = 1, init = init),
            x -> reshape(x, :, size(x)[end]),
            NoisyDense(11 * 11 * 64, 512, relu; init_μ = init),
            NoisyDense(512, N_ACTIONS; init_μ = init),
        ) |> gpu

        B_approximator = NeuralNetworkApproximator(
                        model = B_model,
                        optimizer = Optimiser(ClipNorm(1.0), ADAM(1e-5)),
                    )

        Q_model = Chain(
            x -> x ./ 255,
            CrossCor((8, 8), N_FRAMES => 32, relu; stride = 4, pad = 2, init = init),
            CrossCor((4, 4), 32 => 64, relu; stride = 2, pad = 2, init = init),
            CrossCor((3, 3), 64 => 64, relu; stride = 1, pad = 1, init = init),
            x -> reshape(x, :, size(x)[end]),
            NoisyDense(11 * 11 * 64, 512, relu; init_μ = init),
            NoisyDense(512, N_ACTIONS; init_μ = init),
        ) |> gpu

        Q_approximator = NeuralNetworkApproximator(
                        model = Q_model,
                        optimizer = Optimiser(ClipNorm(1.0), ADAM(1e-7)),
                    )

    else
        @load restore B_approximator Q_approximator
    end


    """
    CREATE AGENT
    """
    agent = Agent(
        policy = QBasedPolicy(
            learner = DUQNLearner(
                B_approximator = B_approximator,
                Q_approximator = Q_approximator,
                γ = 0.99f0,
                update_horizon = 1,
                batch_size = 32,
                min_replay_history = 20_000,
                B_update_freq = 4,
                Q_update_freq = 4,
                updates_per_step = 1,
                obs_var = 0.01f0,
                stack_size = N_FRAMES,
                prior=GaussianPrior(0f0, 10f0)
            ),
            explorer = GreedyExplorer(),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 500_000,
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
    reward_per_episode = TotalOriginalRewardPerEpisode()

    hook = ComposedHook(
        step_per_episode,
        reward_per_episode,
        DoEveryNStep(;n=STEP_LOG_FREQ) do t, agent, env
            with_logger(lg) do
                p = agent.policy.learner.logging_params
                KL, MSE, H, S, L, Q, V = p["KL"], p["mse"], p["H"], p["S"], p["𝐿"], p["Q"], p["Σ"]
                @info "training" KL = KL MSE = MSE H = H S = S L = L Q = Q V = V log_step_increment = STEP_LOG_FREQ
            end
        end,
        DoEveryNEpisode(;n=EPISODE_LOG_FREQ) do t, agent, env
            with_logger(lg) do
                @info "training" episode_length = step_per_episode.steps[end] reward = reward_per_episode.rewards[end] log_step_increment = 0
            end
        end,
        DoEveryNStep(;n=EVALUATION_FREQ) do t, agent, env
            @info "saving model at $t step..."
            file_dir = save_dir * "/model-$t.bson"
            @save file_dir B_model = agent.policy.learner.B_approximator Q_model = agent.policy.learner.Q_approximator
            
            @info "evaluating agent at $t step..."
            p = agent.policy

            # set evaluation epsilon
            p = @set p.explorer = EpsilonGreedyExplorer(0.001; rng = rng)
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

            @info "finished evaluating agent in $s seconds" avg_length = avg_length avg_score = avg_score
            with_logger(lg) do
                @info "evaluating" avg_length = avg_length avg_score = avg_score log_step_increment = 0
            end
        end,
        CloseLogger(lg),
    )
    stop_condition = StopAfterStep(50_000_000, is_show_progress=true)

    """
    RETURN EXPERIMENT
    """
    Experiment(agent, env, stop_condition, hook, "# DUQN <-> Atari($(name))")
end