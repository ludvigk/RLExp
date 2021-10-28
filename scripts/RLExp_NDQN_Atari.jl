using Conda
using CUDA
using DrWatson
using Flux
using Flux: mse, logitcrossentropy
using Images
using Logging
using RLExp
using Random
using ReinforcementLearning
using Setfield
using Statistics
using Wandb

function RL.Experiment(
    ::Val{:Dopamine},
    ::Val{:NDQN},
    ::Val{:Atari},
    name::AbstractString;
    save_dir = nothing,
    seed = nothing
)
    """
    SEEDS
    """
    rng = Random.GLOBAL_RNG
    Random.seed!(rng, seed)
    device_rng = CUDA.functional() ? CUDA.CURAND.RNG() : device_rng
    Random.seed!(device_rng, isnothing(seed) ? nothing : hash(seed + 1))

    """
    SET UP LOGGING
    """
    simulation = @ntuple name
    lg = WandbLogger(project = "RLExp", name=savename(simulation))
    save_dir = datadir("sim", "NDQN", savename(simulation, "jld2"))

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
    init = glorot_uniform(rng)
    create_model() =
        Chain(
            x -> x ./ 255,
            CrossCor((8, 8), N_FRAMES => 32, relu; stride = 4, pad = 2, init = init),
            CrossCor((4, 4), 32 => 64, relu; stride = 2, pad = 2, init = init),
            CrossCor((3, 3), 64 => 64, relu; stride = 1, pad = 1, init = init),
            x -> reshape(x, :, size(x)[end]),
            NoisyDense(11 * 11 * 64, 512, relu; init = init),
            NoisyDense(512, N_ACTIONS; init = init),
        ) |> gpu

    """
    CREATE AGENT
    """
    agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = create_model(),
                    optimizer = ADAM(0.0000625),
                ),  # unlike TF/PyTorch RMSProp doesn't support center
                target_approximator = NeuralNetworkApproximator(model = create_model()),
                update_freq = 4,
                γ = 0.99f0,
                update_horizon = 1,
                batch_size = 32,
                stack_size = N_FRAMES,
                min_replay_history = haskey(ENV, "CI") ? 900 : 20_000,
                loss_func = mse,
                target_update_freq = 8_000,
                rng = rng,
            ),
            explorer = EpsilonGreedyExplorer(
                ϵ_init = 1.0,
                ϵ_stable = 0.01,
                decay_steps = 250_000,
                kind = :linear,
                rng = rng,
            ),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = haskey(ENV, "CI") ? 1_000 : 1_000_00,
            state = Matrix{Float32} => STATE_SIZE,
        ),
    )

    """
    SET UP EVALUATION
    """
    EVALUATION_FREQ = 250_000
    STEP_LOG_FREQ = 1000
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
                @info "training" loss = agent.policy.learner.loss
            end
        end,
        DoEveryNEpisode(;n=EPISODE_LOG_FREQ) do t, agent, env
            with_logger(lg) do
                @info "training" episode_length = step_per_episode.steps[end] reward = reward_per_episode.rewards[end] log_step_increment = 0
            end
        end,
        DoEveryNStep(;n=EVALUATION_FREQ) do t, agent, env
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

    stop_condition = StopAfterStep(
        haskey(ENV, "CI") ? 1_000 : 50_000_000,
        is_show_progress=!haskey(ENV, "CI")
    )

    """
    RETURN EXPERIMENT
    """
    Experiment(agent, env, stop_condition, hook, "# NDQN <-> Atari($name)")
end