using Base.Iterators: tail
# using BSON: @load, @save
using JLD2
using Conda
using CUDA
using Dates: now
using Distributions: Uniform, Product
using DrWatson
using Flux
using Flux: Optimiser
using Images
using Logging
using Plots
using StatsPlots
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
    ::Val{:QQFLOW},
    ::Val{:Atari},
    name;
    restore=nothing,
    config=nothing,
)

    """
    SET UP LOGGING
    """
    if isnothing(config)
        config = Dict(
            "B_lr" => 5e-5,
            "Q_lr" => 1,
            "B_clip_norm" => 1000.0,
            "B_update_freq" => 4,
            "Q_update_freq" => 10_000,
            "n_samples_act" => 100,
            "n_samples_target" => 100,
            "B_opt" => "ADAM",
            "gamma" => 0.99,
            "update_horizon" => 3,
            "batch_size" => 32,
            "min_replay_history" => 50_000,
            "updates_per_step" => 1,
            "is_enable_double_DQN" => true,
            "traj_capacity" => 1_000_000,
            "seed" => 1,
            "flow_depth" => 8,
            "terminal_on_life_loss" => false,
        )
    end

    lg = WandbLogger(project="BE",
        name="QQFLOW_Atari($name)",
        config=config,
    )
    save_dir = datadir("sims", "QQFLOW", "Atari($name)", "$(now())")
    mkpath(save_dir)

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
    terminal_on_life_loss = get_config(lg, "terminal_on_life_loss")
    N_FRAMES = 4
    STATE_SIZE = (84, 84)
    env = atari_env_factory(
        name,
        STATE_SIZE,
        N_FRAMES;
        seed=isnothing(seed) ? nothing : hash(seed + 1),
        terminal_on_life_loss = terminal_on_life_loss,
    )
    N_ACTIONS = length(action_space(env))

    if restore === nothing
        """
        CREATE MODEL
        """
        initc = glorot_uniform(rng)
        init(a, b) = (2 .* rand(a, b) .- 1) ./ sqrt(b)
        init_σ(dims...) = fill(0.4f0 / Float32(sqrt(dims[end])), dims)
        
        flow_depth = get_config(lg, "flow_depth")

        B_model = Chain(
            x -> x ./ 255,
            Conv((8, 8), N_FRAMES => 32, gelu; stride=4, pad=2, init=initc),
            Conv((4, 4), 32 => 64, gelu; stride=2, pad=2, init=initc),
            Conv((3, 3), 64 => 64, gelu; stride=1, pad=1, init=initc),
            x -> reshape(x, :, size(x)[end]),
            Dense(11 * 11 * 64, 512, gelu, init=initc),
            Dense(512, (2 + 3 * flow_depth) * N_ACTIONS, relu, init=initc),
        ) |> gpu

        Q_model = Chain(
            x -> x ./ 255,
            Conv((8, 8), N_FRAMES => 32, gelu; stride=4, pad=2, init=initc),
            Conv((4, 4), 32 => 64, gelu; stride=2, pad=2, init=initc),
            Conv((3, 3), 64 => 64, gelu; stride=1, pad=1, init=initc),
            x -> reshape(x, :, size(x)[end]),
            Dense(11 * 11 * 64, 512, gelu, init=initc),
            Dense(512, (2 + 3 * flow_depth) * N_ACTIONS, relu, init=initc),
        ) |> gpu

        B_opt = eval(Meta.parse(get_config(lg, "B_opt")))

        B_approximator = NeuralNetworkApproximator(
            model=FlowNet(
                net=B_model,
                n_actions=N_ACTIONS,
            ),
            optimizer=Optimiser(ClipNorm(get_config(lg, "B_clip_norm")), B_opt(get_config(lg, "B_lr"))),
        ) |> gpu

        Q_approximator = NeuralNetworkApproximator(
            model=FlowNet(
                net=Q_model,
                n_actions=N_ACTIONS,
            ),
        ) |> gpu

        """
        CREATE AGENT
        """
        B_opt = eval(Meta.parse(get_config(lg, "B_opt")))
        # B_opt = ADAM(6.25e-5, (0.4, 0.5))

        agent = Agent(
            policy=QBasedPolicy(
                learner=QQFLOWLearner(
                    B_approximator=B_approximator,
                    Q_approximator=Q_approximator,
                    num_actions=N_ACTIONS,
                    Q_lr=get_config(lg, "Q_lr"),
                    γ=get_config(lg, "gamma"),
                    update_horizon=get_config(lg, "update_horizon"),
                    n_samples_act=get_config(lg, "n_samples_act"),
                    n_samples_target=get_config(lg, "n_samples_target"),
                    batch_size=get_config(lg, "batch_size"),
                    min_replay_history=get_config(lg, "min_replay_history"),
                    B_update_freq=get_config(lg, "B_update_freq"),
                    Q_update_freq=get_config(lg, "Q_update_freq"),
                    is_enable_double_DQN=get_config(lg, "is_enable_double_DQN"),
                    stack_size = N_FRAMES,
                ),
                explorer=EpsilonGreedyExplorer(
                    ϵ_init = 1.0,
                    ϵ_stable = 0.01,
                    decay_steps = 1_000_000,
                    kind = :linear,
                    rng=rng,
                ),
            ),
            trajectory = CircularArraySARTTrajectory(
                capacity = get_config(lg, "traj_capacity"),
                state = Matrix{Float32} => STATE_SIZE,
            ),
        )
    else
        agent = load(restore, "agent")
        agent.policy.learner = convert(QQFLOWLearner, agent.policy.learner) |> gpu
        # @load restore agent
    end

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
        DoEveryNStep(; n=STEP_LOG_FREQ) do t, agent, env
            try
                with_logger(lg) do
                    p = agent.policy.learner.logging_params
                    L, nll, sldj, Qt, QA = p["𝐿"], p["nll"], p["sldj"], p["Qₜ"], p["QA"]
                    Q1, Q2, mu, sigma, l2norm = p["Q1"], p["Q2"], p["mu"], p["sigma"], p["l2norm"]
                    min_weight, max_weight, min_pred, max_pred = p["min_weight"], p["max_weight"], p["min_pred"],p["max_pred"]
                    @info "training" L nll sldj Qt QA Q1 Q2 mu sigma l2norm min_weight max_weight min_pred max_pred
                end
            catch
                close(lg)
                stop("Program most likely terminated through WandB interface.")
            end
        end,
        DoEveryNEpisode(; n=EPISODE_LOG_FREQ) do t, agent, env
            with_logger(lg) do
                @info "training" episode = t log_step_increment = 0
                @info "training" episode_length = step_per_episode.steps[end] reward = reward_per_episode.rewards[end] log_step_increment = 0
            end
            try
                s = agent.trajectory[:state]
                beg = rand((1 + N_FRAMES):size(s,3))
                s = s[:,:,(beg - N_FRAMES):(beg - 1)]
                s = Flux.unsqueeze(s, 4) |> gpu
                samples = agent.policy.learner.B_approximator(s, 500)[1] |> cpu
                p = plot()
                for action in 1:size(samples, 1)
                    density!(samples[action, 1, :], c=action, label="action $(action)")
                    vline!([mean(samples[action, 1, :])], c=action, label=false)
                end
                Plots.savefig(p, save_dir * "/qdistr_$(t).png")
            catch
                @warn "Could not save plot"
            end
        end,
        DoEveryNStep(; n=EVALUATION_FREQ) do t, agent, env
            @info "Saving agent at step $t to $save_dir"
            jldsave(save_dir * "/model_latest.jld2"; agent)
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
                    seed=isnothing(seed) ? nothing : hash(seed + t)
                ),
                StopAfterStep(25_000; is_show_progress=false),
                h,
            )

            run(
                p,
                atari_env_factory(
                    name,
                    STATE_SIZE,
                    N_FRAMES,
                    MAX_EPISODE_STEPS_EVAL;
                    seed=isnothing(seed) ? nothing : hash(seed + t)
                ),
                StopAfterEpisode(1; is_show_progress=false),
                ComposedHook(
                    DoEveryNStep() do tt, agent, env
                        push!(screens, get_screen(env))
                    end,
                    DoEveryNEpisode() do tt, agent, env
                        Images.save(joinpath(save_dir, "$(t).gif"), cat(screens..., dims=3), fps=30)
                        Wandb.log(lg, Dict(
                                "evaluating" => Wandb.Video(joinpath(save_dir, "$(t).gif"))
                            ); step=lg.global_step)
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
    Experiment(agent, env, stop_condition, hook, "# Noisy <-> Atari($name)")
end