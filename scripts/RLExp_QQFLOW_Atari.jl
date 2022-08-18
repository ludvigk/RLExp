using Base.Iterators: tail
# using BSON: @load, @save
using JLD2
using Conda
using CUDA
using Dates: now
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
    ::Val{:Atari};
    game
)

    """
    SET UP LOGGING
    """
    config = Dict(
        "lr" => 1e-5,
        "clip_norm" => 10,
        "update_freq" => 4,
        "target_update_freq" => 8_000,
        "n_samples_act" => 100,
        "n_samples_target" => 100,
        "opt" => "ADAM",
        "gamma" => 0.99,
        "update_horizon" => 3,
        "batch_size" => 32,
        "min_replay_history" => 20_000,
        "is_enable_double_DQN" => true,
        "traj_capacity" => 1_000_000,
        "seed" => 1,
        "flow_depth" => 8,
        "terminal_on_life_loss" => false,
        "n_steps" => 50_000_000,
    )

    lg = WandbLogger(project="BE",
        name="QQFLOW_Atari($game)",
        config=config,
    )
    save_dir = datadir("sims", "QQFLOW", "Atari($game)", "$(now())")
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
        game,
        STATE_SIZE,
        N_FRAMES;
        seed=isnothing(seed) ? nothing : hash(seed + 1),
        terminal_on_life_loss = terminal_on_life_loss,
    )
    N_ACTIONS = length(action_space(env))

    """
    CREATE MODEL
    """
    # initc = glorot_uniform(rng)
    initc = Flux.kaiming_normal(rng)
    
    flow_depth = get_config(lg, "flow_depth")
    opt = eval(Meta.parse(get_config(lg, "opt")))
    lr = get_config(lg, "lr")
    clip_norm = get_config(lg, "clip_norm")

    model = Chain(
        x -> x ./ 255,
        CrossCor((8, 8), N_FRAMES => 32, relu; stride=4, pad=2, init=initc),
        CrossCor((4, 4), 32 => 64, relu; stride=2, pad=2, init=initc),
        CrossCor((3, 3), 64 => 64, relu; stride=1, pad=1, init=initc),
        x -> reshape(x, :, size(x)[end]),
        Dense(11 * 11 * 64, 512, relu, init=initc),
        Dense(512, (2 + 3 * flow_depth) * N_ACTIONS, init=initc),
    ) |> gpu


    approximator = Approximator(
        model=TwinNetwork(
            FlowNet(
                net=model
            ),
            sync_freq=get_config(lg, "target_update_freq"),
        );
        optimiser=Optimiser(ClipNorm(clip_norm), opt(lr)),
    )

    """
    CREATE AGENT
    """
    agent = Agent(
        policy=QBasedPolicy(
            learner=QQFLOWLearner(
                approximator=approximator,
                n_actions=N_ACTIONS,
                γ=get_config(lg, "gamma"),
                update_horizon=get_config(lg, "update_horizon"),
                n_samples_act=get_config(lg, "n_samples_act"),
                n_samples_target=get_config(lg, "n_samples_target"),
                is_enable_double_DQN=get_config(lg, "is_enable_double_DQN"),
            ),
            explorer=EpsilonGreedyExplorer(                                                                                                                                                                                                             
                ϵ_init = 1.0,
                ϵ_stable = 0.01,
                decay_steps = 250_000,
                kind = :linear,
                rng=rng,
            ),
        ),
        trajectory=Trajectory(
            container = CircularArraySARTTraces(
                capacity=get_config(lg, "traj_capacity"),
                state=Float32 => STATE_SIZE,
            ),                                                                                                                                                                                                              
            sampler=NStepBatchSampler{SS′ART}(
                n=get_config(lg, "update_horizon"),
                γ=get_config(lg, "gamma"),
                batch_size=get_config(lg, "batch_size"),
                stack_size = N_FRAMES,
                rng=rng
            ),
            controller = InsertSampleRatioController(
                threshold=get_config(lg, "min_replay_history"),
                n_inserted=-1,
            ),
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
    every_n_step = DoEveryNStep(; n=STEP_LOG_FREQ) do t, agent, env
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
    end
    every_n_ep = DoEveryNEpisode(; n=EPISODE_LOG_FREQ, stage=PostEpisodeStage()) do t, agent, env
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
    end
    eval_hook = DoEveryNStep(; n=EVALUATION_FREQ) do t, agent, env
        # @info "Saving agent at step $t to $save_dir"
        # jldsave(save_dir * "/model_latest.jld2"; agent)
        @info "evaluating agent at $t step..."
        p = agent.policy
        p = @set p.explorer = EpsilonGreedyExplorer(0.001; rng = rng)
        tot_reward = TotalOriginalRewardPerEpisode()
        n_steps = StepsPerEpisode()
        h = tot_reward + n_steps
        s = @elapsed run(
            p,
            atari_env_factory(
                game,
                STATE_SIZE,
                N_FRAMES,
                MAX_EPISODE_STEPS_EVAL;
                seed=isnothing(seed) ? nothing : hash(seed + t)
            ),
            StopAfterStep(125_000; is_show_progress=false),
            h,
        )
        p_every_step = DoEveryNStep() do tt, agent, env
            push!(screens, get_screen(env))
        end
        p_every_ep = DoEveryNEpisode(;stage=PostEpisodeStage()) do tt, agent, env
            Images.save(joinpath(save_dir, "$(t).gif"), cat(screens..., dims=3), fps=30)
            Wandb.log(lg, Dict(
                    "evaluating" => Wandb.Video(joinpath(save_dir, "$(t).gif"))
                ); step=lg.global_step)
            screens = []
        end

        run(
            p,
            atari_env_factory(
                game,
                STATE_SIZE,
                N_FRAMES,
                MAX_EPISODE_STEPS_EVAL;
                seed=isnothing(seed) ? nothing : hash(seed + t)
            ),
            StopAfterEpisode(1; is_show_progress=false),
            p_every_step + p_every_ep,
        )
        avg_score = mean(tot_reward.rewards[1:end-1])
        avg_length = mean(n_steps.steps[1:end-1])

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
    end
    hook = step_per_episode + reward_per_episode + every_n_step + every_n_ep +
        eval_hook + CloseLogger(lg)
    stop_condition = StopAfterStep(get_config(lg, "n_steps"), is_show_progress=true)

    """
    RETURN EXPERIMENT
    """
    Experiment(agent, env, stop_condition, hook)
end