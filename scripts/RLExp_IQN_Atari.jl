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

using Flux: glorot_uniform

applychain(::Tuple{}, x, n; kwargs...) = x
function applychain(fs::Tuple, x, n; kwargs...)
    if isa(first(fs), NoisyDense) || isa(first(fs), Split)
        return applychain(tail(fs), first(fs)(x, n; kwargs...), n; kwargs...)
    else
        return applychain(tail(fs), first(fs)(x), n; kwargs...)
    end
end
(c::Chain)(x, n; kwargs...) = applychain(c.layers, x, n; kwargs...)

function _create_bias(weights::AbstractArray, bias::Bool, dims::Integer...)
    bias ? fill!(similar(weights, dims...), 0) : false
end
function _create_bias(weights::AbstractArray, bias::AbstractArray, dims::Integer...)
    size(bias) == dims || throw(DimensionMismatch("expected bias of size $(dims), got size $(size(bias))"))
    bias
end

struct PDense{F,M<:AbstractMatrix,B}
    weight::M
    bias::B
    σ::F
    function PDense(W::M, bias=true, σ::F=identity) where {M<:AbstractMatrix,F}
        b = _create_bias(W, bias, size(W, 1))
        new{F,M,typeof(b)}(W, b, σ)
    end
end

function PDense((in, out)::Pair{<:Integer,<:Integer}, σ=identity;
    init=glorot_uniform, bias=true)
    PDense(init(out, in), bias, σ)
end

Flux.@functor PDense

function (a::PDense)(x)
    σ = NNlib.fast_act(a.σ, x)
    w = abs.(a.weight)
    return σ.(w * x .+ a.bias)
end

struct MonotonicDense{F,M<:AbstractMatrix,B,I}
    convexity::I
    weight::M
    bias::B
    σ::F
end

Flux.@functor MonotonicDense


function MonotonicDense((in, out)::Pair{<:Integer,<:Integer}, σ=identity;
    init=glorot_uniform, bias=true, ϵ=0.5)
    n_ones = Int(round(out * ϵ))
    n_zeros = out - n_ones
    convexity = vcat(ones(Float32, n_ones), zeros(Float32, n_zeros))
    MonotonicDense(convexity, init(out, in), bias, σ)
end

function (a::MonotonicDense)(x)
    σ = NNlib.fast_act(a.σ, x)
    w = abs.(a.weight)
    y = w * x .+ a.bias
    return σ.(y) .* a.convexity .- σ.(y) .* (1 .- a.convexity)
end


function RL.Experiment(
    ::Val{:RLExp},
    ::Val{:IQNPP},
    ::Val{:Atari};
    game
)

    """
    SET UP LOGGING
    """
    config = Dict(
        "lr" => 0.00005,
        "update_freq" => 4,
        "target_update_freq" => 10_000,
        "opt" => "ADAM",
        "gamma" => 0.99,
        "update_horizon" => 1,
        "batch_size" => 32,
        "min_replay_history" => 20_000,
        "traj_capacity" => 1_000_000,
        "seed" => 1,
        "terminal_on_life_loss" => true,
        "n_steps" => 50_000_000,
    )

    lg = WandbLogger(project="BE",
        name="QQFLOW_Atari($game)",
        config=config,
    )
    save_dir = datadir("sims", "IQNPP", "Atari($game)", "$(now())")
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
        repeat_action_probability=0.0,
        terminal_on_life_loss=terminal_on_life_loss
    )
    N_ACTIONS = length(action_space(env))

    """
    CREATE MODEL
    """
    # initc = Flux.glorot_uniform(rng)
    # initc = Flux.kaiming_normal(rng)

    # flow_depth = get_config(lg, "flow_depth")
    # opt = eval(Meta.parse(get_config(lg, "opt")))
    # opt = ADAM(config["lr"], (0.9, 0.999), config["adam_epsilon"])
    # opt = ADAM(config["lr"])
    # opt = AdaBelief(config["lr"])
    # opt = CenteredRMSProp(config["lr"], 0.0, config["adam_epsilon"])
    # lr = get_config(lg, "lr")
    # clip_norm = get_config(lg, "clip_norm")

    # model = Chain(
    #     x -> x ./ 255,
    #     CrossCor((8, 8), N_FRAMES => 32, relu; stride=4, pad=2, init=initc),
    #     CrossCor((4, 4), 32 => 64, relu; stride=2, pad=2, init=initc),
    #     CrossCor((3, 3), 64 => 64, relu; stride=1, pad=1, init=initc),
    #     x -> reshape(x, :, size(x)[end]),
    #     Dense(11 * 11 * 64, 512, relu, init=initc),
    #     Dense(512, (3 * flow_depth) * N_ACTIONS, init=initc),
    # ) |> gpu
    init = glorot_uniform(rng)

    # approximator = Approximator(
    #     model=TwinNetwork(
    #         FlowNet(
    #             net=model
    #         ),
    #         sync_freq=get_config(lg, "target_update_freq"),
    #     );
    #     optimiser=opt
    # )

    create_model() =
        ImplicitQuantileNet(
            ψ=Chain(
                x -> x ./ 255,
                CrossCor((8, 8), N_FRAMES => 32, relu; stride=4, pad=2, init=init),
                CrossCor((4, 4), 32 => 64, relu; stride=2, pad=2, init=init),
                CrossCor((3, 3), 64 => 64, relu; stride=1, pad=1, init=init),
                x -> reshape(x, :, size(x)[end]),
            ),
            ϕ=Dense(Nₑₘ => 11 * 11 * 64, relu; init=init),
            header=Chain(
                Dense(11 * 11 * 64 => 512, relu; init=init),
                Dense(512 => N_ACTIONS; init=init),
            ),
        ) |> gpu

    """
    CREATE AGENT
    """
    Nₑₘ = 64
    agent = Agent(
        policy=QBasedPolicy(
            learner=IQNPPLearner(
                approximator=Approximator(
                    model=TwinNetwork(
                        create_model(),
                        sync_freq=10_000
                    ),
                    # optimiser=Adam(0.00005),
                    optimiser=Adam(0.00005, (0.9, 0.999), 1e-2 / 32),
                ),
                N=200,
                N′=200,
                Nₑₘ=Nₑₘ,
                K=32,
                γ=0.99f0,
                rng=rng,
                device_rng=device_rng,
            ),
            explorer=EpsilonGreedyExplorer(
                ϵ_init=1.0,
                ϵ_stable=0.01,
                decay_steps=1_000_000,
                kind=:linear,
                rng=rng,
            ),
        ),
        trajectory=Trajectory(
            container=CircularArraySARTTraces(
                capacity=get_config(lg, "traj_capacity"),
                state=Float32 => STATE_SIZE,
            ),
            sampler=NStepBatchSampler{SS′ART}(
                n=get_config(lg, "update_horizon"),
                γ=get_config(lg, "gamma"),
                batch_size=get_config(lg, "batch_size"),
                stack_size=N_FRAMES,
                rng=rng
            ),
            controller=InsertSampleRatioController(
                ratio=1 // get_config(lg, "update_freq"),
                threshold=get_config(lg, "min_replay_history"),
            ),
        ),
    )

    """
    SET UP EVALUATION
    """
    EVALUATION_FREQ = 250_000
    STEP_LOG_FREQ = 1_000
    EPISODE_LOG_FREQ = 1_000
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
                min_weight, max_weight, min_pred, max_pred = p["min_weight"], p["max_weight"], p["min_pred"], p["max_pred"]
                lsi = (STEP_LOG_FREQ * 4)
                @info "training" L nll sldj Qt QA Q1 Q2 mu sigma l2norm min_weight max_weight min_pred max_pred log_step_increment = lsi
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
        # try
        #     s = agent.trajectory[:state]
        #     beg = rand((1+N_FRAMES):size(s, 3))
        #     s = s[:, :, (beg-N_FRAMES):(beg-1)]
        #     s = Flux.unsqueeze(s, 4) |> gpu
        #     samples = agent.policy.learner.approximator.source(s, 500)[1] |> cpu
        #     p = plot()
        #     for action in 1:size(samples, 1)
        #         density!(samples[action, 1, :], c=action, label="action $(action)")
        #         vline!([mean(samples[action, 1, :])], c=action, label=false)
        #     end
        #     Plots.savefig(p, save_dir * "/qdistr_$(t).png")
        # catch
        #     @warn "Could not save plot"
        # end
    end
    eval_hook = DoEveryNStep(; n=EVALUATION_FREQ) do t, agent, env
        # @info "Saving agent at step $t to $save_dir"
        # jldsave(save_dir * "/model_latest.jld2"; agent)
        @info "evaluating agent at $t step..."
        p = agent.policy
        p = @set p.explorer = EpsilonGreedyExplorer(0.001; rng=rng)
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
                repeat_action_probability=0.0,
                seed=isnothing(seed) ? nothing : hash(seed + t)
            ),
            StopAfterStep(125_000; is_show_progress=false),
            h,
        )
        p_every_step = DoEveryNStep() do tt, agent, env
            push!(screens, get_screen(env))
        end
        p_every_ep = DoEveryNEpisode(; stage=PostEpisodeStage()) do tt, agent, env
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
    # hook = EmptyHook()
    stop_condition = StopAfterStep(get_config(lg, "n_steps"), is_show_progress=true)

    """
    RETURN EXPERIMENT
    """
    Experiment(agent, env, stop_condition, hook)
end