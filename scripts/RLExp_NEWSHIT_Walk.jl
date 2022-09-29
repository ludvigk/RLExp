using Base.Iterators: tail
using JLD2
using CUDA
using Dates: now
using DrWatson
using Flux
using Flux: Optimiser
using Logging
using RLExp
using Random
using ReinforcementLearning
using RLExp
using Setfield
using Statistics
using StatsPlots
using Wandb
using Zygote

device = gpu

applychain(::Tuple{}, x, n; kwargs...) = x
function applychain(fs::Tuple, x, n; kwargs...)
    if isa(first(fs), NoisyDense) || isa(first(fs), Split)
        return applychain(tail(fs), first(fs)(x, n; kwargs...), n; kwargs...)
    else
        return applychain(tail(fs), first(fs)(x), n; kwargs...)
    end
end
(c::Chain)(x, n; kwargs...) = applychain(c.layers, x, n; kwargs...)
(d::Dense)(x, n; kwargs...) = (d::Dense)(x)

function RL.Experiment(
    ::Val{:RLExp},
    ::Val{:NEWSHIT},
    ::Val{:Walk};
    seed=123
)
    """
    SET UP LOGGING
    """
    config = Dict(
        "lr" => 5e-5,
        "update_freq" => 100,
        "target_update_freq" => 100,
        "n_samples_act" => 100,
        "n_samples_target" => 100,
        "opt" => "ADAM",
        "gamma" => 0.99,
        "update_horizon" => 1,
        "batch_size" => 32,
        "min_replay_history" => 100,
        "is_enable_double_DQN" => true,
        "traj_capacity" => 100_000,
        "seed" => 2,
        "flow_depth" => 5,
        "num_steps" => 50_000,
        "epsilon_decay_steps" => 500,
        "epsilon_stable" => 0.01,
    )

    lg = WandbLogger(project="BE",
        name="NEWSHIT_Walk",
        config=config,
    )
    save_dir = datadir("sims", "NEWSHIT", "Walk", "$(now())")
    mkpath(save_dir)

    """
    SEEDS
    """
    seed = config["seed"]
    rng = Xoshiro()
    Random.seed!(rng, seed)
    device_rng = CUDA.functional() ? CUDA.CURAND.RNG() : rng
    Random.seed!(device_rng, isnothing(seed) ? nothing : hash(seed + 1))

    """
    SET UP ENVIRONMENT
    """
    env = RandomWalk1D(;rewards=-1. => 1.0, N=7, start_pos=(7+1) ÷ 2, actions=[-1,1])
    ns, na = length(state(env)), length(action_space(env))

    """
    CREATE MODEL
    """
    init = Flux.glorot_uniform()
    # inil = (args...) -> init(args...) ./ 100
    # init = Flux.glorot_normal()
    # init = Flux.kaiming_normal()
    # init = Flux.kaiming_uniform()

    flow_depth = config["flow_depth"]
    # opt = eval(Meta.parse(get_config(lg, "opt")))
    # opt = AdaBelief(config["lr"])
    # opt = AdamW(config["lr"], (0.9, 0.999), 0.0001)
    opt = Optimiser(WeightDecay(1f-4), Adam(config["lr"]))
    approximator = Approximator(
        model=TwinNetwork(
            Chain(
                Dense(ns, 256, leakyrelu; init=init),
                Dense(256, 256, leakyrelu; init=init),
                Dense(256, 2flow_depth * na; init=init),
            )
            ;
            sync_freq=config["target_update_freq"]
        ),
        optimiser=opt,
    ) |> device

    """
    CREATE AGENT
    """
    agent = Agent(
        policy=QBasedPolicy(
            learner=NEWSHITLearner(
                approximator=approximator,
                n_actions=na,
                γ=config["gamma"],
                update_horizon=config["update_horizon"],
                n_samples_act=config["n_samples_act"],
                n_samples_target=config["n_samples_target"],
                is_enable_double_DQN=config["is_enable_double_DQN"],
            ),
            explorer=EpsilonGreedyExplorer(
                kind=:exp,
                ϵ_stable=config["epsilon_stable"],
                decay_steps=config["epsilon_decay_steps"],
                rng=rng,
            ),
        ),
        trajectory=Trajectory(
            container=CircularArraySARTTraces(
                capacity=config["traj_capacity"],
                state=Float32 => (ns,),
            ),
            sampler=NStepBatchSampler{SS′ART}(
                n=config["update_horizon"],
                γ=config["gamma"],
                batch_size=config["batch_size"],
                rng=rng
            ),
            controller=InsertSampleRatioController(
                ratio=config["update_freq"],
                threshold=config["min_replay_history"],
            ),
            # controller = AsyncInsertSampleRatioController(
            #     1 ./ get_config(lg, "update_freq"),
            #     get_config(lg, "min_replay_history");
            #     ch_in_sz = 10,
            #     ch_out_sz = 10,
            # ),
        )
    )
    """
    SET UP HOOKS
    """
    step_per_episode = StepsPerEpisode()
    reward_per_episode = TotalRewardPerEpisode()
    every_step = DoEveryNStep() do t, agent, env
        try
            with_logger(lg) do
                p = agent.policy.learner.logging_params
                L, nll, sldj, Qt, QA = p["loss"], p["nll"], p["sldj"], p["Qₜ"], p["QA"]
                Q1, Q2, mu, sigma, l2norm = p["Q1"], p["Q2"], p["mu"], p["sigma"], p["l2norm"]
                min_weight, max_weight, min_pred, max_pred = p["min_weight"], p["max_weight"], p["min_pred"], p["max_pred"]
                @info "training" L nll sldj Qt QA Q1 Q2 mu sigma l2norm min_weight max_weight min_pred max_pred

                # last_layer = agent.policy.learner.B_approximator.model[end].paths[1][end].w_ρ
                # penultimate_layer = agent.policy.learner.B_approximator.model[end].paths[1][end-1].w_ρ
                # sul = sum(abs.(last_layer)) / length(last_layer)
                # spl = sum(abs.(penultimate_layer)) / length(penultimate_layer)
                # @info "training" sigma_ultimate_layer = sul sigma_penultimate_layer = spl log_step_increment = 0
            end
        catch
            close(lg)
            stop("Program most likely terminated through WandB interface.")
        end
    end
    every_ep = DoEveryNEpisode(; stage=PostEpisodeStage()) do t, agent, env
        try
            with_logger(lg) do
                @info "training" episode_length = step_per_episode.steps[end] reward = reward_per_episode.rewards[end] log_step_increment = 0
                @info "training" episode = t log_step_increment = 0
            end
        catch
            close(lg)
            stop("Program most likely terminated through WandB interface.")
        end
    end
    every_n_step = DoEveryNStep(n=500) do t, agent, env
        @info "evaluating agent at $t step..."
        p = agent.policy
        total_reward = TotalRewardPerEpisode()
        steps = StepsPerEpisode()
        s = @elapsed run(
            p,
            RandomWalk1D(;rewards=-1. => 1.0, N=7, start_pos=(7+1) ÷ 2, actions=[-1,1]),
            StopAfterEpisode(100; is_show_progress=false),
            total_reward + steps,
        )
        avg_score = mean(total_reward.rewards[1:end-1])
        avg_length = mean(steps.steps[1:end-1])

        @info "finished evaluating agent in $(round(s, digits=2)) seconds" avg_length = avg_length avg_score = avg_score
        try
            with_logger(lg) do
                @info "evaluating" avg_length = avg_length avg_score = avg_score log_step_increment = 0
                # @info "training" episode_length = step_per_episode.steps[end] reward = reward_per_episode.rewards[end] log_step_increment = 0
                # @info "training" episode = t log_step_increment = 0
            end
        catch
            close(lg)
            stop("Program most likely terminated through WandB interface.")
        end

        # @info "Saving agent at step $t to $save_dir"
        # try
        #     env = CartPoleEnv(; T=Float32, max_steps=500)
        #     s = Flux.unsqueeze(env.state, 2) |> device
        #     samples = agent.policy.learner.approximator.model.source(s, 500, na)[1] |> cpu
        #     p = plot(; size=(600, 400))
        #     for action in 1:size(samples, 1)
        #         density!(samples[action, 1, :], c=action, label="action $(action)")
        #         vline!([mean(samples[action, 1, :])], c=action, label=false)
        #     end
        #     Plots.savefig(p, save_dir * "/qdistr_$(t).png")
        #     Wandb.log(lg, Dict(
        #             "evaluating" => Wandb.Image(joinpath(save_dir, "qdistr_$(t).png"))
        #         ); step=lg.global_step)
        # catch
        #     close(lg)
        #     @error "Failed to save plot. Probably NaN values."
        #     throw(Error())
        # end
    end

    every_n_ep = DoEveryNEpisode(n=5000; stage=PostEpisodeStage()) do t, agent, env
        @info "Saving agent at step $t to $save_dir"
        jldsave(save_dir * "/model_$t.jld2"; agent)
    end

    hook = step_per_episode + reward_per_episode + every_step + every_ep +
           every_n_step + every_n_ep + CloseLogger(lg)
    # hook = EmptyHook()
    stop_condition = StopAfterStep(config["num_steps"], is_show_progress=true)

    """
    RETURN EXPERIMENT
    """
    Experiment(agent, env, stop_condition, hook)
end