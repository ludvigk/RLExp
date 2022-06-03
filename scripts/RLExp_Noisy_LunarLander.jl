using Base.Iterators: tail
using BSON: @load, @save
using CUDA
using Dates: now
using Distributions: Uniform, Product
using DrWatson
using Flux
using Flux: Optimiser
using Flux.Losses
using Logging
using RLExp
using Random
using ReinforcementLearning
using Setfield
using Statistics
using Wandb

function RL.Experiment(
    ::Val{:RLExp},
    ::Val{:Noisy},
    ::Val{:LunarLander},
    name,
    restore=nothing,
)
    """
    SET UP LOGGING
    """
    lg = WandbLogger(project="RLExp",
        name="Noisy_LunarLander",
        config=Dict(
            "B_lr" => 1e-4,
            "Q_lr" => 1.0,
            "B_clip_norm" => 10.0,
            "B_update_freq" => 1,
            "Q_update_freq" => 500,
            "B_opt" => "ADAM",
            "gamma" => 0.99f0,
            "update_horizon" => 1,
            "batch_size" => 64,
            "min_replay_history" => 10_000,
            "updates_per_step" => 1,
            "is_enable_double_DQN" => true,
            "traj_capacity" => 50_000,
            "seed" => 1,
        ),
    )
    save_dir = datadir("sims", "Noisy", "LunarLander", "$(now())")

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
    env = GymEnv("LunarLander-v2"; seed=1)
    env = discrete2standard_discrete(env)
    ns, na = length(state(env)), length(action_space(env))

    """
    CREATE MODEL
    """
    # init = glorot_uniform(rng)
    init(a, b) = (2 .* rand(rng, Float32, a, b) .- 1) ./ Float32(sqrt(b))
    init_σ(dims...) = fill(0.4f0 / Float32(sqrt(dims[end])), dims)


    agent = Agent(
        policy=QBasedPolicy(
            learner=DQNLearner(
                approximator=NeuralNetworkApproximator(
                    model=Chain(
                        NoisyDense(ns, 128, selu; init_μ=init, init_σ=init_σ, rng=device_rng),
                        NoisyDense(128, 128, selu; init_μ=init, init_σ=init_σ, rng=device_rng),
                        NoisyDense(128, na; init_μ=init, init_σ=init_σ, rng=device_rng),
                    ),
                    optimizer=ADAM(get_config(lg, "B_lr")),
                ) |> gpu,
                target_approximator=NeuralNetworkApproximator(
                    model=Chain(
                        NoisyDense(ns, 128, selu; init_μ=init, init_σ=init_σ, rng=device_rng),
                        NoisyDense(128, 128, selu; init_μ=init, init_σ=init_σ, rng=device_rng),
                        NoisyDense(128, na; init_μ=init, init_σ=init_σ, rng=device_rng),
                    )
                ) |> gpu,
                loss_func=mse,
                stack_size=nothing,
                γ=Float32(get_config(lg, "gamma")),
                batch_size=get_config(lg, "batch_size"),
                update_horizon=get_config(lg, "update_horizon"),
                min_replay_history=get_config(lg, "min_replay_history"),
                update_freq=get_config(lg, "B_update_freq"),
                target_update_freq=get_config(lg, "Q_update_freq"),
                rng=rng,
            ),
            explorer=GreedyExplorer(),
        ),
        trajectory=CircularArraySARTTrajectory(
            capacity=get_config(lg, "traj_capacity"),
            state=Vector{Float32} => (ns,),
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
            try
                with_logger(lg) do
                    last_layer = agent.policy.learner.approximator.model[end].w_ρ
                    penultimate_layer = agent.policy.learner.approximator.model[end-1].w_ρ
                    sul = sum(abs.(last_layer)) / length(last_layer)
                    spl = sum(abs.(penultimate_layer)) / length(penultimate_layer)
                    @info "training" sigma_penultimate_layer = spl sigma_ultimate_layer = sul
                end
            catch
                close(lg)
                stop("Program most likely terminated through WandB interface.")
            end
        end,
        DoEveryNEpisode(n=50) do t, agent, env
            @info "evaluating agent at $t step..."
            p = agent.policy
            h = ComposedHook(
                TotalRewardPerEpisode(),
                StepsPerEpisode(),
            )
            env = GymEnv("LunarLander-v2"; seed=1)
            env = discrete2standard_discrete(env)
            s = @elapsed run(
                p,
                env,
                StopAfterEpisode(100; is_show_progress=false),
                h,
            )
            avg_score = mean(h[1].rewards[1:end-1])
            avg_length = mean(h[2].steps[1:end-1])

            @info "finished evaluating agent in $(round(s, digits=2)) seconds" avg_length = avg_length avg_score = avg_score
            try
                with_logger(lg) do
                    @info "evaluating" avg_length = avg_length avg_score = avg_score log_step_increment = 0
                    @info "training" episode_length = step_per_episode.steps[end] reward = reward_per_episode.rewards[end] log_step_increment = 0
                    @info "training" episode = t log_step_increment = 0

                    last_layer = agent.policy.learner.approximator.model[end].w_ρ
                    penultimate_layer = agent.policy.learner.approximator.model[end-1].w_ρ
                    sul = sum(abs.(last_layer)) / length(last_layer)
                    spl = sum(abs.(penultimate_layer)) / length(penultimate_layer)
                    @info "training" sigma_penultimate_layer = spl sigma_ultimate_layer = sul log_step_increment = 0
                end
            catch
                close(lg)
                stop("Program most likely terminated through WandB interface.")
            end
        end,
        CloseLogger(lg),
    )
    stop_condition = StopAfterStep(2_000_000, is_show_progress=true)

    """
    RETURN EXPERIMENT
    """
    Experiment(agent, env, stop_condition, hook, "# Noisy <-> LunarLander")
end