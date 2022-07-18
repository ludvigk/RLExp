using Base.Iterators: tail
# using BSON: @load, @save
using JLD2
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
using Zygote

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
    ::Val{:QFLOW},
    ::Val{:Cartpole},
    name;
    restore=nothing,
    config=nothing
)

    """
    SET UP LOGGING
    """
    if isnothing(config)
        config = Dict(
            "B_lr" => 1e-3,
            "Q_lr" => 1,
            "B_clip_norm" => 10.0,
            "B_update_freq" => 1,
            "Q_update_freq" => 100,
            "B_opt" => "ADAM",
            "gamma" => 0.99,
            "update_horizon" => 1,
            "batch_size" => 32,
            "min_replay_history" => 100,
            "updates_per_step" => 1,
            "is_enable_double_DQN" => true,
            "traj_capacity" => 100_000,
            "seed" => 1,
        )
    end

    lg = WandbLogger(project="BE",
        name="QFLOW_CartPole",
        config=config,
    )
    save_dir = datadir("sims", "QFLOW", "CartPole", "$(now())")
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
    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))

    if restore === nothing
        """
        CREATE MODEL
        """
        # init = glorot_uniform(rng)
        init(dims...) = (2 .* rand(dims...) .- 1) ./ Float32(sqrt(dims[end]))
        init_σ(dims...) = fill(0.4f0 / Float32(sqrt(dims[end])), dims)

        # B_model = Chain(
        #     NoisyDense(ns, 128, selu; init_μ=init, init_σ=init_σ, rng=device_rng),
        #     NoisyDense(128, 128, selu; init_μ=init, init_σ=init_σ, rng=device_rng),
        #     QSplit(
        #         NoisyDense(128, na; init_μ=init, init_σ=init_σ, rng=device_rng),
        #         # NoisyDense(128, na; init_μ=init, init_σ=init_σ, rng=device_rng),
        #         NoisyDense(128, 8 * na, tanh; init_μ=init, init_σ=init_σ, rng=device_rng),
        #     ),
        # ) |> gpu

        # Q_model = Chain(
        #     NoisyDense(ns, 128, selu; init_μ=init, init_σ=init_σ, rng=device_rng),
        #     NoisyDense(128, 128, selu; init_μ=init, init_σ=init_σ, rng=device_rng),
        #     QSplit(
        #         NoisyDense(128, na; init_μ=init, init_σ=init_σ, rng=device_rng),
        #         # NoisyDense(128, na; init_μ=init, init_σ=init_σ, rng=device_rng),
        #         NoisyDense(128, 8 * na, tanh; init_μ=init, init_σ=init_σ, rng=device_rng),
        #     ),
        # ) |> gpu

        B_model = Chain(
            Dense(ns, 64, selu),
            Dense(64, 64, selu),
            QSplit(
                Dense(64, na),
                # NoisyDense(128, na; init_μ=init, init_σ=init_σ, rng=device_rng),
                Dense(64, 4 * na),
            ),
        ) |> gpu

        Q_model = Chain(
            Dense(ns, 64, selu),
            Dense(64, 64, selu),
            QSplit(
                Dense(64, na),
                # NoisyDense(128, na; init_μ=init, init_σ=init_σ, rng=device_rng),
                Dense(64, 4 * na),
            ),
        ) |> gpu

        Flux.loadparams!(Q_model, Flux.params(B_model))

        flow = PlanarFlow(
            [
            PlanarLayer(4),
            PlanarLayer(4),
            PlanarLayer(4),
            PlanarLayer(4),
            # PlanarLayer(8),
            # PlanarLayer(8),
        ]
        ) |> gpu


        """
        CREATE AGENT
        """
        B_opt = eval(Meta.parse(get_config(lg, "B_opt")))

        agent = Agent(
            policy=QBasedPolicy(
                learner=QFLOWLearner(
                    B_approximator=NeuralNetworkApproximator(
                        model=B_model,
                        optimizer=Optimiser(ClipNorm(get_config(lg, "B_clip_norm")), B_opt(get_config(lg, "B_lr"))),
                    ),
                    Q_approximator=NeuralNetworkApproximator(
                        model=Q_model,
                    ),
                    flow=flow,
                    Q_lr=get_config(lg, "Q_lr"),
                    γ=get_config(lg, "gamma"),
                    update_horizon=get_config(lg, "update_horizon"),
                    batch_size=get_config(lg, "batch_size"),
                    min_replay_history=get_config(lg, "min_replay_history"),
                    B_update_freq=get_config(lg, "B_update_freq"),
                    Q_update_freq=get_config(lg, "Q_update_freq"),
                    is_enable_double_DQN=get_config(lg, "is_enable_double_DQN"),
                ),
                explorer=EpsilonGreedyExplorer(
                    kind=:exp,
                    ϵ_stable=0.01,
                    decay_steps=500,
                    rng=rng,
                ),
            ),
            trajectory=CircularArraySARTTrajectory(
                capacity=get_config(lg, "traj_capacity"),
                state=Vector{Float32} => ns,
            ),
        )
    else
        agent = load(restore; agent)
        # @load restore agent
    end

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
                    p = agent.policy.learner.logging_params

                    KL, H, S, L, Q = p["KL"], p["H"], p["S"], p["𝐿"], p["Q"]
                    B_var, QA, J = p["B_var"], p["QA"], p["J"]
                    @info "training" KL = KL H = H S = S L = L Q = Q B_var = B_var QA = QA J = J

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
                CartPoleEnv(; T=Float32),
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
                end
            catch
                close(lg)
                stop("Program most likely terminated through WandB interface.")
            end
        end,
        DoEveryNEpisode(n=100) do t, agent, env
            @info "Saving agent at step $t to $save_dir"
            jldsave(save_dir * "/model_$t.jld2"; agent)
        end,
        CloseLogger(lg),
    )
    stop_condition = StopAfterStep(30_000, is_show_progress=true)

    """
    RETURN EXPERIMENT
    """
    Experiment(agent, env, stop_condition, hook, "# QFLOW <-> CartPole")
end