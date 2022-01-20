using Base.Iterators: tail
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
    ::Val{:Cartpole},
    name,
   )

   """
   SET UP LOGGING
   """
   lg = WandbLogger(project = "RLExp",
                    name="Noisy_CartPole",
                    config = Dict(
                       "B_lr" => 1e-4,
                       "Q_lr" => 1.0,
                       "B_clip_norm" => 1000.0,
                       "B_update_freq" => 1,
                       "Q_update_freq" => 1000,
                       "B_opt" => "ADAM",
                       "gamma" => 0.99,
                       "update_horizon" => 1,
                       "batch_size" => 32,
                       "min_replay_history" => 32,
                       "updates_per_step" => 1,
                       "λ" => 1.0,
                       # "prior" => "GaussianPrior(0, 10)",
                       "prior" => "FlatPrior()",
                       "n_samples" => 100,
                       "η" => 0.01,
                       "nev" => 10,
                       "is_enable_double_DQN" => true,
                       "traj_capacity" => 1_000_000,
                       "seed" => 1,
                    ),
   )
   save_dir = datadir("sims", "Noisy", "CartPole", "$(now())")

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
   env = CartPoleEnv(; T = Float32, rng = rng)
   ns, na = length(state(env)), length(action_space(env))

   """
   CREATE MODEL
   """
   # init = glorot_uniform(rng)
    init(a, b) = (2 .* rand(a, b) .- 1) ./ sqrt(b)
   init_σ(dims...) = fill(0.4f0 / Float32(sqrt(dims[end])), dims)


    agent = Agent(
        policy = QBasedPolicy(
            learner = DQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                            NoisyDense(ns, 128, relu; init_μ = init, init_σ = init_σ, rng = device_rng),
                            NoisyDense(128, 128, relu; init_μ = init, init_σ = init_σ, rng = device_rng),
                            NoisyDense(128, na; init_μ = init, init_σ = init_σ, rng = device_rng),
                    ),
                    optimizer = ADAM(1e-4),
                ) |> gpu,
                target_approximator = NeuralNetworkApproximator(
                    model = Chain(
                            NoisyDense(ns, 128, relu; init_μ = init, init_σ = init_σ, rng = device_rng),
                            NoisyDense(128, 128, relu; init_μ = init, init_σ = init_σ, rng = device_rng),
                            NoisyDense(128, na; init_μ = init, init_σ = init_σ, rng = device_rng),
                        ),
                    optimizer = ADAM(1e-4),
                ) |> gpu,
                loss_func = mse,
                stack_size = nothing,
                batch_size = 32,
                update_horizon = 1,
                min_replay_history = 32,
                update_freq = 1,
                target_update_freq = 1_000,
                rng = rng,
            ),
            explorer = GreedyExplorer(),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 1_000_000,
            state = Vector{Float32} => (ns,),
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

            @info "finished evaluating agent in $(round(s, digits=2)) seconds" avg_length = avg_length avg_score = avg_score
            try
                with_logger(lg) do
                    @info "evaluating" avg_length = avg_length avg_score = avg_score log_step_increment = step_per_episode.steps[end]
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
    stop_condition = StopAfterStep(30_000, is_show_progress=true)

    """
    RETURN EXPERIMENT
    """
    Experiment(agent, env, stop_condition, hook, "# Noisy <-> CartPole")
end