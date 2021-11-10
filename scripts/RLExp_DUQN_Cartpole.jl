using Base.Iterators: tail
using Conda
using CUDA
using Distributions: Uniform, Product
using DrWatson
using Flux
using Logging
using RLExp
using Random
using ReinforcementLearning
using Setfield
using Statistics
using Wandb

applychain(::Tuple{}, x, n) = x
applychain(fs::Tuple, x, n) = applychain(tail(fs), first(fs)(x, n), n)
(c::Chain)(x, n) = applychain(c.layers, x, n)

function RL.Experiment(
    ::Val{:RLExp},
    ::Val{:DUQN},
    ::Val{:Cartpole},
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
    env = CartPoleEnv(; T = Float32, rng = rng)
    ns, na = length(state(env)), length(action_space(env))

    """
    SET UP LOGGING
    """
    simulation = @ntuple seed
    lg = WandbLogger(project = "RLExp", name="DUQN_CartPole" * savename(simulation))
    save_dir = datadir("sims", "DUQN", savename(simulation, "jld2"))

    """
    CREATE MODEL
    """
    init = glorot_uniform(rng)


    model = Chain(
        NoisyDense(ns, 128, relu; init_μ = init),
        NoisyDense(128, 128, relu; init_μ = init),
        Split(NoisyDense(128, na; init_μ = init),
              NoisyDense(128, na, softplus; init_μ = init))
    ) |> cpu
    opt = ADAM(3e-3)

    model2 = Chain(
        Dense(ns, 128, relu; init = init),
        Dense(128, 128, relu; init = init),
        Dense(128, na; init = init),
    ) |> cpu
    opt2 = ADAM(3e-3)

    # u = Product([Uniform(-2.4f0, 2.4f0),
    #                  Uniform(-10.0f0, 10.0f0),
    #                  Uniform(-0.418f0, 0.418f0),
    #                  Uniform(-10.0f0, 10.0f0)])
    # sse = SpectralSteinEstimator(0.05, nothing, 0.99)

    # for _=1:1_000
    #     samples = rand(u, 32)

    #     gs = gradient(params(model)) do
    #         b_all = model(samples, 100)
    #         b_rand = reshape(b_all[1], :, 100)
    #         b_noisy = b_rand
    #         S = entropy_surrogate(sse, permutedims(b_noisy, (2, 1)))

    #         # pr = [1 -1] .* (samples[3,:] .< 0) .+ [-1 1] .* (samples[3,:] .> 0)
    #         pr = [1 2] .* samples[4,:]
    #         # pr += [1 -1] .* (samples[4,:] .< 0) .+ [-1 1] .* (samples[4,:] .> 0)
    #         # pr += [1 -1] .* (samples[2,:] .> 0) .+ [-1 1] .* (samples[2,:] .< 0)
    #         # pr += [1 -1] .* (samples[1,:] .> 0) .+ [-1 1] .* (samples[1,:] .< 0)
    #         pr = reshape(repeat(pr, 1,  100), :, 100)
    #         H = sum((b_noisy .- 10 .* pr) .^ 2 ./ (2 * 100.0f0 .^ 2)) ./ (size(b_noisy, 2) .* 32)
    #         KL = H - S
    #     end
    #     Flux.update!(opt, params(model), gs)
        
    #     gs = gradient(params(model2)) do 
    #         b = model(samples, 100)[1]
    #         B̂ = dropdims(mean(b, dims=ndims(b)), dims=ndims(b))
    #         q = model2(samples)
    #         𝐿 = sum((q .- B̂) .^ 2)
    #         return 𝐿
    #     end
    #     Flux.update!(opt2, params(model2), gs)
    # end

    """
    CREATE AGENT
    """
    agent = Agent(
        policy = QBasedPolicy(
            learner = DUQNLearner(
                B_approximator = NeuralNetworkApproximator(
                    model = model,
                    optimizer = ADAM(3e-3),
                ),
                Q_approximator = NeuralNetworkApproximator(
                    model = model2,
                    optimizer = ADAM(1e-4)
                ),
                γ = 0.99f0,
                update_horizon = 1,
                batch_size = 32,
                min_replay_history = 1,
                B_update_freq = 1,
                Q_update_freq = 5,
            ),
            explorer = GreedyExplorer(),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 1_000_000,
            state = Vector{Float32} => ns,
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
            with_logger(lg) do
                p = agent.policy.learner.logging_params
                KL, MSE, H, S, L, Q, V = p["KL"], p["mse"], p["H"], p["S"], p["𝐿"], p["Q"], p["Σ"]
                @info "training" KL = KL MSE = MSE H = H S = S L = L Q = Q V = V
            end
        end,
        DoEveryNEpisode() do t, agent, env
            with_logger(lg) do
                @info "training" episode_length = step_per_episode.steps[end] reward = reward_per_episode.rewards[end] log_step_increment = 0
            end
        end,
        DoOnExit(() -> close(lg))
    )
    stop_condition = StopAfterStep(10_000, is_show_progress=true)

    """
    RETURN EXPERIMENT
    """
    Experiment(agent, env, stop_condition, hook, "# DUQN <-> CartPole")
end