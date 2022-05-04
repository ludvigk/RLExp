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
using StableRNGs
using Statistics
using Wandb

function RL.Experiment(
    ::Val{:RLExp},
    ::Val{:FAC},
    ::Val{:Cartpole},
    name;
    save_dir=nothing,
    seed=123
)
    lg = WandbLogger(project="FAC",
        name="CartPole",
    )
    save_dir = datadir("sims", "FAC", "CartPole", "$(now())")
    mkpath(save_dir)

    rng = StableRNG(seed)
    N_ENV = 16
    UPDATE_FREQ = 10
    N_EVAL = 1_000
    init(a, b) = (2 .* rand(a, b) .- 1) ./ sqrt(b)
    init_σ(dims...) = fill(0.05f0 / Float32(sqrt(dims[end])), dims)
    env = MultiThreadEnv([
        CartPoleEnv(; T=Float32, rng=StableRNG(hash(seed + i))) for i in 1:N_ENV
    ])
    ns, na = length(state(env[1])), length(action_space(env[1]))
    RLBase.reset!(env, is_force=true)

    init(dims...) = (2 .* rand(dims...) .- 1) ./ Float32(sqrt(dims[end]))
    init_σ(dims...) = fill(0.4f0 / Float32(sqrt(dims[end])), dims)

    agent = Agent(
        policy=FACPolicy(
            approximator=NeuralNetworkApproximator(
                model=Chain(
                    NoisyDense(ns, 256, relu; init_μ=init, init_σ=init_σ),
                    NoisyDense(256, na, relu; init_μ=init, init_σ=init_σ),
                ),
                optimizer=ADAM(1e-3),
            ) |> gpu,
            baseline=NeuralNetworkApproximator(
                model=Chain(
                    Dense(ns, 256, relu; init=glorot_uniform(rng)),
                    Dense(256, 1; init=glorot_uniform(rng)),
                ),
                optimizer=ADAM(1e-3),
            ) |> gpu,
            γ=0.99f0,
            λ=0.97f0,
            actor_loss_weight=1.0f0,
            critic_loss_weight=0.5f0,
            entropy_loss_weight=0.001f0,
            update_freq=UPDATE_FREQ,
        ),
        explorer=BatchExplorer(GumbelSoftmaxExplorer()),
        trajectory=CircularArraySARTTrajectory(;
            capacity=UPDATE_FREQ,
            state=Matrix{Float32} => (ns, N_ENV),
            action=Vector{Int} => (N_ENV,),
            reward=Vector{Float32} => (N_ENV,),
            terminal=Vector{Bool} => (N_ENV,)
        ),
    )

    stop_condition = StopAfterStep(50_000, is_show_progress=true)
    reward_per_episode = TotalBatchRewardPerEpisode(N_ENV)
    batch_steps_per_episode = BatchStepsPerEpisode(N_ENV)

    hook = ComposedHook(
        reward_per_episode,
        batch_steps_per_episode,
        DoEveryNStep(; n=N_EVAL) do t, agent, env
            @info "evaluating agent at $t step..."
            p = agent.policy
            h = ComposedHook(
                TotalBatchRewardPerEpisode(N_ENV),
                BatchStepsPerEpisode(N_ENV),
            )
            s = @elapsed run(
                p,
                MultiThreadEnv([
                    CartPoleEnv(; T=Float32, rng=StableRNG(hash(seed + i))) for i in 1:N_ENV
                ]),
                StopAfterStep(1_000; is_show_progress=false),
                h,
            )
            avg_score = mean(Iterators.flatten(h[1].rewards))
            avg_length = mean(Iterators.flatten(h[2].steps))

            @info "finished evaluating agent in $(round(s, digits=2)) seconds" avg_length = avg_length avg_score = avg_score
            try
                with_logger(lg) do
                    @info "evaluating" avg_length = avg_length avg_score = avg_score log_step_increment = N_EVAL
                    @info "training" reward_per_episode.rewards[end] log_step_increment = 0
                    @info "training" episode = t log_step_increment = 0
                end
            catch
                close(lg)
                stop("Program most likely terminated through WandB interface.")
            end
        end,
        CloseLogger(lg),
    )
    Experiment(agent, env, stop_condition, hook, "# FAC with Cartpole")
end