using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using Flux: glorot_uniform
using RLExp


function RL.Experiment(
    ::Val{:RLExp},
    ::Val{:IQN},
    ::Val{:CartPole},
    ; seed=1
)
    rng = StableRNG(seed)
    device_rng = rng
    # device_rng = CUDA.functional() ? CUDA.CURAND.RNG() : rng
    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))
    init = glorot_uniform(rng)
    Nₑₘ = 8
    Aₑₘ = 8
    n_hidden = 64
    κ = 1.0f0

    # nn_creator() =
    #     ImplicitQuantileNet(
    #         ψ=Dense(ns, n_hidden, softplus; init=init),
    #         ϕ=PDense(Nₑₘ, n_hidden, leakyrelu; init=init),
    #         header=PDense(n_hidden, na; init=init),
    #     ) |> gpu

    agent = Agent(
        policy=QBasedPolicy(
            learner=IQNPPLearner(
                approximator=Approximator(
                    model=TwinNetwork(
                        ImplicitQuantileNet(
                            ψ=Dense(ns, n_hidden, relu; init=init),
                            ϕ=Dense(Nₑₘ => n_hidden, relu; init=init),
                            header=Dense(n_hidden => na; init=init),
                        ),
                        sync_freq=100
                    ),
                    optimiser=ADAM(0.001, (0.9, 0.999)),
                ),
                κ=κ,
                N=32,
                N′=32,
                Nₑₘ=Nₑₘ,
                K=32,
                γ=0.99f0,
                rng=rng,
                device_rng=device_rng,
            ),
            explorer=EpsilonGreedyExplorer(
                kind=:exp,
                ϵ_stable=0.01,
                decay_steps=500,
                rng=rng,
            ),
        ),
        trajectory=Trajectory(
            container=CircularArraySARTTraces(
                capacity=1000,
                state=Float32 => (ns,),
            ),
            sampler=BatchSampler{SS′ART}(
                batch_size=32,
                rng=rng
            ),
            controller=InsertSampleRatioController(
                threshold=100,
                n_inserted=-1
            )
        )
    )

    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    hook = TotalRewardPerEpisode()
    Experiment(agent, env, stop_condition, hook)
end


#+ tangle=false
using Plots
# pyplot() #hide
ex = E`RLExp_IQN_CartPole`
run(ex)
display(plot(ex.hook.rewards))
# savefig("assets/JuliaRL_IQN_CartPole.png") #hide
