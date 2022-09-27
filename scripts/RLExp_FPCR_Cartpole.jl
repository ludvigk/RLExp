using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using Flux: glorot_uniform
using RLExp

function RL.Experiment(
    ::Val{:RLExp},
    ::Val{:FPCR},
    ::Val{:CartPole},
    ; seed=123
)
    rng = StableRNG(seed)
    device_rng = rng
    # device_rng = CUDA.functional() ? CUDA.CURAND.RNG() : rng
    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))
    # init = glorot_uniform(rng)
    κ = 1.0f0

    net = FPCRNet(
        base=Chain(
            Dense(ns => 64, relu),
            # Dense(64 => 64, relu),
        ),
        support=SupportProposalNet(64, 32, na),
        cdf=CDFNet(
            Chain(
                # Dense(64 => 64, relu),
                Dense(64 => 64na),
            ), 32),
    )

    agent = Agent(
        policy=QBasedPolicy(
            learner=FPCRLearner(
                approximator=Approximator(
                    model=TwinNetwork(
                        net,
                        sync_freq=1,
                        ρ=0.999f0,
                    ),
                    optimiser=ADAM(2.5e-9),
                ),
                support_optimiser=RMSProp(2.5f-4),
                ent_coef=0.001f0,
                κ=κ,
                γ=0.99f0,
                rng=rng,
                device_rng=device_rng,
            ),
            explorer=EpsilonGreedyExplorer(
                kind=:exp,
                ϵ_stable=0.025,
                decay_steps=5000,
                rng=rng,
            ),
        ),
        trajectory=Trajectory(
            container=CircularArraySARTTraces(
                capacity=20_000,
                state=Float32 => (ns,),
            ),
            sampler=BatchSampler{SS′ART}(
                batch_size=64,
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
ex = E`RLExp_FPCR_CartPole`
run(ex)
display(plot(ex.hook.rewards))
# savefig("assets/JuliaRL_IQN_CartPole.png") #hide
