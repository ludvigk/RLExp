using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using Flux: glorot_uniform
using RLExp

function _create_bias(weights::AbstractArray, bias::Bool, dims::Integer...)
    bias ? fill!(similar(weights, dims...), 0) : false
end
function _create_bias(weights::AbstractArray, bias::AbstractArray, dims::Integer...)
    size(bias) == dims || throw(DimensionMismatch("expected bias of size $(dims), got size $(size(bias))"))
    bias
end

struct PDense{F, M<:AbstractMatrix, B}
    weight::M
    bias::B
    σ::F
    function PDense(W::M, bias = true, σ::F = identity) where {M<:AbstractMatrix, F}
        b = _create_bias(W, bias, size(W,1))
        new{F,M,typeof(b)}(W, b, σ)
    end
end
  
function PDense((in, out)::Pair{<:Integer, <:Integer}, σ = identity;
                init = glorot_uniform, bias = true)
    PDense(init(out, in), bias, σ)
end

Flux.@functor PDense

function (a::PDense)(x)
    σ = NNlib.fast_act(a.σ, x)
    w = abs.(a.weight)
    return σ.(w * x .+ a.bias)
end

struct MonotonicDense{F, M<:AbstractMatrix, B, I}
    convexity::I
    weight::M
    bias::B
    σ::F
end

Flux.@functor MonotonicDense


function MonotonicDense((in, out)::Pair{<:Integer, <:Integer}, σ = identity;
    init = glorot_uniform, bias = true, ϵ=0.5)
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
    ::Val{:JuliaRL},
    ::Val{:IQN},
    ::Val{:CartPole},
    ; seed=123
)
    rng = StableRNG(seed)
    device_rng = rng
    # device_rng = CUDA.functional() ? CUDA.CURAND.RNG() : rng
    env = CartPoleEnv(; T=Float32, rng=rng)
    ns, na = length(state(env)), length(action_space(env))
    init = glorot_uniform(rng)
    Nₑₘ = 16
    n_hidden = 64
    κ = 1.0f0

    # nn_creator() =
    #     ImplicitQuantileNet(
    #         ψ=Dense(ns, n_hidden, softplus; init=init),
    #         ϕ=PDense(Nₑₘ, n_hidden, leakyrelu; init=init),
    #         header=PDense(n_hidden, na; init=init),
    #     ) |> gpu

    net = 

    agent = Agent(
        policy=QBasedPolicy(
            learner=IQNPPLearner(
                approximator=Approximator(
                    model=TwinNetwork(
                        ImplicitQuantileNet(
                            ψ=Dense(ns, n_hidden, relu; init=init),
                            ϕ=MonotonicDense(Nₑₘ => n_hidden, relu; init=init),
                            header=MonotonicDense(n_hidden => na; init=init),
                        ),
                        sync_freq=100
                    ),
                    optimiser=ADAM(0.001),
                ),
                κ=κ,
                N=8,
                N′=8,
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
ex = E`JuliaRL_IQN_CartPole`
run(ex)
display(plot(ex.hook.rewards))
# savefig("assets/JuliaRL_IQN_CartPole.png") #hide
