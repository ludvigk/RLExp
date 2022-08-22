export QQFLOWLearner, FlowNet
import ReinforcementLearning.RLBase

using DataStructures: DefaultDict
using StatsBase: sample
# using Flux.Losses
using CUDA: randn
using MLUtils
using SpecialFunctions
using ChainRules: ignore_derivatives, @ignore_derivatives
# using StatsFuns
using RLExp
import Statistics.mean
using Functors: @functor

# const erfratio = Float32(sqrt(2π) * erf(1/sqrt(2)) / (sqrt(2π) * erf(1/sqrt(2)) + 2exp(-1/2)))

function v2(x, b, c, d)
    xc = x .- c
    axc = abs.(xc)
    u = max.(axc, b)
    excu = exp.(axc .- u)
    exbu = exp.(b .- u)
    exu = exp.(-u)
    s = excu .+ exbu .- exu
    r = u .+ log.(s) .- b
    out = sign.(xc) .* r .+ d
    # dupper = excu
    # dlower = s
    out, log.(excu ./ s)
end

v2⁻¹(x, b, c, d) = v2(x, -b, d, c)

# function v2⁻¹(x, b, c, d)
#     xc = x .- c
#     axc = abs.(xc)
#     u = max.(axc, b)
#     s = exp.(axc .- u) .+ exp.(b .- u) .- exp.(-u)
#     r = u .+ log.(s) .- b
#     out = sign.(xc) .* r .+ d
#     out, 0f0
# end

Base.@kwdef struct FlowNet{P}
    net::P
end

@functor FlowNet (net,)

function (m::FlowNet)(state::AbstractArray, num_samples::Int, na::Int)
    ξ = m.net(state)
    i = size(ξ, 1)
    μ = @inbounds ξ[i:i,:]
    # ρ = @inbounds ξ[(na+1):(2na),:]
    # σ = Flux.softplus.(ρ) 
    
    z = @ignore_derivatives randn!(similar(μ, size(μ)..., num_samples))
    # μcpu = cpu(μ)
    # r = Zygote.@ignore rand!(similar(μcpu, size(μ)..., num_samples))
    # tn = Zygote.@ignore rand!(TruncatedNormal(0,1,-1,1), similar(μcpu, size(μ)..., num_samples))
    # lap = Zygote.@ignore rand!(Exponential(), similar(μcpu, size(μ)..., num_samples))
    # sig = Zygote.@ignore sign.(rand!(similar(μcpu, size(μ)..., num_samples)) .- 0.5)
    # z = Zygote.@ignore (r .< erfratio) .* tn .+ (r .> erfratio) .* (lap .+ 1) .* sig
    # z = gpu(z)

    # lz = Zygote.@ignore fill!(similar(z), 0f0)
    
    μ = reshape(μ, size(μ)..., 1)
    # σ = reshape(σ, size(σ)..., 1)
    
    for i=1:(3na):(size(ξ,1) - 3na)
        b = @inbounds ξ[i:(i + na - 1), :]
        c = @inbounds ξ[(i+na):(i + 2na - 1), :]
        d = @inbounds ξ[(i+2na):(i + 3na - 1), :]
        z, _ = v2(z, b, c, d)
        # lz = lz .+ lz_
    end
    z = μ .+ z
    # z = μ .+ z .* σ
    z, nothing
end

function (m::FlowNet)(z::AbstractArray, state::AbstractArray, na::Int)
    ξ = m.net(state)
    i = size(ξ, 1)
    μ = @inbounds ξ[i:i,:]
    # ρ = @inbounds ξ[(na+1):(2na),:]
    # σ = Flux.softplus.(ρ)

    lz = @ignore_derivatives fill!(similar(z), 0f0)

    μ = reshape(μ, size(μ)..., 1)
    # σ = reshape(σ, size(σ)..., 1)
    
    z = z .- ignore_derivatives(μ)
    # z = (z .- μ) ./ σ
    for i=(size(ξ,1) - 3na):(-3na):1
        b = @inbounds ξ[i:(i + na - 1), :]
        c = @inbounds ξ[(i+na):(i + 2na - 1), :]
        d = @inbounds ξ[(i+2na):(i + 3na - 1), :]
        z, lz_ = v2⁻¹(z, b, c, d)
        lz = lz .+ lz_
    end
    z, lz, μ, nothing
end

mutable struct QQFLOWLearner{A<:Approximator{<:TwinNetwork}} <: AbstractLearner
    approximator::A
    n_actions::Int
    n_samples_act::Int
    n_samples_target::Int
    update_horizon::Int
    γ::Float32
    rng::AbstractRNG
    is_enable_double_DQN::Bool
    training::Bool
    logging_params
end

function QQFLOWLearner(;
    approximator::A,
    n_actions::Int,
    γ::Real=0.99f0,
    update_horizon::Int=1,
    n_samples_act::Int=30,
    n_samples_target::Int=30,
    is_enable_double_DQN::Bool=false,
    training::Bool=true,
    rng=Random.GLOBAL_RNG
) where {A}
    return QQFLOWLearner(
        approximator,
        n_actions,
        n_samples_act,
        n_samples_target,
        update_horizon,
        Float32(γ),
        rng,
        is_enable_double_DQN,
        training,
        DefaultDict(0.0),
    )
end

Flux.@functor QQFLOWLearner (approximator,)

function (L::QQFLOWLearner)(s::AbstractArray)
    q = L.approximator(s, L.n_samples_act, L.n_actions)[1]
    q = dropdims(mean(q, dims=3), dims=3)
end

function (learner::QQFLOWLearner)(env::AbstractEnv)
    s = send_to_device(device(learner.approximator), state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    s |> learner |> vec |> send_to_host
end

function RLBase.optimise!(learner::QQFLOWLearner, batch::NamedTuple)
    A = learner.approximator
    Z = A.model.source
    Zₜ = A.model.target
    n_actions = learner.n_actions
    n_samples_target = learner.n_samples_target
    γ = learner.γ
    update_horizon = learner.update_horizon
    lp = learner.logging_params
    
    D = device(Z)
    states = send_to_device(D, collect(batch.state))
    rewards = send_to_device(D, batch.reward)
    terminals = send_to_device(D, batch.terminal)
    next_states = send_to_device(D, collect(batch.next_state))

    batch_size = length(terminals)
    actions = CartesianIndex.(batch.action, 1:batch_size)

    if learner.is_enable_double_DQN
        q_values = Z(next_states, n_samples_target, n_actions)[1]
    else
        q_values = Zₜ(next_states, n_samples_target, n_actions)[1]
    end

    mean_q = dropdims(mean(q_values, dims=3), dims=3)

    if haskey(batch, :next_legal_actions_mask)
        l′ = send_to_device(D, batch[:next_legal_actions_mask])
        q_values .+= ifelse.(l′, 0.0f0, typemin(Float32))
    end

    selected_actions = dropdims(argmax(mean_q; dims=1); dims=1)
    if learner.is_enable_double_DQN
        q_values = Zₜ(next_states, n_samples_target, n_actions)[1]
    end
    next_q = @inbounds q_values[selected_actions, :]

    target_distribution =
        Flux.unsqueeze(rewards, 2) .+
        Flux.unsqueeze(γ^update_horizon .* (1 .- terminals), 2) .* next_q
    target_distribution = Flux.unsqueeze(target_distribution, 1)
    # target_distribution = repeat(Flux.unsqueeze(target_distribution, 1),
    #                              n_actions, 1, 1)
    # target_distribution = reshape(target_distribution, size(target_distribution))

    gs = gradient(Flux.params(Z)) do
        preds, sldj, μ, σ = Z(target_distribution, states, n_actions)

        nll = preds[actions, :] .^ 2 ./ 2
        
        # abs_error = abs.(TD_error)
        # quadratic = min.(abs_error, 1)
        # linear = abs_error .- quadratic
        # nll = 0.5f0 .* quadratic .* quadratic .+ 1 .* linear

        m = sum(target_distribution, dims=3) ./ n_samples_target
        extra_loss = Flux.huber_loss(μ, m) ./ 100
        # extra_loss = sum((μ .- m) .^ 2)
        sldj = sldj[actions, :]
        loss = (sum(nll) - sum(sldj)) / n_samples_target #+ sum(log.(σ[actions, :]))
        # loss = (sum(nll) - sum(sldj)) / n_samples_target + extra_loss
        loss = (loss + extra_loss) / batch_size

        ignore_derivatives() do
            lp["loss"] = loss
            lp["nll"] = sum(nll) / (batch_size * n_samples_target)
            lp["sldj"] = sum(sldj) / (batch_size * n_samples_target)
            lp["Qₜ"] = sum(target_distribution) / length(target_distribution)
            lp["QA"] = sum(selected_actions)[1] / length(selected_actions)
            lp["mu"] = sum(μ) / length(μ)
            # lp["sigma"] = sum(σ[actions,:]) / length(σ[actions,:])
            lp["max_weight"] = maximum(maximum.(Flux.params(Z)))
            lp["min_weight"] = minimum(minimum.(Flux.params(Z)))
            lp["max_pred"] = maximum(preds)
            lp["min_pred"] = minimum(preds)
            for i = 1:n_actions
                lp["Q$i"] = sum(target_distribution[i,:]) / batch_size
            end
        end

        return loss
    end
    optimise!(A, gs)
end
