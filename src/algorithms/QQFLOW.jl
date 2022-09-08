export QQFLOWLearner, FlowNet
import ReinforcementLearning.RLBase

using DataStructures: DefaultDict
using StatsBase: sample
# using Flux.Losses
using CUDA: randn
using MLUtils
using SpecialFunctions
using ChainRules: ignore_derivatives, @ignore_derivatives
using ChainRulesCore
# using StatsFuns
using RLExp
import Statistics.mean
using Functors: @functor
using Base.Broadcast: broadcasted
using Zygote: @adjoint

# const erfratio = Float32(sqrt(2π) * erf(1/sqrt(2)) / (sqrt(2π) * erf(1/sqrt(2)) + 2exp(-1/2)))

myexp(x) = exp(x)

Zygote.@adjoint function broadcasted(::typeof(myexp), x)
    myexp.(x), ȳ -> (nothing, ȳ .* exp.(x))
end

mylog(x) = mylog(x)

Zygote.@adjoint function broadcasted(::typeof(mylog), x)
    mylog.(x), ȳ -> (nothing, ȳ .* inv.(x))
end

function v2(x, b, c, d)
    xc = x .- c
    axc = abs.(xc)
    u = max.(axc, b)
    excu = myexp.(axc .- u)
    exbu = myexp.(b .- u)
    exu = myexp.(-u)
    s = excu .+ exbu .- exu
    r = u .+ log.(s) .- b
    out = sign.(xc) .* r .+ d
    out, log.(excu ./ s)
end

v2⁻¹(x, b, c, d) = v2(x, -b, d, c)

function v3(x, b, c, d)
    xc = x - c
    axc = abs(xc)
    u = max(axc, b)
    excu = exp(axc - u)
    exbu = exp(b - u)
    exu = exp(-u)
    s = excu + exbu - exu
    r = u + log(s) - b
    out = sign(xc) * r + d
    out
end

function dv3(x, b, c, d)
    xc = x - c
    axc = abs(xc)
    u = max(axc, b)
    excu = exp(axc - u)
    exbu = exp(b - u)
    exu = exp(-u)
    s = excu + exbu - exu
    log(excu / s)
end

@adjoint function v3(x, b, c, d)
    xc = x .- c
    axc = abs(xc)
    u = max(axc, b)
    sxc = sign(xc)
    excu = exp(axc - u)
    exbu = exp(b - u)
    exu = exp(-u)
    s = excu + exbu - exu
    r = u + log(s) - b
    out = sxc * r + d
    Δb = sxc * (exbu / s - 1)
    Δx = excu / s
    Δc = -Δx
    out, cc -> (cc .* Δx, cc .* Δb, cc .* Δc, cc)
end

# function ChainRulesCore.rrule(::typeof(v3), x, b, c, d)
#     xc = x - c
#     axc = abs(xc)
#     u = max(axc, b)
#     excu = myexp(axc - u)
#     exbu = myexp(b - u)
#     exu = myexp(-u)
#     s = excu + exbu - exu
#     r = u + log(s) - b
#     out = sxcu * r + d
#     Δb = sxc * (exbu / s - 1)
#     Δx = excu / s
#     Δc = -Δx
#     out, cc -> (nothing, cc * Δx, cc * Δb, cc * Δc, cc)
# end

@adjoint function dv3(x, b, c, d)
    xc = x .- c
    sxc = sign.(xc)
    axc = abs.(xc)
    u = max(axc, b)
    excu = exp(axc - u)
    excuu = exp(axc - 2u)
    exbu = exp(b - u)
    excbu = exp(axc + b - 2u)
    exu = exp(-u)
    s = excu + exbu - exu
    out = log(excu / s)
    s2 = s ^ 2
    Δx = sxc * (excbu - excuu) / s2
    Δb = -excbu / s2
    Δc = -Δx
    out, cc -> (cc .* Δx, cc .* Δb, cc .* Δc, nothing)
end

# function ChainRulesCore.rrule(::typeof(dv3), x, b, c, d)
#     xc = x - c
#     axc = abs(xc)
#     u = max(axc, b)
#     excu = myexp(axc - u)
#     exbu = myexp(b - u)
#     exu = myexp(-u)
#     s = excu + exbu - exu
#     log(excu / s)
#     s2 = s ^ 2
#     Δx = -sxc * excu * (exbu - 1) / s2
#     Δb = -myexp(axc + b - u) / s2
#     Δc = -Δx
#     out, cc -> (nothing, cc * Δx, cc * Δb, cc * Δc, nothing)
# end

v3⁻¹(x, b, c, d) = v3(x, -b, d, c)
dv3⁻¹(x, b, c, d) = dv3(x, -b, d, c)

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
    # μ = @inbounds ξ[i:i,:]
    # ρ = @inbounds ξ[(na+1):(2na),:]
    # σ = Flux.softplus.(ρ) 
    
    z = @ignore_derivatives randn!(similar(ξ, 1, size(ξ, 2), num_samples))
    pz = z
    # μcpu = cpu(μ)
    # r = Zygote.@ignore rand!(similar(μcpu, size(μ)..., num_samples))
    # tn = Zygote.@ignore rand!(TruncatedNormal(0,1,-1,1), similar(μcpu, size(μ)..., num_samples))
    # lap = Zygote.@ignore rand!(Exponential(), similar(μcpu, size(μ)..., num_samples))
    # sig = Zygote.@ignore sign.(rand!(similar(μcpu, size(μ)..., num_samples)) .- 0.5)
    # z = Zygote.@ignore (r .< erfratio) .* tn .+ (r .> erfratio) .* (lap .+ 1) .* sig
    # z = gpu(z)

    # lz = Zygote.@ignore fill!(similar(z), 0f0)
    
    # μ = reshape(μ, size(μ)..., 1)
    # σ = reshape(σ, size(σ)..., 1)
    
    @inbounds for i=1:(3na):(size(ξ,1) - 3na + 1)
        b = ξ[i:(i + na - 1), :]
        c = ξ[(i+na):(i + 2na - 1), :]
        d = ξ[(i+2na):(i + 3na - 1), :]
        z = v3⁻¹.(z, b, c, d)
        # lz = lz .+ lz_
    end
    # z = μ .+ z
    # z = μ .+ z .* σ
    z, pz
end

function (m::FlowNet)(z::AbstractArray{Float32}, state::AbstractArray{Float32}, na::Int)
    ξ = m.net(state)
    # i = size(ξ, 1)
    # μ = @inbounds ξ[i:i,:]
    # ρ = @inbounds ξ[(na+1):(2na),:]
    # σ = Flux.softplus.(ρ)

    lz = @ignore_derivatives fill!(similar(z), 0f0)

    # μ = reshape(μ, size(μ)..., 1)
    # σ = reshape(σ, size(σ)..., 1)
    
    # z = z .- ignore_derivatives(μ)
    # z = (z .- μ) ./ σ
    @inbounds for i=(size(ξ,1) - 3na + 1):(-3na):1
        b = ξ[i:(i + na - 1), :, :]
        c = ξ[(i+na):(i + 2na - 1), :, :]
        d = ξ[(i+2na):(i + 3na - 1), :, :]
        lz_ = dv3.(z, b, c, d)
        z = v3.(z, b, c, d)
        lz = lz .+ lz_
    end
    z, lz
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
    # states = DenseCuArray(batch.state)
    states = send_to_device(D, collect(batch.state))
    rewards = send_to_device(D, batch.reward)
    terminals = send_to_device(D, batch.terminal)
    # next_states = DenseCuArray(batch.next_state)
    next_states = send_to_device(D, collect(batch.next_state))

    batch_size = length(terminals)
    actions = CartesianIndex.(batch.action, 1:batch_size)

    if learner.is_enable_double_DQN
        q_values, pz = Z(next_states, n_samples_target, n_actions)
    else
        # q_values, pz = Z(next_states, n_samples_target, n_actions)
        q_values, pz = Zₜ(next_states, n_samples_target, n_actions)
    end

    mean_q = dropdims(mean(q_values, dims=3), dims=3)

    if haskey(batch, :next_legal_actions_mask)
        l′ = send_to_device(D, batch[:next_legal_actions_mask])
        q_values .+= ifelse.(l′, 0.0f0, typemin(Float32))
    end

    selected_actions = dropdims(argmax(mean_q; dims=1); dims=1)
    if learner.is_enable_double_DQN
        q_values, pz = Zₜ(next_states, n_samples_target, n_actions)
    end
    next_q = @inbounds q_values[selected_actions, :]

    target_distribution =
        Flux.unsqueeze(rewards, 2) .+
        Flux.unsqueeze(γ^update_horizon .* (1 .- terminals), 2) .* next_q
    target_distribution = Flux.unsqueeze(target_distribution, 1)
    # target_distribution = repeat(Flux.unsqueeze(target_distribution, 1),
    #                              n_actions, 1, 1)
    # target_distribution = reshape(target_distribution, size(target_distribution))
    nll2 = pz[1,:,:] .^ 2 ./ 2
    gs = gradient(Flux.params(Z)) do
        preds, sldj = Z(target_distribution, states, n_actions)

        nll = preds[actions, :] .^ 2 ./ 2 .- sldj[actions, :]

        # loss = Flux.huber_loss(nll, nll2)
        loss = mean(nll)

        ignore_derivatives() do
            lp["loss"] = loss
            # lp["extra_loss"] = extra_loss
            # lp["sldj"] = sum(sldj) / (batch_size * n_samples_target)
            # lp["Qₜ"] = sum(target_distribution) / length(target_distribution)
            # lp["QA"] = sum(selected_actions)[1] / length(selected_actions)
            # lp["mu"] = mean(μ)
            # lp["sigma"] = sum(σ[actions,:]) / length(σ[actions,:])
            lp["max_weight"] = maximum(maximum.(Flux.params(Z)))
            lp["min_weight"] = minimum(minimum.(Flux.params(Z)))
            lp["max_pred"] = maximum(preds)
            lp["min_pred"] = minimum(preds)
            # for i = 1:n_actions
            #     lp["Q$i"] = sum(target_distribution[i,:]) / batch_size
            # end
        end

        return loss
    end
    optimise!(A, gs)
end
