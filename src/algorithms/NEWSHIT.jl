export NEWSHITLearner, FlowNet
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
using Flux: chunk
using Distributions


function mixture_gauss_cdf(x, weights, loc, log_scales)
    x = repeat(Flux.unsqueeze(x, 1), size(weights, 1))
    component_dist = Normal.(loc, exp.(log_scales))
    z_cdf = cdf.(component_dist, x)
    weights = softmax(weights)
    return sum(z_cdf .* weights, dims=1)
end

function compute_forward(x, params, na)
    weights, loc, log_scale = chunk(params, 3, dims=1)

    weights = reshape(weights, :, na, size(weights, 2))
    loc = reshape(loc, :, na, size(loc, 2))
    log_scale = reshape(log_scale, :, na, size(log_scale, 2))

    z = mixture_gauss_cdf(x, weights, loc, log_scale)
end

function mixture_inv_cdf(x, prior_logits, means, log_scales; max_it=100, eps=1.0f-10)
    z = zero(x) |> gpu
    max_scales = sum(exp.(log_scales), dims=1)
    t = ones(eltype(x), size(x)) |> gpu
    lb = minimum(means .- 20 .* max_scales, dims=1)
    lb = lb .* t
    ub = maximum(means .+ 20 .* max_scales, dims=1)
    ub = ub .* t

    for _ = 1:max_it
        old_z = z
        y = mixture_gauss_cdf(z, prior_logits, means, log_scales)
        gt = convert(typeof(x), y .> x)
        lt = 1 .- gt
        z = gt .* (old_z .+ lb) ./ 2 .+ lt .* (old_z .+ ub) ./ 2
        lb = gt .* lb .+ lt .* old_z
        ub = gt .* old_z .+ lt .* ub
        if maximum(abs.(z .- old_z)) < eps
            break
        end
    end
    return z
end

function compute_backward(x, params, na; eps=1.0f-5)
    weights, loc, log_scale = chunk(params, 3, dims=1)
    weights = reshape(weights, :, na, size(weights, 2), 1)
    loc = reshape(loc, :, na, size(loc, 2), 1)
    log_scale = reshape(log_scale, :, na, size(log_scale, 2), 1)

    clamp!(x, eps, 1 - eps)
    z = mixture_inv_cdf(x, weights, loc, log_scale)
end

Base.@kwdef struct FlowNet{P}
    net::P
end

@functor FlowNet (net,)

function (m::FlowNet)(state::AbstractArray, num_samples::Int, na::Int)
    ξ = m.net(state)
    forward
    z
end

function (m::FlowNet)(z::AbstractArray, state::AbstractArray, na::Int)
    ξ = m.net(state)

    lz = @ignore_derivatives fill!(similar(z), 0.0f0)

    @inbounds for i = (size(ξ, 1)-3na):(-3na):1
        b = ξ[i:(i+na-1), :, :]
        c = ξ[(i+na):(i+2na-1), :, :]
        d = ξ[(i+2na):(i+3na-1), :, :]
        lz_ = dv3.(z, b, c, d)
        z = v3.(z, b, c, d)
        lz = lz .+ lz_
    end
    z, lz
end

mutable struct NEWSHITLearner{A<:Approximator{<:TwinNetwork}} <: AbstractLearner
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
    states::Array{Float32,2}
    next_states::Array{Float32,2}
end

function NEWSHITLearner(;
    approximator::A,
    n_actions::Int,
    γ::Real=0.99f0,
    update_horizon::Int=1,
    n_samples_act::Int=30,
    n_samples_target::Int=30,
    is_enable_double_DQN::Bool=false,
    training::Bool=true,
    batch_size=32,
    rng=Random.GLOBAL_RNG
) where {A}
    return NEWSHITLearner(
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
        Array{Float32,2}(undef, n_actions, batch_size),
        Array{Float32,2}(undef, n_actions, batch_size),
    )
end

Flux.@functor NEWSHITLearner (approximator,)

function (L::NEWSHITLearner)(s::AbstractArray)
    ξ = L.approximator(s)
    quant_samples = rand(L.n_actions, 1, L.n_samples_act) # try other methods
    q = compute_backward(quant_samples, ξ, L.n_actions)

    q = dropdims(sum(q, dims=3), dims=3)
end

function (learner::NEWSHITLearner)(env::AbstractEnv)
    s = send_to_device(device(learner.approximator), state(env))
    s = Flux.unsqueeze(s, ndims(s) + 1)
    s |> learner |> vec |> send_to_host
end

function RLBase.optimise!(learner::NEWSHITLearner, batch::NamedTuple)
    A = learner.approximator
    Z = A.model.source
    Zₜ = A.model.target
    n_actions = learner.n_actions
    n_samples_target = learner.n_samples_target
    γ = learner.γ
    update_horizon = learner.update_horizon
    lp = learner.logging_params

    D = device(Z)
    states = DenseCuArray(batch.state)
    # copyto!(learner.states, batch.state)
    # states = send_to_device(D, learner.states)
    # states = send_to_device(D, collect(batch.state))
    rewards = send_to_device(D, batch.reward)
    terminals = send_to_device(D, batch.terminal)
    next_states = DenseCuArray(batch.next_state)
    # copyto!(learner.next_states, batch.next_state)
    # next_states = send_to_device(D, learner.next_states)
    # next_states = send_to_device(D, collect(batch.next_state))

    batch_size = length(terminals)
    actions = CartesianIndex.(batch.action, 1:batch_size)

    ξₜ = Zₜ(next_states)
    quantₜ_samples = rand(n_actions, batch_size, n_samples_target) # try other methods
    quantₜ = compute_backward(quantₜ_samples, ξₜ, n_actions)


    if haskey(batch, :next_legal_actions_mask)
        l′ = send_to_device(D, batch[:next_legal_actions_mask])
        q_values .+= ifelse.(l′, 0.0f0, typemin(Float32))
    end
    mean_q = dropdims(sum(quantₜ, dims=3), dims=3) ./ size(quantₜ, 3)
    selected_actions = dropdims(argmax(mean_q; dims=1); dims=1)
    # if learner.is_enable_double_DQN
    #     q_values = Zₜ(next_states, n_samples_target, n_actions)[1]
    # end
    # next_q = @inbounds q_values[selected_actions, :]
    quantₜ = @inbounds quantₜ[selected_actions, :]
    target_q =
        Flux.unsqueeze(rewards, 2) .+
        Flux.unsqueeze(γ^update_horizon .* (1 .- terminals), 2) .* quantₜ
    target_q = Flux.unsqueeze(target_distribution, 1)

    target_F = compute_forward(target_q, ξₜ, n_actions)
    # target_distribution = repeat(Flux.unsqueeze(target_distribution, 1),
    #                              n_actions, 1, 1)
    # target_distribution = reshape(target_distribution, size(target_distribution))

    gs = gradient(Flux.params(Z)) do
        ξ = Z(quantₜ_samples, states, n_actions)
        F = compute_forward(quantₜ_samples, ξ, n_actions)

        loss = sum(abs.(F[actions, :] - target_F[actions, :])) ./ length(F[actions, :])

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
