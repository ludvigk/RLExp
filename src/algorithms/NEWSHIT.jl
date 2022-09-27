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

using Plots


function mixture_gauss_cdf(x, loc, log_scales)
    x = Flux.unsqueeze(x, 1)
    scales = softplus.(log_scales)
    z_cdf = sigmoid.((x .- loc) ./ scales)
    # z_cdf = (1 .+ tanh.((x .- loc) ./ softplus.(log_scales))) ./ 2
    z_pdf = z_cdf .* (1 .- z_cdf) ./ scales
    return dropdims(sum(z_cdf, dims=1), dims=1) ./ size(z_cdf, 1), dropdims(sum(z_pdf, dims=1), dims=1) ./ size(z_cdf, 1)
end

function compute_forward(x, params, na)
    loc, log_scale = chunk(params, 2, dims=1)
    # log_scale = zero(loc) .- 2

    loc = reshape(loc, :, na, size(loc, 2))
    log_scale = reshape(log_scale, :, na, size(log_scale, 2))

    mixture_gauss_cdf(x, loc, log_scale)
end

function mixture_inv_cdf(x, means, log_scales; max_it=100, eps=1.0f-10)
    z = CUDA.zero(x)  #TODO: Speed
    max_scales = sum(softplus.(log_scales), dims=1)
    t = CUDA.ones(eltype(x), size(x))  #TODO: Speed
    lb = dropdims(minimum(means .- 20 .* max_scales, dims=1), dims=1)
    lb = lb .* t
    ub = dropdims(maximum(means .+ 20 .* max_scales, dims=1), dims=1)
    ub = ub .* t

    for i = 1:max_it
        old_z = z
        y = mixture_gauss_cdf(z, means, log_scales)
        gt = convert(typeof(y), y .> x)
        lt = 1 .- gt
        # z = (ub + lb) / 2
        z = gt .* (old_z .+ lb) ./ 2 .+ lt .* (old_z .+ ub) ./ 2
        lb = gt .* lb .+ lt .* old_z
        ub = gt .* old_z .+ lt .* ub
        if maximum(abs.(z .- old_z)) < eps
            # @show y
            # @show old_z
            # @show i
            # @show maximum(ub)
            # @show minimum(lb)
            # @show maximum(log_scales)
            # @show minimum(log_scales)
            # @show maximum(means)
            # @show minimum(means)
            # p = scatter(x[1,1,:] |> cpu, z[1,1,:] |> cpu, xlims=(0,1))
            # ord = sortperm(means[:,1,1])
            # p = scatter(1:size(means, 1), (means[ord,1,1]))
            # p = scatter!(1:size(means, 1), (log_scales[ord,1,1]))
            # display(p)
            # @show i
            # p1 = scatter(y[1,1,:])
            # p2 = scatter(z[1,1,:])
            # display(plot(p1, p2))
            # @show maximum(abs.(z .- old_z))
            # @show x
            @assert !any(isnan.(z))
            break
        end
    end
    return z
end

function compute_backward(x, params, na; eps=1.0f-5)
    loc, log_scale = chunk(params, 2, dims=1)
    # log_scale = zero(loc) .- 2

    loc = reshape(loc, :, na, size(loc, 2))
    log_scale = reshape(log_scale, :, na, size(log_scale, 2))
    clamp!(x, eps, 1 - eps)
    mixture_inv_cdf(x, loc, log_scale)
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
    quant_samples = reshape(collect(0.05:0.01:0.95), 1, 1, :) # try other methods
    # quant_samples = rand(1, 1, L.n_samples_act) # try other methods
    quant_samples = quant_samples |> gpu  #TODO: Speed
    q = compute_backward(quant_samples, ξ, L.n_actions)

    q = dropdims(mean(q, dims=3), dims=3)
end

function (learner::NEWSHITLearner)(env::AbstractEnv)
    if state(env) isa Int
        s = send_to_device(device(learner.approximator), [state(env)])
    else
        s = send_to_device(device(learner.approximator), state(env))
    end
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
    quantₜ_samples = repeat(reshape(collect(0.05:0.01:0.95), 1, 1, :), 1, batch_size, 1) # try other methods
    support = repeat(reshape(collect(0:1:200), 1, 1, :), batch_size, 1)
    support = send_to_device(D, support)
    z_cdf, z_pdf = compute_forward(support, ξₜ, n_actions)
    quantₜ = z_pdf .* support

    if haskey(batch, :next_legal_actions_mask)
        l′ = send_to_device(D, batch[:next_legal_actions_mask])
        quantₜ .+= ifelse.(l′, 0.0f0, typemin(Float32))
    end
    mean_q = dropdims(sum(quantₜ, dims=3), dims=3)
    selected_actions = dropdims(argmax(mean_q; dims=1); dims=1)

    support = (support .- rewards) ./ γ
    target_cdf, _ = compute_forward(support, ξₜ, n_actions)
    target_cdf = target_cdf[selected_actions]
    gs = gradient(Flux.params(Z)) do
        ξ = Z(states)
        # quant = Zygote.@ignore compute_backward(quantₜ_samples, ξ, n_actions)

        cdf, pdf = compute_forward(support, ξ, n_actions)

        loss = sum(abs.(cdf[actions, :] .- target_cdf)) ./ length(quantₜ_samples)
        ignore_derivatives() do
            lp["loss"] = loss
            lp["F"] = mean(F)
            lp["maxF"] = maximum(F)
            lp["minF"] = minimum(F)
            # lp["extra_loss"] = extra_loss
            # lp["sldj"] = sum(sldj) / (batch_size * n_samples_target)
            # lp["QA"] = sum(selected_actions)[1] / length(selected_actions)
            # lp["mu"] = mean(μ)
            # lp["sigma"] = sum(σ[actions,:]) / length(σ[actions,:])
            lp["max_weight"] = maximum(maximum.(Flux.params(Z)))
            lp["min_weight"] = minimum(minimum.(Flux.params(Z)))
            # lp["max_pred"] = maximum(preds)
            # lp["min_pred"] = minimum(preds)
            for i = 1:n_actions
                lp["Q$i"] = sum(target_q[i, :, :]) / (91 * batch_size)
            end
        end

        return loss
    end
    optimise!(A, gs)
    # for p in Flux.params(Z)
    #     clamp!(p, -0.01f0, 0.01f0)
    # end
end
