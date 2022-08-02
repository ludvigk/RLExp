using Flux
using Zygote
using MLUtils
using Setfield
using NNlib
using DataStructures: counter

# using DataFrames
# using SyntheticDatasets
# using Plots
# using StatsPlots
# using ProgressMeter

export CouplingLayer, ConditionalCouplingLayer
export ConditionalRealNVP, RealNVP
export PlanarLayer, PlanarFlow
export FlowNorm
export Flow
export LuddeFlow

struct LuddeFlow
    net
end

Flux.@functor LuddeFlow

function LuddeFlow(in::Int, h_size::Int, h_dims::Int, init=Flux.glorot_uniform())
    net = Chain(
        Dense(h_size, h_dims, leakyrelu, init=init),
        Dense(h_dims, h_dims, leakyrelu, init=init),
        Dense(h_dims, 3 * in, init=init),
    )
    return LuddeFlow(net)
end

function (l::LuddeFlow)(x, h)
    ϵ = 1f-6
    b, c, d = MLUtils.chunk(l.net(h), 3, dims=1)
    w = 1f0
    xd = x .- d
    sb = softplus.(b)
    inner = exp.(abs.(xd) .+ sb) .- 1
    out = sign.(xd) .* (log.(max.(inner, ϵ)) .- b) ./ w .- c
    eax = exp.(abs.(xd))
    eb = exp.(b) .+ 1
    d_upper = eb .* eax
    d_lower = w .* (d_upper .- 1) .+ ϵ
    return out, log.(max.(abs.(d_upper ./ d_lower), ϵ))
end

function inverse(l::LuddeFlow, x, h)
    b, c, d = MLUtils.chunk(l.net(h), 3, dims=1)
    w = 1f0
    inner = abs.(w .* (x .+ c)) .+ b
    out = sign.(x .+ c) .* (softplus.(inner) .- softplus.(b)) .+ d
    # TODO: Calculate inverse derivative
    d_inner = exp.(abs.(w .* (c .+ x)) .+ b)
    d = w .* d_inner ./ (d_inner .+ 1)
    return out, log.(abs.(d))
end

tanh_prime(x) = 1 - tanh_fast(x) ^ 2

struct PlanarLayer
    net
    neg_slope
end

Flux.@functor PlanarLayer

function PlanarLayer(in::Int, h_size::Int, h_dims::Int, init=Flux.glorot_normal())
    net = Chain(
        Dense(h_size, h_dims, relu, init=init),
        Dense(h_dims, h_dims, relu, init=init),
        Dense(h_dims, 3 * in, init=(args...) -> init(args...) ./ 100),
    )
    return PlanarLayer(net, 0.2f0)
end

# function (l::PlanarLayer)(x, h)
#     m(x) = -1 + log(1 + exp(x))
#     w, u, b = MLUtils.chunk(l.net(h), 3, dims=1)
#     u, w = tanh_fast.(u), tanh_fast.(w)
#     # u = u .+ (m.(w .* u) .- w .* u) .* (w .+ 1f-8) ./ (abs2.(w) .+ 1f-8)
#     f(x) = x .+ u .* tanh_fast.(w .* x .+ b)
#     ψ(x) = tanh_prime.(w .* x .+ b) .* w
#     f′(x) = 1 .+ u .* ψ(x)
#     x = f(x)
#     return x, log.(abs.(f′(x)))
# end

# function inverse(l::PlanarLayer, x, h)
#     m(x) = -1 + log(1 + exp(x))
#     w, u, b = MLUtils.chunk(l.net(h), 3, dims=1)
#     u, w = tanh_fast.(u), tanh_fast.(w)
#     # if any(w .* u .<= -1)
#     # u = u .+ (m.(w .* u) .- w .* u) .* (w .+ 1f-8) ./ (abs2.(w) .+ 1f-8)
#     # end
#     f(x) = x .+ u .* tanh_fast.(w .* x .+ b)
#     ψ(x) = tanh_prime.(w .* x .+ b) .* w
#     f′(x) = 1 .+ u .* ψ(x)

#     x_inv_new = similar(x)
#     x_inv_new .= x .- sign.(w) .* u
#     x_diff = Inf
#     wtx = w .* x
#     wtu = w .* u
#     for _ = 1:10
#         x_inv = x_inv_new
#         fx = x_inv .+ wtu .* tanh_fast.(x_inv .+ b) .- wtx
#         fx′ = 1 .+ wtu .* tanh_prime.(x_inv .+ b)
#         x_inv_new = x_inv .- fx ./ (fx′ .+ 1f-6)
#         x_diff = maximum(abs, x_inv .- x_inv_new)
#         if x_diff < 1f-6
#             break
#         end
#     end
#     return x .- u .* tanh_fast.(x_inv_new .+ b), 0f0
# end

# struct ScaleAndShiftLayer
#     net
# end

# Flux.@functor ScaleAndShiftLayer

# function (l::ScaleAndShiftLayer)(x, h)
#     μ, ρ = 
# end

struct Flow
    layers
end

Flux.@functor Flow

function (f::Flow)(x, h)
    sldj = Zygote.@ignore similar(x)
    Zygote.@ignore fill!(sldj, 0)
    for layer in f.layers
        x, sldj_ = layer(x, h)
        sldj = sldj .+ sldj_
    end
    return x, sldj
end

function inverse(f::Flow, x, h)
    sldj = Zygote.@ignore similar(x)
    Zygote.@ignore fill!(sldj, 0)
    for layer in Base.reverse(f.layers)
        x, sldj_ = inverse(layer, x, h)
        sldj = sldj .+ sldj_
    end
    return x, sldj
end

# function (r::Flow)(x, h::AbstractArray{T,3}, a; reverse=false) where {T}
#     if reverse
#         for layer in r.coupling_layers
#             x = layer(x, h; action=a, reverse)
#         end
#         return x
#     else
#         sldj = Zygote.@ignore similar(x, size(x, 2))
#         Zygote.@ignore fill!(sldj, zero(eltype(x)))
#         for layer in Base.reverse(r.coupling_layers)
#             x, sldj = layer(x, h, sldj; action=a, reverse)
#         end
#         return x, sldj
#     end
# end

# function PlanarLayer(in::Int, h_size::Int, h_dims::Int)
#     net = Chain(
#         Dense(h_size, h_dims, leakyrelu),
#         Dense(h_dims, h_dims, leakyrelu),
#         Dense(h_dims, 3 * in),
#     )
#     f = tanh
#     f_prime = tanh_prime
#     return PlanarLayer(net, f, f_prime)
# end

# function (l::PlanarLayer)(x, h, sldj=nothing; action=nothing, reverse=true)
#     m(x) = -1 + log(1 + exp(x))
#     w, u, b = MLUtils.chunk(l.net(h), 3, dims=1)

#     # if any(w .* u .<= -1)
#     u = u .+ (m.(w .* u) .- w .* u) .* (w .+ 1f-5) ./ (abs2.(w) .+ 1f-5)
#     # end
#     f(x) = x .+ u .* tanh.(w .* x .+ b)
#     ψ(x) = tanh_prime.(w .* x .+ b) .* w
#     f′(x) = 1 .+ u .* ψ(x)
#     if reverse
#         x_inv_new = similar(x)
#         x_inv_new .= x .- sign.(w) .* u
#         x_diff = Inf
#         wtx = w .* x
#         wtu = w .* u
#         for _ = 1:30
#             x_inv = x_inv_new
#             fx = x_inv .+ wtu .* tanh.(x_inv .+ b) .- wtx
#             fx′ = 1 .+ wtu .* tanh_prime.(x_inv .+ b)
#             x_inv_new = x_inv .- fx ./ fx′
#             x_diff = maximum(abs, x_inv .- x_inv_new)
#             if x_diff < 1f-4
#                 break
#             end
#         end
#         return x .- u .* tanh.(x_inv_new .+ b)
#     else
#         x = f(x)
#         if isnothing(action)
#             sldj += sum(log.(abs.(f′(x))))
#         else
#             sldj += sum(log.(abs.(f′(x)))[action, :])
#         end
#         return x, sldj
#     end
# end

struct PlanarFlow
    layers
end

Flux.@functor PlanarFlow

function (r::PlanarFlow)(x, h; reverse=false)
    sldj = Zygote.@ignore similar(x, size(x, 2))
    Zygote.@ignore fill!(sldj, zero(eltype(x)))
    for layer in r.layers
        x, sldj_ = layer(x, h; reverse)
        sldj += dropdims(sum(sldj_, dims=1), dims=1)
    end
    return x, sldj
end

struct RescaleLayer
    v
    g
end

Flux.@functor RescaleLayer
Flux.trainable(r::RescaleLayer) = (r.v, r.g)

function RescaleLayer(in)
    v = glorot_uniform(in)
    g = ones(1)
    return RescaleLayer(v, g)
end

(r::RescaleLayer)(x) = (r.v ./ sum(r.v) .* r.g) .* x


## Coupling Layers
struct CouplingLayer
    mask
    net
    rescale
end

Flux.@functor CouplingLayer
Flux.trainable(c::CouplingLayer) = (c.net, c.rescale)

function CouplingLayer(in::Int, hidden::Int, mask)
    net = Chain(
        Dense(in, hidden, leakyrelu),
        Dense(hidden, hidden, leakyrelu),
        Split(
            Dense(hidden, in, tanh),
            Dense(hidden, in),
        ),
    )
    return CouplingLayer(mask, net, RescaleLayer(in))
end

function (c::CouplingLayer)(x, sldj=nothing; reverse=true)
    x_ = x .* c.mask
    s, t = c.net(x_)

    s = c.rescale(s)
    s = s .* (1 .- c.mask)
    t = t .* (1 .- c.mask)

    if reverse
        inv_exp_s = exp.(-s)
        x = x .* inv_exp_s - t
        return x
    else
        x = (x .+ t) .* exp.(s)
        sldj += dropdims(sum(s, dims=1), dims=1)
        return x, sldj
    end
end

struct ConditionalCouplingLayer
    mask
    net
    rescale
end

Flux.@functor ConditionalCouplingLayer
Flux.trainable(c::ConditionalCouplingLayer) = (c.net, c.rescale)

ConditionalCouplingLayer(in::Int, cond::Int, hidden::Int) = ConditionalCouplingLayer(in, cond, hidden, nothing)

function ConditionalCouplingLayer(in::Int, cond::Int, hidden::Int, mask)
    net = Chain(
        Dense(cond, hidden, leakyrelu),
        Dense(hidden, hidden, leakyrelu),
        Dense(hidden, 2in),
    )
    return ConditionalCouplingLayer(mask, net, RescaleLayer(in))
end

function (c::ConditionalCouplingLayer)(x, h::AbstractMatrix{T}, sldj=nothing; action=nothing, reverse=true) where {T}
    x_ = x
    x_h_ = h
    s, t = MLUtils.chunk(c.net(x_h_), 2, dims=1)
    # s = tanh.(s)

    # s = c.rescale(s)
    # s = s .* (1 .- c.mask)
    # t = t .* (1 .- c.mask)

    if reverse
        # inv_exp_s = exp.(-s)
        # x = x .* inv_exp_s .- t
        x = x .- t
        return x
    else
        x = (x .+ t)
        # x = (x .+ t) .* exp.(s)
        # if isnothing(action)
        #     sldj = sldj + sum(s)
        # else
        #     sldj = sldj + sum(s[action, :])
        # end
        return x, sldj
    end
end

function (c::ConditionalCouplingLayer)(x::AbstractArray{T,3}, h::AbstractMatrix{T}, sldj=nothing; action=nothing, reverse=true) where {T}
    h_broadcast = Zygote.@ignore similar(h, size(h)..., size(x, 3))
    Zygote.@ignore fill!(h_broadcast, 1)
    h_b = h .* h_broadcast
    return c(x, h_b, sldj; action, reverse)
end

function (c::ConditionalCouplingLayer)(x::AbstractMatrix, h::AbstractArray{T,3}, sldj=nothing; action=nothing, reverse=true) where {T}
    x_broadcast = Zygote.@ignore similar(x, size(x)..., size(h, 3))
    Zygote.@ignore fill!(x_broadcast, 1)
    x_b = x .* x_broadcast
    return c(x_b, h, sldj; action, reverse)
end

function (c::ConditionalCouplingLayer)(x::AbstractArray{T,3}, h::AbstractArray{T,3}, sldj=nothing; action=nothing, reverse=true) where {T}
    x_ = x
    x_h_ = h
    s, t = MLUtils.chunk(c.net(x_h_), 2, dims=1)
    # s = tanh.(s)

    # s = c.rescale(s)
    # s = s .* (1 .- c.mask)
    # t = t .* (1 .- c.mask)

    if reverse
        # inv_exp_s = exp.(-s)
        x = x .- t
        # x = x .* inv_exp_s - t
        return x
    else
        x = (x .+ t)
        # x = (x .+ t) .* exp.(s)
        # if isnothing(action)
        #     sldj = sldj + sum(s)
        # else
        #     sldj = sldj + sum(s[action, :])
        # end
        return x, sldj
    end
end


## Real-NVP

struct RealNVP
    coupling_layers
end

Flux.@functor RealNVP

function (r::RealNVP)(x; reverse=false)
    if reverse
        for layer in r.coupling_layers
            x = layer(x; reverse)
        end
        return x
    else
        sldj = 0f0
        for layer in Base.reverse(r.coupling_layers)
            x, sldj = layer(x; reverse)
        end
        return x, sldj
    end
end

struct ConditionalRealNVP
    coupling_layers
end

Flux.@functor ConditionalRealNVP

function (r::ConditionalRealNVP)(x, h::AbstractMatrix, a; reverse=false)
    if reverse
        for layer in r.coupling_layers
            x = layer(x, h; action=a, reverse=reverse)
        end
        return x
    else
        sldj = 0f0
        for layer in Base.reverse(r.coupling_layers)
            x, sldj = layer(x, h, sldj; action=a, reverse=reverse)
        end
        return x, sldj
    end
end

function (r::ConditionalRealNVP)(x, h::AbstractArray{T,3}, a; reverse=false) where {T}
    if reverse
        for layer in r.coupling_layers
            x = layer(x, h; action=a, reverse)
        end
        return x
    else
        sldj = Zygote.@ignore similar(x, size(x, 2))
        Zygote.@ignore fill!(sldj, zero(eltype(x)))
        for layer in Base.reverse(r.coupling_layers)
            x, sldj = layer(x, h, sldj; action=a, reverse)
            println(sldj)
        end
        return x, sldj
    end
end

struct FlowNorm
    beta
    gamma
    train_m
    train_s
    decay
    ϵ
end

Flux.@functor FlowNorm
Flux.trainable(f::FlowNorm) = (f.beta, f.gamma)

function FlowNorm(n, decay=0.95f0)
    return FlowNorm(glorot_uniform(n),
                    glorot_uniform(n),
                    zeros(n),
                    ones(n),
                    Float32(decay),
                    1f-6)
end

function (a::FlowNorm)(x, h, sldj=nothing; action=nothing, reverse=false)
    if reverse
        z = (x .- a.beta) .* exp.(-a.gamma) .* (a.train_s .+ a.ϵ) .+ a.train_m
        return z
    else
        m = Zygote.@ignore sum(x, dims=2) ./ size(x, 2)
        s = Zygote.@ignore std(x, dims=2)
        Zygote.ignore() do 
            @set a.train_m = a.train_m - a.decay .* (a.train_m .- m)
            @set a.train_s = a.train_s - a.decay .* (a.train_s .- s)
        end
        z = (x .- m) ./ (s .+ a.ϵ) .* exp.(a.gamma) .+ a.beta
        if isnothing(action)
            sldj = sldj + sum(a.gamma .- log.(s .+ a.ϵ))
        else
            bs = length(action)
            s = repeat(s, 1, bs)[action]
            gamma = repeat(s, 1, bs)[action]
            sldj = sldj + sum(gamma .- log.(s .+ a.ϵ))
        end
        return z, sldj
    end
end

# function main()

#     moons = SyntheticDatasets.make_blobs(n_samples = 200, 
#     n_features = 2,
#     centers = [-1 1; -0.5 0.5], 
#     cluster_std = 0.1,
#     center_box = (-2.0, 2.0), 
#     shuffle = true,
#     random_state = nothing)
#     data = Matrix(moons[:, 1:2])'

#     DTYPE = Float32
#     # dist = MvNormal(zeros(DTYPE, 2), ones(DTYPE, 2))

#     # loss(td, data) = -mean(logpdf_forward(td, data))

#     layers = [
#         PlanarLayer(2, 1, 32),
#         PlanarLayer(2, 1, 32),
#         PlanarLayer(2, 1, 32),
#         PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#         # PlanarLayer(2, 1, 32),
#     ]
#     rnvp = Flow(layers)

#     function loss(model, data)
#         action = [
#             CartesianIndex(rand(1:2), i) for i = 1:size(data, 2)
#         ]
#         x, sldj = model(data, ones(1, size(data, 2)))
#         nll = sum(x[action] .^ 2) / 2 - sum(sldj[action])
#         return nll / size(data, 2)
#     end

#     opt = ADAM(0.0001)
#     function nf_train(td, data, opt, ps, epochs; batchsize=100)
#         p = Progress(epochs, 1)
#         for i ∈ 1:epochs
#             d = Flux.Data.DataLoader(data, batchsize=batchsize)
#             for minibatch = d
#                 gs = gradient(() -> loss(td, minibatch), ps)

#                 # println(gs.grads[b.ts[1].s[1].weight])
#                 # println(gs.grads[b.ts[1].t[1].weight])
#                 Flux.update!(opt, ps, gs)
#             end
#             ProgressMeter.next!(p; showvalues=[(:nll, loss(td, data))])
#         end
#     end
#     nf_train(rnvp, data, opt, Flux.params(rnvp), 5000)
#     # contour(-2:0.1:3, -2:0.1:2, (x, y) -> pdf(td, [x, y]), fill=true)
#     # samples = rand(td, 100)
#     samples = inverse(rnvp, randn(2, 300), ones(1, 300))[1]
#     @df moons scatter(:feature_1, :feature_2)
#     scatter!(samples[1, :], samples[2, :])
# end