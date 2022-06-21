using Flux
using Zygote
# using DataFrames
# using SyntheticDatasets
# using Plots
# using StatsPlots
# using ProgressMeter

import Distributions: logpdf

export UvPlanar
export CouplingLayer, ConditionalCouplingLayer
export ConditionalRealNVP, RealNVP
export PlanarLayer, PlanarFlow

struct UvPlanar
    u
    w
    b
end

Flux.@functor UvPlanar
Flux.trainable(f::UvPlanar) = (f.u, f.w)

tanh_prime(x) = 1 - tanh(x)^2

function (f::UvPlanar)(x)
    return x .+ f.u .* tanh.(f.w .* x .+ b)
end

function logpdf(f, x)
    return log.(abs.(1 .+ f.u .* tanh_prime.(f.w .* x .+ b) .* f.w) .+ 1.0f-6)
end

struct PlanarLayer
    u
    w
    b
    f
    f_prime
end

Flux.@functor PlanarLayer
Flux.trainable(f::PlanarLayer) = (f.u, f.w, fb)

function PlanarLayer(h_size)
    u = glorot_uniform(1, 1)
    w = glorot_uniform(1, h_size)
    b = glorot_uniform(1, 1)
    f = tanh
    f_prime = tanh_prime
    return PlanarLayer(u, w, b, f, f_prime)
end

function (l::PlanarLayer)(x, h; reverse=true)
    wh = l.w * h
    fwb = wh .* x #.+ l.b
    x = x .+ l.u .* l.f.(fwb)
    sldj = log.(abs.(1 .+ l.f_prime.(fwb) .* wh .* l.u))
    return x, sldj
end

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

function ConditionalCouplingLayer(in::Int, cond::Int, hidden::Int, mask)
    net = Chain(
        Dense(in + cond, hidden, leakyrelu),
        Dense(hidden, hidden, leakyrelu),
        Split(
            Dense(hidden, in, tanh),
            Dense(hidden, in),
        ),
    )
    return ConditionalCouplingLayer(mask, net, RescaleLayer(in))
end

function (c::ConditionalCouplingLayer)(x, h::AbstractMatrix{T}, sldj=nothing; action=nothing, reverse=true) where {T}
    x_ = x .* c.mask
    x_h_ = vcat(x_, h)
    s, t = c.net(x_h_)

    s = c.rescale(s)
    s = s .* (1 .- c.mask)
    t = t .* (1 .- c.mask)

    if reverse
        inv_exp_s = exp.(-s)
        x = x .* inv_exp_s - t
        return x
    else
        x = (x .+ t) .* exp.(s)
        if isnothing(action)
            sldj += dropdims(sum(s, dims=1), dims=1)
        else
            sldj += s[action, :]
        end
        return x, sldj
    end
end

function (c::ConditionalCouplingLayer)(x, h::AbstractArray{T,3}, sldj=nothing; action=nothing, reverse=true) where {T}
    x_ = x .* c.mask
    x_h_ = cat(repeat(x_, 1, 1, size(h, 3)), h, dims=1)
    x_h_ = reshape(x_h_, size(x_h_, 1), :)
    s, t = c.net(x_h_)

    s = c.rescale(s)
    s = s .* (1 .- c.mask)
    t = t .* (1 .- c.mask)

    if reverse
        inv_exp_s = exp.(-s)
        x = x .* inv_exp_s - t
        return reshape(x, size(x, 1), :, size(h, 3))
    else
        x = (x .+ t) .* exp.(s)
        if isnothing(action)
            sldj += dropdims(sum(s, dims=1), dims=1)
        else
            sldj += s[action, :]
        end
        return reshape(x, size(x, 1), :, size(h, 3)), reshape(sldj, :, size(h, 3))
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
        sldj = Zygote.@ignore similar(x, size(x, 2))
        Zygote.@ignore fill!(sldj, zero(eltype(x)))
        for layer in Base.reverse(r.coupling_layers)
            x, sldj = layer(x, sldj; reverse)
        end
        return x, sldj
    end
end

struct ConditionalRealNVP
    coupling_layers
end

Flux.@functor ConditionalRealNVP

function (r::ConditionalRealNVP)(x, h::AbstractMatrix; action=nothing, reverse=false)
    if reverse
        for layer in r.coupling_layers
            x = layer(x, h; reverse)
        end
        return x
    else
        sldj = Zygote.@ignore similar(x, size(x, 2))
        Zygote.@ignore fill!(sldj, zero(eltype(x)))
        for layer in Base.reverse(r.coupling_layers)
            x, sldj = layer(x, h, sldj; action, reverse)
        end
        return x, sldj
    end
end

function (r::ConditionalRealNVP)(x, h::AbstractArray{T,3}; action=nothing, reverse=false) where {T}
    if reverse
        for layer in r.coupling_layers
            x = layer(x, h; reverse)
        end
        return x
    else
        sldj = Zygote.@ignore similar(x, size(x, 2), size(h, 3))
        Zygote.@ignore fill!(sldj, zero(eltype(x)))
        for layer in Base.reverse(r.coupling_layers)
            x, sldj = layer(x, h, sldj; action, reverse)
        end
        return x, sldj
    end
end

# function main()

#     moons = SyntheticDatasets.make_moons(n_samples=100; noise=0.05)
#     data = Matrix(moons[:, 1:2])'

#     DTYPE = Float32
#     # dist = MvNormal(zeros(DTYPE, 2), ones(DTYPE, 2))

#     loss(td, data) = -mean(logpdf_forward(td, data))

#     coupling_layers = [
#         ConditionalCouplingLayer(2, 1, 100, [0, 1]),
#         ConditionalCouplingLayer(2, 1, 100, [1, 0]),
#         ConditionalCouplingLayer(2, 1, 100, [0, 1]),
#         ConditionalCouplingLayer(2, 1, 100, [1, 0]),
#     ]
#     rnvp = ConditionalRealNVP(coupling_layers)

#     function loss(model, data)
#         action = [
#             CartesianIndex(rand(1:2), i) for i = 1:size(data, 2)
#         ]
#         x, sldj = model(data, rand(1, size(data, 2)); action)
#         nll = sum(x[action] .^ 2) / 2 - sum(sldj)
#         return nll / size(data, 2)
#     end

#     opt = ADAM(0.0003)
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
#     samples = rnvp(randn(2, 100), rand(1, 100), reverse=true)
#     @df moons scatter(:feature_1, :feature_2)
#     scatter!(samples[1, :], samples[2, :])
# end