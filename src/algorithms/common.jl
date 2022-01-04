using Arpack
using CUDA
using CUDA: randn!
using Flux
using Flux: glorot_uniform, unsqueeze, @nograd, convfilter, create_bias, expand, calc_padding
using KrylovKit
using LinearAlgebra
using NNlib: softplus
using Random
using Zygote

abstract type AbstractNoisy end

struct NoisyDense <: AbstractNoisy
    w_μ::AbstractMatrix
    w_ρ::AbstractMatrix
    b_μ::AbstractVector
    b_ρ::AbstractVector
    f::Any
    rng::AbstractRNG
end

function NoisyDense(
    in,
    out,
    f=identity;
    init_μ=glorot_uniform,
    init_σ=(dims...) -> fill(0.5f0 / Float32(sqrt(dims[end])), dims),
    # init_σ=(dims...) -> fill(0.017f0, dims),
    rng=Random.GLOBAL_RNG,
)
    return NoisyDense(
        init_μ(out, in),
        log.(exp.(init_σ(out, in)) .- 1),
        zeros(out),
        log.(exp.(init_σ(out)) .- 1),
        f,
        rng,
    )
end

Flux.@functor NoisyDense

# function (l::NoisyDense)(x, num_samples::Union{Int, Nothing}=nothing; rng::Union{AbstractRNG, Nothing}=nothing)
#     rng = rng === nothing ? l.rng : rng
#     x = ndims(x) == 2 ? unsqueeze(x, 3) : x
#     tmp_x = reshape(x, size(x, 1), :)
#     μ = l.w_μ * tmp_x .+ l.b_μ
#     σ² = softplus.(l.w_ρ) * tmp_x .^ 2 .+ softplus.(l.b_ρ)  ## SLOW
#     if num_samples === nothing
#         ϵ = Zygote.@ignore randn!(rng, similar(μ, size(μ, 1), 1))
#     else
#         # ϵ_1 = Zygote.@ignore randn!(rng, similar(μ, size(μ, 1), 1, 1))
#         # ϵ_2 = Zygote.@ignore randn!(rng, similar(μ, 1, 1, num_samples))
#         # ϵ = Zygote.@ignore ϵ_1 .* ϵ_2
#         ϵ = Zygote.@ignore randn!(rng, similar(μ, size(μ, 1), 1, num_samples))
#     end
#     μ = reshape(μ, size(μ, 1), size(x, 2), :)
#     σ² = reshape(σ², size(μ, 1), size(x, 2), :)
#     return y = l.f.(μ .+ ϵ .* sqrt.(σ²))
# end

function (l::NoisyDense)(x, num_samples::Union{Int, Nothing}=nothing; rng::Union{AbstractRNG, Nothing}=nothing)
    rng = rng === nothing ? l.rng : rng
    x = ndims(x) == 2 ? unsqueeze(x, 3) : x
    # μ = l.w_μ * tmp_x .+ l.b_μ
    wσ² = softplus.(l.w_ρ)
    bσ² = softplus.(l.b_ρ)
    if num_samples === nothing
        tmp_x = reshape(x, size(x, 1), :)
        wϵ = Zygote.@ignore randn!(rng, similar(x, size(wσ², 1), size(wσ², 2)))
        bϵ = Zygote.@ignore randn!(rng, similar(x, size(bσ², 1), 1))

        w = l.w_μ .+ wϵ .* wσ²
        b = l.b_μ .+ bϵ .* bσ²
        return y = l.f.(w * tmp_x .+ b)
    else
        tmp_x = x
        wϵ_1 = Zygote.@ignore randn!(rng, similar(x, size(wσ², 1), 1, 1))
        wϵ_2 = Zygote.@ignore randn!(rng, similar(x, 1, size(wσ², 2), 1))
        wϵ_3 = Zygote.@ignore randn!(rng, similar(x, 1, 1, num_samples))
        wϵ = Zygote.@ignore wϵ_1 .* wϵ_2 .* wϵ_2
        bϵ_1 = Zygote.@ignore randn!(rng, similar(x, size(bσ², 1), 1, 1))
        bϵ_2 = Zygote.@ignore randn!(rng, similar(x, 1, 1, num_samples))
        bϵ = Zygote.@ignore bϵ_1 .* bϵ_2

        w = l.w_μ .+ wϵ .* wσ²
        b = l.b_μ .+ bϵ .* bσ²
        println(size(x))
        return y = l.f.(batched_mul(w, x) .+ b)
    end
end

struct NoisyConv{N, M, F, A, V} <: AbstractNoisy
    f::F
    w_μ::A
    w_ρ::A
    b_μ::V
    b_ρ::V
    stride::NTuple{N, Int}
    pad::NTuple{M, Int}
    dilation::NTuple{N, Int}
    groups::Int
    rng::AbstractRNG
end

function NoisyConv(w_μ::AbstractArray{T,N},
                   w_ρ::AbstractArray{T,N},
                   b_μ::AbstractVector,
                   b_ρ::AbstractVector,
                   f = identity;
                   stride = 1,
                   pad = 0,
                   dilation = 1,
                   groups = 1,
                   rng = Random.GLOBAL_RNG
                   ) where {T, N}
    stride = expand(Val(N-2), stride)
    dilation = expand(Val(N-2), dilation)
    pad = calc_padding(Conv, pad, size(w_μ)[1:N-2], dilation, stride)
    return NoisyConv(f, w_μ, w_ρ, b_μ, b_ρ, stride, pad, dilation, groups, rng)
end

function NoisyConv(k::NTuple{N,Integer},
                   ch::Pair{<:Integer,<:Integer},
                   f = identity;
                   init_μ = glorot_uniform,
                   init_σ=(dims...) -> fill(0.0017f0, dims),
                   stride = 1,
                   pad = 0,
                   dilation = 1,
                   groups = 1,
                   rng = Random.GLOBAL_RNG,
                   ) where N
    w_μ = convfilter(k, (ch[1] ÷ groups => ch[2]); init = init_μ)
    b_μ = create_bias(w_μ, true, size(w_μ, ndims(w_μ)))
    w_ρ = convfilter(k, (ch[1] ÷ groups => ch[2]); init = init_σ)
    b_ρ = create_bias(w_μ, true, size(w_μ, ndims(w_μ)))
 
    NoisyConv(w_μ::AbstractArray,
              w_ρ::AbstractArray,
              b_μ::AbstractVector,
              b_ρ::AbstractVector,
              f;
              stride = stride,
              pad = pad,
              dilation = dilation,
              groups = groups,
              rng = rng
              )
end

Flux.@functor NoisyConv

function (c::NoisyConv)(x::AbstractArray, num_samples::Union{Int, Nothing}=nothing)
    # TODO: breaks gpu broadcast :(
    # ndims(x) == ndims(c.weight)-1 && return squeezebatch(c(reshape(x, size(x)..., 1)))
    b_μ = reshape(c.b_μ, ntuple(_ -> 1, length(c.stride))..., :, 1)
    cdims = DenseConvDims(x, c.w_μ; stride = c.stride, padding = c.pad, dilation = c.dilation, groups = c.groups)
    μ = conv(x, c.w_μ, cdims) .+ b_μ
    if num_samples === nothing
        ϵ = Zygote.@ignore randn!(c.rng, similar(μ, size(μ)[1:3]..., 1))
    else
        ϵ = Zygote.@ignore randn!(c.rng, similar(μ, size(μ)[1:3]..., 1, 1))
    end
    b_ρ = reshape(c.b_μ, ntuple(_ -> 1, length(c.stride))..., :, 1)
    σ² = conv(x .^ 2, softplus.(c.w_ρ), cdims) .+ softplus.(b_ρ)
    r = c.f.(μ .+ ϵ .* sqrt.(σ²))
    reshape(r, size(r)[1:3]..., :)
  end

# custom split layer
struct Split{T}
  paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

function (m::Split)(x::AbstractArray)
    if Flux.istraining()
        return map(f -> f(x), m.paths)
    end
    return m.paths[1](x)
end

function (m::Split)(x::AbstractArray, n)
    if Flux.istraining()
        return map(f -> f(x, n), m.paths)  ## IS THIS SLOW?
    end
    return m.paths[1](x, n)
end


# ----- Spectral stein gradient estimator ----- #

function rbf_kernel(x1::AbstractArray, x2::AbstractArray, lengthscale)
    rb = -sum((x1 .- x2) .^ 2 ./ (2 .* lengthscale .^ 2); dims=3)  ## SLOW
    rb = dropdims(rb; dims=3)
    return exp.(rb)
end

function gram(x1, x2, lengthscale)
    x1 = Flux.unsqueeze(x1, 2)
    x2 = Flux.unsqueeze(x2, 1)
    lengthscale = reshape(lengthscale, 1, 1, :)
    return Kxx = rbf_kernel(x1, x2, lengthscale)
end

function grad_gram(x1::AbstractArray, x2::AbstractArray, lengthscale)
    x1 = Flux.unsqueeze(x1, 2)
    x2 = Flux.unsqueeze(x2, 1)
    lengthscale = reshape(lengthscale, 1, 1, :)
    Kxx = rbf_kernel(x1, x2, lengthscale)  ## SLOW
    diff = (x1 .- x2) ./ lengthscale .^ 2
    dKxx_dx1 = Flux.unsqueeze(Kxx, 3) .* (-diff)
    dKxx_dx2 = Flux.unsqueeze(Kxx, 3) .* diff
    return Kxx, dKxx_dx1, dKxx_dx2
end

Zygote.@nograd function heuristic_lengthscale(x::AbstractArray, xm::AbstractArray)
    x_dim = size(x, 2)
    n_samples = size(x, 1)
    n_basis = size(xm, 1)
    x1 = Flux.unsqueeze(x, 2)
    x2 = Flux.unsqueeze(xm, 1)
    pdist_mat = abs.(x1 .- x2)
    pdist_mat = permutedims(pdist_mat, (3, 1, 2))  ## SLOW
    kernel_width = sort(reshape(pdist_mat, :, x_dim, n_samples * n_basis); dims=3)[
        :, :, div(end, 2) ## SLOW
    ]
    kernel_width = flatten(kernel_width)
    kernel_width = kernel_width .* sqrt(Float32(x_dim))
    return kernel_width = kernel_width .+ (kernel_width .< 1e-6)
end

function nystrom_ext(x, eval_points, eigen_vecs, eigen_vals, lengthscale)
    M = size(x, 1)
    Kxxm = gram(eval_points, x, lengthscale)
    phi_x = sqrt(M) .* (Kxxm * eigen_vecs)
    return phi_x = phi_x ./ Flux.unsqueeze(eigen_vals, 1)
end

abstract type BaseScoreEstimator end

struct SpectralSteinEstimator <: BaseScoreEstimator
    η::Any
    nev::Any
    n_eigen_threshold::Any
end

function compute_gradients(
    sse, x::AbstractArray, xm::AbstractArray, lengthscale::Union{AbstractArray,Float32}
)
    M = size(xm, 1)
    Kxx, dKxx, _ = grad_gram(xm, xm, lengthscale)

    if typeof(x) <: CuArray
        Kxx = cpu(Kxx)
        if !isnothing(sse.η)
            Kxx = Kxx + sse.η * I
        end
        eigen_vals, eigen_vecs = eigsolve(Kxx, sse.nev, :LR, issymmetric=true)
		eigen_vecs = hcat(eigen_vecs...)
        eigen_vals, eigen_vecs = gpu(eigen_vals), gpu(eigen_vecs)
    else
        if !isnothing(sse.η)
            Kxx = Kxx + sse.η * I
        end
        eigen_vals, eigen_vecs = eigsolve(Kxx, sse.nev, :LR, issymmetric=true)
		eigen_vecs = hcat(eigen_vecs...)
    end

    phi_x = nystrom_ext(xm, x, eigen_vecs, eigen_vals, lengthscale)
    dKxx_dx_avg = dropdims(mean(dKxx; dims=1); dims=1)
    beta = -sqrt(M) .* (eigen_vecs' * dKxx_dx_avg)
    beta = beta ./ Flux.unsqueeze(eigen_vals, 2)
    return phi_x * beta
end

function compute_gradients(sse, x::AbstractArray, xm::AbstractArray)
    _xm = cat(x, xm; dims=1)
    lengthscale = heuristic_lengthscale(_xm, _xm)
    return compute_gradients(sse, x, xm, lengthscale)
end

function compute_gradients(sse, xm::AbstractArray)
    x = xm
    lengthscale = heuristic_lengthscale(xm, xm)
    return compute_gradients(sse, x, xm, lengthscale)
end

function entropy_surrogate(sse, samples)
    dlog_q = Zygote.@ignore -compute_gradients(sse, samples)
    return surrogate = sum(dlog_q .* samples) / size(samples, 1)
end

function cross_entropy_surrogate(sse, q_data, p_data)
    cross_entropy_gradients = Zygote.@ignore compute_gradients(sse, q_data, p_data)
    cross_entropy_sur = sum(cross_entropy_gradients .* q_data) / size(q_data, 1)
    return -cross_entropy_sur
end
