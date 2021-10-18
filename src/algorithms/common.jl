using CUDA
using CUDA: randn!
using Flux
using Flux: glorot_uniform, unsqueeze, @nograd
using NNlib: softplus
using Zygote

struct NoisyDense
    w_μ::AbstractMatrix
    w_ρ::AbstractMatrix
    b_μ::AbstractVector
    b_ρ::AbstractVector
    f::Any
end

function NoisyDense(
    in, out, f=identity; init_μ=glorot_uniform(), init_σ=(dims...) -> fill(0.0017f0, dims)
)
    return NoisyDense(
        init_μ(out, in),
        log.(exp.(init_σ(out, in)) .- 1),
        init_μ(out),
        log.(exp.(init_σ(out)) .- 1),
        f,
    )
end

Flux.@functor NoisyDense

function (l::NoisyDense)(x)
    x = ndims(x) == 2 ? unsqueeze(x, 3) : x
    tmp_x = reshape(x, size(x, 1), :)
    μ = l.w_μ * tmp_x .+ l.b_μ
    σ² = softplus.(l.w_ρ) * tmp_x .^ 2 .+ softplus.(l.b_ρ)
    if Flux.istraining()
        ϵ = Zygote.@ignore randn!(similar(μ, size(μ, 1), 1, 100))
    else
        ϵ = Zygote.@ignore randn!(similar(μ, size(μ, 1), 1))
    end
    μ = reshape(μ, size(μ, 1), size(x, 2), :)
    σ² = reshape(σ², size(μ, 1), size(x, 2), :)
    return y = l.f.(μ .+ ϵ .* sqrt.(σ²))
end

function rbf_kernel(x1::AbstractArray, x2::AbstractArray, lengthscale)
    rb = -sum((x1 .- x2) .^ 2 ./ (2 .* lengthscale .^ 2); dims=3)
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
    Kxx = rbf_kernel(x1, x2, lengthscale)
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
    pdist_mat = permutedims(pdist_mat, (3, 1, 2))
    kernel_width = sort(reshape(pdist_mat, :, x_dim, n_samples * n_basis); dims=3)[
        :, :, div(end, 2)
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
    num_eigs::Any
    n_eigen_threshold::Any
end

function compute_gradients(
    sse, x::AbstractArray, xm::AbstractArray, lengthscale::Union{AbstractArray,Float32}
)
    M = size(xm, 1)
    Kxx, dKxx, _ = grad_gram(xm, xm, lengthscale)

    if !isnothing(sse.η)
        Kxx = Kxx .+ Diagonal(CUDA.fill(sse.η, size(Kxx, 1)))
    end

    eigen_vals, eigen_vecs = CUDA.CUSOLVER.syevd!('V', 'U', Kxx)
    num_eigs = sse.num_eigs
    if isnothing(num_eigs) && !isnothing(sse.n_eigen_threshold)
        eigen_arr = mean(reshape(eigen_vals, :, M); dims=1)
        eigen_arr = eigen_arr ./ sum(eigen_arr)
        eigen_cum = eigen_arr ./ sum(eigen_arr; dims=2)
        num_eigs = count(eigen_cum .< sse.n_eigen_threshold)
    end
    if !isnothing(num_eigs)
        eigen_vals = eigen_vals[(end - num_eigs + 1):end]
        eigen_vecs = eigen_vecs[:, (end - num_eigs + 1):end]
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
    dlog_q = nothing
    Zygote.ignore() do
        return dlog_q = -compute_gradients(sse, samples)
    end
    return surrogate = mean(sum(dlog_q .* samples; dims=2))
end

function exp_ssge()
    LB = -5
    UB = 5
    x = collect(range(LB, UB; length=150))

    M = 100
    η = 0.01
    q = Cauchy(0, 1)
    ssg = SpectralSteinEstimator(η, nothing, 0.99)
    gs = gradient(x) do x
        return log_q_x = sum(loglikelihood(q, x))
    end
    samples = randn(M, 1)
    dlog_q_dx = flatten(compute_gradients(ssg, Flux.unsqueeze(x, 2), samples))
    log_q_x = logpdf.(q, x)
    p = plot(x, gs; label="∇ₓlog q(x), Truth")
    plot!(p, x, dlog_q_dx; label="∇ₓlog q(x), Spectral")
    plot!(p, x, log_q_x; label="log q(x)")
    samples = flatten(samples)
    scatter!(
        p,
        samples,
        logpdf.(q, samples);
        markersize=2,
        markeralpha=0.6,
        markerstrokecolor=:red,
        markerstrokestyle=:dot,
    )
    return display(p)
end
