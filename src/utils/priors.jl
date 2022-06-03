using Tullio, KernelAbstractions, CUDAKernels


export FlatPrior, GeneralPrior, GaussianPrior, MountainCarPrior, CartpolePrior, AcrobotPrior
export LunarLanderPrior, KernelPrior


abstract type AbstractPrior end

struct FlatPrior <: AbstractPrior end
(p::FlatPrior)(s::AbstractArray, t::AbstractArray) = zero(eltype(t))

struct GeneralPrior{F} <: AbstractPrior
    f::F
end
(p::GeneralPrior)(s::AbstractArray, t::AbstractArray) = p.f(s, t)

struct GaussianPrior <: AbstractPrior
    μ
    σ
end
(p::GaussianPrior)(_, t::AbstractArray) = sum((t .- p.μ) .^ 2 ./ (2p.σ .^ 2))


struct MountainCarPrior <: AbstractPrior
    μ
    σ
end

function MountainCarPrior()
    μ = s -> Zygote.@ignore -100 .+ 100.0f0 .* gpu([0 -1; 0 0; 0 1]) * s
    # μ = s -> Zygote.@ignore 100f0 .* gpu([0 -1; 0 0; 0 1]) * s
    σ = 10.0f0
    return CartpolePrior(μ, σ)
end

function MountainCarPrior(σ; ν=1)
    μ = s -> Zygote.@ignore -100 .+ ν * 100.0f0 .* gpu([0 -1; 0 0; 0 1]) * s
    return CartpolePrior(μ, Float32(σ))
end

function MountainCarPrior(μ, σ::Float64)
    return CartpolePrior(μ, Float32(σ))
end

function (p::MountainCarPrior)(s::AbstractArray, t::AbstractArray)
    return sum((t .- p.μ(s)) .^ 2 ./ (2p.σ .^ 2))
end

struct CartpolePrior <: AbstractPrior
    μ
    σ
end

function CartpolePrior()
    μ = s -> Zygote.@ignore 100.0f0 .* gpu([0 0 -1 -1; 0 0 1 1]) * s
    σ = 10.0f0
    return CartpolePrior(μ, σ)
end

function CartpolePrior(σ; ν=1)
    μ = s -> Zygote.@ignore ν * 100.0f0 .* gpu([0 0 -1 -1; 0 0 1 1]) * s
    return CartpolePrior(μ, Float32(σ))
end

function CartpolePrior(μ, σ::Float64)
    return CartpolePrior(μ, Float32(σ))
end

function (p::CartpolePrior)(s::AbstractArray, t::AbstractArray)
    return sum((t .- p.μ(s)) .^ 2 ./ (2p.σ .^ 2))
end

struct AcrobotPrior <: AbstractPrior
    μ
    σ
end

function AcrobotPrior()
    μ = s -> Zygote.@ignore 100 .+ 1000.0f0 .* gpu([0 -1 0 -1 -0.1f0 -0.05f0; 0 0 0 0 0 0; 0 1 0 1 0.1f0 0.05f0]) * s
    σ = 10.0f0
    return AcrobotPrior(μ, σ)
end

function AcrobotPrior(σ)
    μ = s -> Zygote.@ignore 100 .+ 1000.0f0 .* gpu([0 -1 0 -1 -0.1f0 -0.05f0; 0 0 0 0 0 0; 0 1 0 1 0.1f0 0.05f0]) * s
    return AcrobotPrior(μ, Float32(σ))
end

function AcrobotPrior(μ, σ::Float64)
    return AcrobotPrior(μ, Float32(σ))
end

function (p::AcrobotPrior)(s::AbstractArray, t::AbstractArray)
    return sum((t .- p.μ(s)) .^ 2 ./ (2p.σ .^ 2))
end

struct LunarLanderPrior <: AbstractPrior
    μ
    σ
end

function LunarLanderPrior()
    μ = s -> Zygote.@ignore -100.0f0 .* gpu([0 0 0 0 0 0 0 0
        0 0 0 0 -1 -2 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 1 2 0 0]) * s
    σ = 10.0f0
    return LunarLanderPrior(μ, σ)
end

function LunarLanderPrior(σ)
    μ = s -> Zygote.@ignore -100.0f0 .* gpu([0 0 0 0 0 0 0 0
        0 0 0 0 -1 -2 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 1 2 0 0]) * s
    return LunarLanderPrior(μ, Float32(σ))
end

function LunarLanderPrior(μ, σ::Float64)
    return LunarLanderPrior(μ, Float32(σ))
end

function (p::LunarLanderPrior)(s::AbstractArray, t::AbstractArray)
    return sum((t .- p.μ(s)) .^ 2 ./ (2p.σ .^ 2))
end

struct KernelPrior <: AbstractPrior
    l::Float32
end

KernelPrior(l::Real) = KernelPrior(Float32(l))

function (p::KernelPrior)(s::AbstractArray, t::AbstractArray)
    # l = Zygote.@ignore heuristic_lengthscale(s', s')
    l = repeat([p.l], size(s, 1))
    Σ = Zygote.@ignore rbf_kernel(s', s', l) + 0.1I
    K = Zygote.@ignore inv(cholesky(Σ))
    L = batched_mul(batched_mul(t, K), permutedims(t, (2, 1, 3)))
    return mean([tr(L[:, :, i]) for i = 1:size(t, 3)])
end
