export FlatPrior, GeneralPrior, GaussianPrior, MountainCarPrior, CartpolePrior, AcrobotPrior
export LunarLanderPrior



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
    μ = s -> Zygote.@ignore -100 .+ 100f0 .* gpu([0 -1; 0 0; 0 1]) * s
    # μ = s -> Zygote.@ignore 100f0 .* gpu([0 -1; 0 0; 0 1]) * s
    σ = 10f0
    return CartpolePrior(μ, σ)
end

function MountainCarPrior(σ)
    μ = s -> Zygote.@ignore -100 .+ 100f0 .* gpu([0 -1; 0 0; 0 1]) * s
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
    μ = s -> Zygote.@ignore 100f0 .* gpu([0 0 -1 -1; 0 0 1 1]) * s
    σ = 10f0
    return CartpolePrior(μ, σ)
end

function CartpolePrior(σ; ν=1)
    μ = s -> Zygote.@ignore ν * 100f0 .* gpu([0 0 -1 -1; 0 0 1 1]) * s
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
    μ = s -> Zygote.@ignore 100 .+ 1000f0 .* gpu([0 -1 0 -1 -0.1f0 -0.05f0; 0 0 0 0 0 0; 0 1 0 1 0.1f0 0.05f0]) * s
    σ = 10f0
    return AcrobotPrior(μ, σ)
end

function AcrobotPrior(σ)
    μ = s -> Zygote.@ignore 100 .+ 1000f0 .* gpu([0 -1 0 -1 -0.1f0 -0.05f0; 0 0 0 0 0 0; 0 1 0 1 0.1f0 0.05f0]) * s
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
    μ = s -> Zygote.@ignore -100f0 .* gpu([ 0 0 0 0  0  0 0 0;
                                           0 0 0 0 -1 -2 0 0;
                                           0 0 0 0  0  0 0 0;
                                           0 0 0 0  1  2 0 0]) * s
    σ = 10f0
    return LunarLanderPrior(μ, σ)
end

function LunarLanderPrior(σ)
    μ = s -> Zygote.@ignore -100f0 .* gpu([ 0 0 0 0  0  0 0 0;
                                           0 0 0 0 -1 -2 0 0;
                                           0 0 0 0  0  0 0 0;
                                           0 0 0 0  1  2 0 0]) * s
    return LunarLanderPrior(μ, Float32(σ))
end

function LunarLanderPrior(μ, σ::Float64)
    return LunarLanderPrior(μ, Float32(σ))
end

function (p::LunarLanderPrior)(s::AbstractArray, t::AbstractArray)
    σ = rbf_kernel(s, s', 10)
    return sum((t .- 0) * inv(σ) * (t .- 0)')
end