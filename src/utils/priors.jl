export FlatPrior, GeneralPrior, GaussianPrior, MountainCarPrior, CartpolePrior



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
    μ = s -> Zygote.@ignore -150 .+ 100f0 .* gpu([0 -10; 0 0; 0 10]) * s
    # μ = s -> Zygote.@ignore 100f0 .* gpu([0 -1; 0 0; 0 1]) * s
    σ = 10f0
    return CartpolePrior(μ, σ)
end

function MountainCarPrior(σ)
    μ = s -> Zygote.@ignore -150 .+ 100f0 .* gpu([0 -10; 0 0; 0 10]) * s
    return CartpolePrior(μ, Float32(σ))
end

function MountainCarPrior(μ, σ)
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

function CartpolePrior(σ)
    μ = s -> Zygote.@ignore 100f0 .* gpu([0 0 -1 -1; 0 0 1 1]) * s
    return CartpolePrior(μ, Float32(σ))
end

function CartpolePrior(μ, σ::Float64)
    return CartpolePrior(μ, Float32(σ))
end

function (p::CartpolePrior)(s::AbstractArray, t::AbstractArray)
    return sum((t .- p.μ(s)) .^ 2 ./ (2p.σ .^ 2))
end
