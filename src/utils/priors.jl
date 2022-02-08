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


struct MountainCarPrior <: AbstractPrior end

function (p::MountainCarPrior)(s::AbstractArray, t::AbstractArray)
    # μ = abs.(s[2,:])
    μ = Zygote.@ignore 100f0 .* gpu([-1 -1; 0 0; 1 1]) * s
    σ = 1f0
    return sum((t .- μ) .^ 2 ./ (2σ .^ 2))
end


struct CartpolePrior <: AbstractPrior end

function (p::CartpolePrior)(s::AbstractArray, t::AbstractArray)
    # μ = Zygote.@ignore view(s .> 0, 3, :)
    # μ = Zygote.@ignore [(1 .+ μ) (1 .- μ)]' .* 100f0
    μ = Zygote.@ignore 100f0 .* gpu([0 0 0 0; 0 0 1 1]) * s
    σ = 10f0
    return sum((t .- μ) .^ 2 ./ (2σ .^ 2))
end
