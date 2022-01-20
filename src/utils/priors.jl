export FlatPrior, GeneralPrior, GaussianPrior, MountainCarPrior



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
    μ = s[2,:]
    μ = [μ zeros(size(μ)) -μ]'
    σ = 0.03f0
    return sum((t .- μ) .^ 2 ./ (2σ .^ 2))
end

