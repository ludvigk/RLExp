export stop, CenteredRMSProp

const ϵ = 1e-8

stop(text="Stop.") = throw(StopException(text))

struct StopException{T}
    S::T
end

function Base.showerror(io::IO, ex::StopException, bt; backtrace=true)
    Base.with_output_color(get(io, :color, false) ? :green : :nothing, io) do io
        showerror(io, ex.S)
    end
end

mutable struct CenteredRMSProp
    eta::Float64
    rho::Float64
    state::IdDict
end

CenteredRMSProp(η = 0.001, ρ = 0.90) = CenteredRMSProp(η, ρ, IdDict())

function Flux.Optimise.apply!(o::CenteredRMSProp, x, Δ)
    η, ρ = o.eta, o.rho

    acc, Δ_ave = get!(o.state, x) do
        (zero(x), zero(x))
    end :: Tuple{typeof(x),typeof(x)}

    @. acc = ρ * acc + (1 - ρ) * Δ^2
    @. Δ_ave = ρ * Δ_ave + (1 - ρ) * Δ

    @. Δ *= η / (√(acc - Δ_ave^2) + ϵ)
end