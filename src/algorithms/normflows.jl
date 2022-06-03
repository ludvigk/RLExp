using Flux
import Distributions: logpdf

export UbPlanar

struct UvPlanar
    u
    w
end

Flux.@functor UvPlanar
Flux.trainable(f::UvPlanar) = (f.u, f.w)

tanh_prime(x) = 1 - tanh(x)^2

function (f::UvPlanar)(x)
    return x .+ f.u .* tanh.(f.w .* x)
end

function logpdf(f, x)
    # println(f.(x))
    # return logpdf.(f.base_dist, f(x)) .- log.(x)
    return log.(abs.(1 .+ f.u .* tanh_prime.(f.w .* x) .* f.w) .+ 1.0f-6)
end

# struct Flow
#     u
#     v
# end
# function (f::Flow)(x)
#     layers
# end
# function (f::Flow)(x)
#     for layer in f.layers
#         x = layer(x)
#         logp = logpdf()
#     return x, logp
# end


# Flux.@functor Flow