using Plots
using StatsPlots
using RLExp
using Flux

function main()
    x = [(randn(1, 1000) .- 4) (randn(1,1000)) (randn(1, 1000) .+ 4)]
    flow = Flow([
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
        LuddeFlow(1,1,1),
    ])

    p = Flux.params(flow)
    opt = ADAM(0.01)

    loss(x) = begin
        r = ones(size(x))
        z, logabspz = inverse(flow, x, r)
        sum(z .^ 2 / 2 .- logabspz)
    end
    # println(p)
    Flux.@epochs 3000 Flux.train!(loss, Flux.params(flow), [x], opt)
    # println(p)
    p = density(x[1,:])
    density!(flow(randn(1,600), ones(1,600))[1][1,:])
    density!(inverse(flow, x, ones(1,3000))[1][1,:])
    display(p)
end

main()