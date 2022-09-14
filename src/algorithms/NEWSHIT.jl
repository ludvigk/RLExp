using Flux: chunk


function mixture_gauss_cdf(x, weights, means, log_scales)
    component_dist = Normal()
end

function flow_forward(x, params, na)
    weights, loc, scale = chunk(params, 3)
    weights = reshape(weights, :, na, size(params, 2))
    loc = reshape(loc, :, na, size(params, 2))
    scale = reshape(scale, :, na, size(params, 2))


end