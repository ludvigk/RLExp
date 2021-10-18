module RLExp

export BDQNLearner, CloseLogger, atari_env_factory, NoisyDense
export SpectralSteinEstimator, entropy_surrogate

using ArcadeLearningEnvironment
using ArcadeLearningEnvironment: getScreenRGB, getScreenWidth, getScreenHeight
using CUDA
using Dates
using Flux
using Images
using ImageTransformations: imresize!
using LinearAlgebra
using Plots
using Random
using Reexport
@reexport using ReinforcementLearning
using Setfield
using Statistics
using Zygote

include("utils/hooks.jl")
include("utils/atari.jl")
include("algorithms/common.jl")
include("algorithms/BDQN.jl")

end
