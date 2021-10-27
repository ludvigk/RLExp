module RLExp

export BDQNLearner, atari_env_factory, NoisyDense, Split, BayesianGreedyExplorer
export SpectralSteinEstimator, entropy_surrogate
export TotalOriginalRewardPerEpisode, TotalBatchOriginalRewardPerEpisode, CloseLogger
export ResizeImage
export get_screen
export NoisyDense2

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
include("algorithms/GDQN.jl")
include("algorithms/explorers.jl")

end
