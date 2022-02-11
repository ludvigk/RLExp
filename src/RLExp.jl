module RLExp

export BDQNLearner, atari_env_factory, NoisyDense, NoisyConv, Split, BayesianGreedyExplorer
export SpectralSteinEstimator, entropy_surrogate
export TotalOriginalRewardPerEpisode, TotalBatchOriginalRewardPerEpisode, CloseLogger
export ResizeImage
export get_screen
export DUQNLearner, FlatPrior, GeneralPrior, GaussianPrior, MountainCarPrior, CartpolePrior, AcrobotPrior
export DUQNSLearner
export stop, CenteredRMSProp
export MountainCarPrior
export AcrobotEnv, GymEnv

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
include("utils/gym.jl")
include("utils/acrobot.jl")
include("utils/priors.jl")
include("utils/utils.jl")
include("algorithms/common.jl")
include("algorithms/BDQN.jl")
include("algorithms/GDQN.jl")
include("algorithms/DUQN.jl")
include("algorithms/DUQNS.jl")

end
