module RLExp

export atari_env_factory
export TotalOriginalRewardPerEpisode, TotalBatchOriginalRewardPerEpisode, CloseLogger
export ResizeImage
export get_screen
export stop, CenteredRMSProp
export AcrobotEnv, GymEnv
export QQFlow
export FlowNet

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
include("utils/utils.jl")
include("utils/custom_grads.jl")
include("algorithms/QQFLOW.jl")

end
