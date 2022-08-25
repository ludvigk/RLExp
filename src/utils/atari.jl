import ReinforcementLearning.RLBase: action_space

"""
    ResizeImage(img::Array{T, N})
    ResizeImage(dims::Int...) -> ResizeImage(Float32, dims...)
    ResizeImage(T::Type{<:Number}, dims::Int...)
By default the `BSpline(Linear())`` method is used to resize the `state` field
of an observation to size of `img` (or `dims`). In some other packages, people
use the
[`cv2.INTER_AREA`](https://github.com/google/dopamine/blob/2a7d91d2831ca28cea0d3b0f4d5c7a7107e846ab/dopamine/discrete_domains/atari_lib.py#L511-L513),
which is not supported in `ImageTransformations.jl` yet.
"""
struct ResizeImage{T,N}
    img::Array{T,N}
end

ResizeImage(dims::Int...) = ResizeImage(Float32, dims...)
ResizeImage(T::Type{<:Number}, dims::Int...) = ResizeImage(Array{T}(undef, dims))

function (p::ResizeImage)(state::AbstractArray)
    imresize!(p.img, state)
    return p.img
end

function atari_env_factory(
    name,
    state_size,
    n_frames,
    max_episode_steps=100_000;
    seed=nothing,
    repeat_action_probability=0.0,
    n_replica=nothing,
    terminal_on_life_loss=false,
)
    function init(seed)
        return RewardTransformedEnv(
            StateCachedEnv(
                StateTransformedEnv(
                    AtariEnv(;
                        name=string(name),
                        grayscale_obs=true,
                        noop_max=30,
                        frame_skip=1,
                        terminal_on_life_loss=terminal_on_life_loss,
                        repeat_action_probability=repeat_action_probability,
                        max_num_frames_per_episode=n_frames * max_episode_steps,
                        color_averaging=false,
                        full_action_space=false,
                        seed=seed,
                    );
                    state_mapping=Chain(
                        ResizeImage(state_size...), StackFrames(state_size..., n_frames)
                    ),
                    state_space_mapping=_ -> Space(fill(0 .. 256, state_size..., n_frames)),
                ),
            );
            reward_mapping=r -> clamp(r, -1, 1),
        )
    end

    # if isnothing(n_replica)
    #     init(seed)
    # else
    #     envs = [init(isnothing(seed) ? nothing : hash(seed + i)) for i in 1:n_replica]
    #     states = Flux.batch(state.(envs))
    #     rewards = reward.(envs)
    #     terminals = is_terminated.(envs)
    #     A = Space([action_space(x) for x in envs])
    #     S = Space(fill(0 .. 255, size(states)))
    #     MultiThreadEnv(envs, states, rewards, terminals, A, S, nothing)
    # end
    init(seed)
end

"Total reward per episode before reward reshaping"
Base.@kwdef mutable struct TotalOriginalRewardPerEpisode <: AbstractHook
    rewards::Vector{Float64} = Float64[]
    reward::Float64 = 0.0
end

function (hook::TotalOriginalRewardPerEpisode)(
    ::PostActStage, agent, env::RewardTransformedEnv
)
    return hook.reward += reward(env.env)
end

function (hook::TotalOriginalRewardPerEpisode)(::PostEpisodeStage, agent, env)
    push!(hook.rewards, hook.reward)
    return hook.reward = 0
end

"Total reward of each inner env per episode before reward reshaping"
struct TotalBatchOriginalRewardPerEpisode <: AbstractHook
    rewards::Vector{Vector{Float64}}
    reward::Vector{Float64}
end

function TotalBatchOriginalRewardPerEpisode(batch_size::Int)
    return TotalBatchOriginalRewardPerEpisode(
        [Float64[] for _ in 1:batch_size], zeros(batch_size)
    )
end

# function (hook::TotalBatchOriginalRewardPerEpisode)(
#     ::PostActStage, agent, env::MultiThreadEnv{<:RewardTransformedEnv}
# )
#     for (i, e) in enumerate(env.envs)
#         hook.reward[i] += reward(e.env)
#         if is_terminated(e)
#             push!(hook.rewards[i], hook.reward[i])
#             hook.reward[i] = 0.0
#         end
#     end
# end

function get_screen(env::T) where {T<:AbstractEnv}
    game = get_base_atari_env(env)
    screen = getScreenRGB(game.ale)
    w = getScreenWidth(game.ale)
    h = getScreenHeight(game.ale)
    screen = reshape(screen, 3, w, h)
    screen = permutedims(screen, (1, 2, 3))
    img = colorview(RGB, screen ./ 256)
    return img'
end

get_base_atari_env(env::AbstractEnv) = get_base_atari_env(env.env)
get_base_atari_env(env::AtariEnv) = env

@recipe function f(::Type{T}, env::T) where {T<:AbstractEnv}
    return get_screen(env)
end

RLBase.action_space(env::AtariEnv, player::DefaultPlayer) = action_space(env)
