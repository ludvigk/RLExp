struct CloseLogger <: AbstractHook
    lg::Any
end

(hook::CloseLogger)(::PostExperimentStage, agent, env) = close(hook.lg)
