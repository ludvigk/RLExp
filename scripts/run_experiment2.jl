#= Julia code for launching jobs on the slurm cluster.

This code is expected to be run from an sbatch script after a module load julia command has been run.
It starts the remote processes with srun within an allocation.
If you get an error make sure to Pkg.checkout("CluterManagers").

=#
using ClusterManagers
using Distributed
using DrWatson

# @quickactivate :RLExp

n_workers = parse(Int, ENV["SLURM_NTASKS"])
addprocs_slurm(n_workers; topology = :master_worker, exeflags=["--project=.", "--color=yes"])

@everywhere begin
    using Pkg; Pkg.activate(".")
    using Distributed
    using DrWatson
    using RLExp
    include("RLExp_DUQNS_MountainCar.jl")
 end

# include("RLExp_DUQN_Atari.jl")
# include("RLExp_DUQNS_Atari.jl")

# include("RLExp_Noisy_Atari.jl")
# include("RLExp_GDQN_Atari.jl")
# include("Dopamine_DQN_Atari.jl")
# include("RLExp_NDQN_Atari.jl")

# games = ["breakout"]
# experiments = [E`RLExp_BDQN_Atari($(game))` for game in games]
# experiments = [E`RLExp_Noisy_Atari(pong)`]

@everywhere begin
    config = Dict(
    "B_lr" => 1e-3,
    "Q_lr" => 1.0,
    "B_clip_norm" => 1000.0,
    "B_update_freq" => 1,
    "Q_update_freq" => 1_000,
    "B_opt" => "ADAM",
    "gamma" => 0.99f0,
    "update_horizon" => 1,
    "batch_size" => 32,
    "min_replay_history" => 10_000,
    "updates_per_step" => 1,
    "λ" => 1.0,
    "prior" => "FlatPrior()",
    "n_samples" => 100,
    "η" => 0.01,
    "nev" => 6,
    "n_eigen_threshold" => 0.99,
    "is_enable_double_DQN" => true,
    "traj_capacity" => 1_000_000,
    "seed" => 1,
    )

    config1 = copy(config)
    config1["prior"] = "FlatPrior()"
    config2 = copy(config)
    config2["prior"] = "MountainCarPrior(1)"
    config3 = copy(config)
    config3["prior"] = "MountainCarPrior(10)"
    config4 = copy(config)
    config4["prior"] = "MountainCarPrior(50)"
    config5 = copy(config)
    config5["prior"] = "MountainCarPrior(10; ν=-1)"
    config6 = copy(config)
    config6["prior"] = "MountainCarPrior(20; ν=-1)"
    config7 = copy(config)
    config7["prior"] = "MountainCarPrior(50; ν=-1)"
    # confs = [config1, config2, config3, config4]
    confs = [config1, config2, config3, config4, config5, config6, config7]

    exp_confs = [conf for conf in confs for _=1:3]
end

function run_experiment(conf)
    ex = RL.Experiment(Val(:RLExp), Val(:DUQNS), Val(:MountainCar), "name"; config = conf)
    run(ex)
end

map(run_experiment, exp_confs)
# run(experiments[1])
