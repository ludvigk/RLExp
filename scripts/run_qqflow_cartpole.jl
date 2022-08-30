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
addprocs_slurm(n_workers; topology=:master_worker, exeflags=["--project=.", "--color=yes"])

@everywhere begin
    using Pkg
    Pkg.activate(".")
    using Distributed
    using DrWatson
    using RLExp
    using Random
    include("RLExp_QQFLOW_Cartpole.jl")
end

run(E`RLExp_QQFLOW_Cartpole`)
