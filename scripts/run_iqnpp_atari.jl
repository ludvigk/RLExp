#= Julia code for launching jobs on the slurm cluster.

This code is expected to be run from an sbatch script after a module load julia command has been run.
It starts the remote processes with srun within an allocation.
If you get an error make sure to Pkg.checkout("CluterManagers").

=#
using DrWatson

# @quickactivate :RLExp

# n_workers = parse(Int, ENV["SLURM_NTASKS"])
# addprocs_slurm(n_workers; topology=:master_worker, exeflags=["--project=.", "--color=yes"])

using Pkg
Pkg.activate(".")
using DrWatson
using RLExp
include("RLExp_IQN_Atari.jl")

run(E`RLExp_IQNPP_Atari(game=pong)`)
