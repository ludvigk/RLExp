#= Julia code for launching jobs on the slurm cluster.

This code is expected to be run from an sbatch script after a module load julia command has been run.
It starts the remote processes with srun within an allocation.
If you get an error make sure to Pkg.checkout("CluterManagers").

=#
try
	using ClusterManagers
catch
	import Pkg
	Pkg.add("ClusterManagers")
	Pkg.checkout("ClusterManagers")
end

using ClusterManagers
using Distributed
using DrWatson
@quickactivate :RLExp

n_workers = parse(Int, ENV["SLURM_NTASKS"])
addprocs_slurm(n_workers; topology = :master_worker, exeflags="--project=.", "--color=yes")

@everywhere using DrWatson
@everywhere @quickactivate :RLExp

games = ["breakout"]
experiments = [E`RLExp_BDQN_Atari($(game))` for game in games]

pmap(run, experiments)

for i in workers()
	rmprocs(i)
end
