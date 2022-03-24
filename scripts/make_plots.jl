using Plots
using CSV
using DataFrames
using DrWatson

pgfplotsx()

df = CSV.read(datadir("exp_raw", "cartpole_results.csv"), DataFrame)

dqns = df[completecases(df[:,[1,2]]), :]
noisy = df[completecases(df[:,[1,5]]), :]

plot(palette = :seaborn_colorblind)
hline!([195], line = (1, :dash, 1, :lightgray), label=false)
plot!(dqns[:,1], dqns[:,2], label="BE", c=1, lw=1)
plot!(noisy[:,1], noisy[:,5], label="NoisyNet", c=2, lw=1)
xaxis!("Frames")
yaxis!("Score")

savefig(plotsdir("cartpole_results.pdf"))
