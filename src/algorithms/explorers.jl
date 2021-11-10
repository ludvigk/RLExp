struct BayesianGreedyExplorer <: AbstractExplorer end

# (s::BayesianGreedyExplorer)(values) = print(values);findmax(values[1])[2]
# (s::BayesianGreedyExplorer)(values, mask) = findmax(values[1], mask)[2]