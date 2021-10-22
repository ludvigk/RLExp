struct BayesianGreedyExplorer <: AbstractExplorer end

(s::BayesianGreedyExplorer)(values) = begin print(values); findmax(values[1])[2] end
(s::BayesianGreedyExplorer)(values, mask) =  begin print(values); findmax(values[1], mask)[2] end