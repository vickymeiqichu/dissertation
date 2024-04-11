using Plots, Random, StatsPlots, Bijectors
using Measures, DelayedKalmanFilter, StochasticDelayDiffEq
using Statistics, Turing, LinearAlgebra, Pathfinder, Optim
using DelimitedFiles, CSV, DataFrames, Serialization
using MCMCChainsStorage, HDF5
using DynamicPPL: getlogp, getval, reconstruct, vectorize, setval!, settrans!!

begin
    black = RGBA{Float64}(colorant"#000000")
    TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
    TolVibrantBlue = RGBA{Float64}(colorant"#0077BB")
    TolVibrantMagenta = RGBA{Float64}(colorant"#EE3377")
    newgreen = RGBA{Float64}(colorant"#6FD661")
end;

# importing hmc chains
g = h5open("hmc_3_param.h5", "r")
hmc_3_param=read(g, Chains)

# importing maximum likelihood estimate
mle_estimate_3_param=readdlm("3_param_mle.csv")

# importing pathfinder chains
f = h5open("pathfinder_1_cell.h5", "r")
pathfinder_3_param = read(f, Chains)

# importing multipathfinder chains
j = h5open("multipathfinder_1_cell_10_runs.h5", "r")
multipathfinder_3_param = read(j, Chains)

# density plots for comparing pathfinder and multipathfinder

# P₀
plot()
density(pathfinder_3_param[:P₀], lw=2, label="Pathfinder", color=TolVibrantMagenta)
density!(multipathfinder_3_param[:P₀], lw=2, label="Multi-pathfinder (10 runs)", color=newgreen)
plot!(title="P₀", xlabel="Parameter values", ylabel = "Density")
vline!([3407.99], color=black, lw=3, line=:dashdot, label="Ground truth", legend=:topleft, alpha=0.7)
savefig("path_multipathfinder_p0_inference_density.pdf")

vline!([mle_estimate_3_param[1]], color=TolVibrantOrange, lw=2, line=:dashdot, label="MLE", alpha=0.7)
savefig("path_multipathfinder_p0_inference_density_mle.pdf")

# h 
plot()
density(pathfinder_3_param[:h], lw=2, label="Pathfinder", color=TolVibrantMagenta)
density!(multipathfinder_3_param[:h], lw=2, label="Multi-pathfinder (10 runs)", color=newgreen)
plot!(title="h", xlabel="Parameter values", ylabel = "Density")
vline!([5.17], color=black, lw=3, line=:dash, label="Ground truth", legend=:topleft, alpha=0.7)
savefig("path_multipathfinder_h_inference_density.pdf")

vline!([mle_estimate_3_param[2]], color=TolVibrantOrange, lw=2, line=:dashdot, label="MLE", alpha=0.7)
savefig("path_multipathfinder_h_inference_density_mle.pdf")

# τ
plot()
density(pathfinder_3_param[:τ], lw=2, label="Pathfinder", color=TolVibrantMagenta)
density!(multipathfinder_3_param[:τ], lw=2, label="Multi-pathfinder (10 runs)", color=newgreen)
plot!(title="τ", xlabel="Parameter values", ylabel = "Density")
vline!([30.], color=black, lw=3, line=:dash, label="Ground truth", legend=:topleft, alpha=0.7)
savefig("path_multipathfinder_tau_inference_density.pdf")

vline!([mle_estimate_3_param[3]], color=TolVibrantOrange, lw=2, line=:dashdot, label="MLE", alpha=0.7)
savefig("path_multipathfinder_tau_inference_density_mle.pdf")

# comparing mcmc with pathfinder and multipathfinder

# P₀
plot()
density(hmc_3_param[:P₀], lw=2, label=["MCMC" nothing nothing nothing], margin=10mm, color=TolVibrantBlue)
density!(pathfinder_3_param[:P₀], lw=2, label="Pathfinder", color=TolVibrantMagenta)
density!(multipathfinder_3_param[:P₀], lw=2, label="Multi-pathfinder (10 runs)", color=newgreen)
plot!(title="P₀", xlabel="Parameter values", ylabel = "Density")
vline!([3407.99], color=black, lw=3, line=:dashdot, label="Ground truth", legend=:topleft, alpha=0.7)
savefig("comparisons_p0_inference_density.pdf")

vline!([mle_estimate_3_param[1]], color=TolVibrantOrange, lw=2, line=:dashdot, label="MLE", alpha=0.7)
savefig("comparisons_p0_inference_density_mle.pdf")

# h 
plot()
density(hmc_3_param[:h], lw=2, label=["MCMC" nothing nothing nothing], margin=10mm, color=TolVibrantBlue)
density!(pathfinder_3_param[:h], lw=2, label="Pathfinder", color=TolVibrantMagenta)
density!(multipathfinder_3_param[:h], lw=2, label="Multi-pathfinder (10 runs)", color=newgreen)
plot!(title="h", xlabel="Parameter values", ylabel = "Density")
vline!([5.17], color=black, lw=3, line=:dash, label="Ground truth", legend=:topleft, alpha=0.7)
savefig("comparisons_h_inference_density.pdf")

vline!([mle_estimate_3_param[2]], color=TolVibrantOrange, lw=2, line=:dashdot, label="MLE", alpha=0.7)
savefig("comparisons_h_inference_density_mle.pdf")

# τ
plot()
density(hmc_3_param[:τ], lw=2, label=["MCMC" nothing nothing nothing], margin=10mm, color=TolVibrantBlue)
density!(pathfinder_3_param[:τ], lw=2, label="Pathfinder", color=TolVibrantMagenta)
density!(multipathfinder_3_param[:τ], lw=2, label="Multi-pathfinder (10 runs)", color=newgreen)
plot!(title="τ", xlabel="Parameter values", ylabel = "Density")
vline!([30.], color=black, lw=3, line=:dash, label="Ground truth", legend=:topleft, alpha=0.7)
savefig("comparisons_tau_inference_density.pdf")

vline!([mle_estimate_3_param[3]], color=TolVibrantOrange, lw=2, line=:dashdot, label="MLE", alpha=0.7)
savefig("comparisons_tau_inference_density_mle.pdf")