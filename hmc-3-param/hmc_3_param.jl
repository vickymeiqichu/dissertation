using Plots, Random, StatsPlots, Bijectors
using Measures, DelayedKalmanFilter, StochasticDelayDiffEq
using Statistics, Turing, LinearAlgebra, Optim
using DelimitedFiles, CSV, DataFrames, Serialization
using MCMCChainsStorage, HDF5
using DynamicPPL: getlogp, getval, reconstruct, vectorize, setval!, settrans!!

begin
    black = RGBA{Float64}(colorant"#000000")
    TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
    TolVibrantBlue = RGBA{Float64}(colorant"#0077BB")
    TolVibrantMagenta = RGBA{Float64}(colorant"#EE3377")
end;

hillr(X, v, K, n) = v * (K^n) / (X^n + K^n) # part of the Hill function, with n = 'h' (Hill coefficient)

function hes_model_drift(du,u,h,p,t) # chemical Langevin equation for simulating system
    P₀, n, μₘ, μₚ, αₘ, αₚ, τ = p
    du[1] = hillr(h(p,t-τ;idxs=2),αₘ,P₀,n) - μₘ*u[1]
    du[2] = αₚ*u[1] - μₚ*u[2]
end

function hes_model_noise(du,u,h,p,t)
    P₀, n, μₘ, μₚ, αₘ, αₚ, τ = p
    du[1] = sqrt(max(0.,hillr(h(p,t-τ;idxs=2),αₘ,P₀,n) + μₘ*u[1]))
    du[2] = sqrt(max(0.,αₚ*u[1] + μₚ*u[2]))
end

h(p, t; idxs::Int) = 1.0;

p = [3407.99, 5.17, log(2)/30, log(2)/90, 15.86, 1.27, 30.]; # choosing a specific set of parameters to solve system
tspan=(0.,1720.);

prob = SDDEProblem(hes_model_drift, hes_model_noise, [30.,500.], h, tspan, p; saveat=10);
sol = solve(prob,RKMilCommute());

unobserved_data = Array(sol)[:,100:end];
measurement_std = 0.1*mean(unobserved_data[2,:])

protein = unobserved_data[2,:] +
    measurement_std*randn(length(unobserved_data[2,:]));

times = 0:10:730
protein_observations = hcat(times,protein)

@model function infer_repression_multi(data, times, repression_mean, measurement_variance)
    P₀ ~ truncated(Normal(repression_mean, 500^2); lower=100., upper=repression_mean*2)
    h ~ truncated(Normal(4, 2); lower=2., upper=6.)
    τ ~ truncated(Normal(18, 10); lower=5., upper=40.)

    _, distributions = kalman_filter(
        hcat(times, data),
        [P₀, h, log(2)/30, log(2)/90, 15.86, 1.27, τ],
        measurement_variance
    )
    data ~ MvNormal(distributions[:,1], diagm(distributions[:,2]))
end


model = infer_repression_multi(
    protein_observations[:,2],
    protein_observations[:,1],
    mean(protein_observations[:,2]),
    measurement_std^2
)

#chain
hmc_3_param = sample(model, NUTS(250, 0.8), MCMCThreads(), 500, 4) # summary statistics

# saving just generated values
A = Array(hmc_3_param, append_chains=false)
f = open("hmc_3_param.jls", "w")
serialize(f, A)
close(f)

# saving chain itself
g = h5open("hmc_3_param.h5", "w")
write(g, hmc_3_param)
close(g)

#mle
mle_estimate_3_param = optimize(model, MLE()) # maximum likelihood estimate (MLE)
for i in 1:10
	mle_loop = optimize(model, MLE())
		if (mle_loop.lp >= mle_estimate_3_param.lp)
			mle_estimate_3_param = mle_loop
		end
end
writedlm("3_param_mle.csv", mle_estimate_3_param.values.array)

#hmc_3_param_plot = plot(hmc_3_param,  margin=10mm)
#savefig(hmc_3_param_plot, "hmc_3_param.pdf")

#traceplot
hmc_3_param_traceplot = plot(hmc_3_param, seriestype = :traceplot, margin = 10mm)
savefig(hmc_3_param_traceplot, "hmc_3_param_inference_traceplot.pdf")

#densityplots
chain_label = ["Chain 1" "Chain 2" "Chain 3" "Chain 4"]

# P₀
plot()
density(hmc_3_param[:P₀], lw=2, label=chain_label, margin=10mm)
plot!(title="P₀", xlabel="Parameter values", ylabel = "Density")
vline!([3407.99], color=black, lw=3, line=:dashdot, label="Ground truth", legend=:topleft, alpha=0.7)
savefig("hmc_3_param_p0_inference_density.pdf")

vline!([mle_estimate_3_param.values.array[1]], color=TolVibrantOrange, lw=2, line=:dashdot, label="MLE", alpha=0.7)
savefig("hmc_3_param_p0_inference_density_mle.pdf")

# h 
plot()
density(hmc_3_param[:h], lw=2, label=chain_label, margin=10mm)
plot!(title="h", xlabel="Parameter values", ylabel = "Density")
vline!([5.17], color=black, lw=3, line=:dash, label="Ground truth", legend=:topleft, alpha=0.7)
savefig("hmc_3_param_h_inference_density.pdf")

vline!([mle_estimate_3_param.values.array[2]], color=TolVibrantOrange, lw=2, line=:dashdot, label="MLE", alpha=0.7)
savefig("hmc_3_param_h_inference_density_mle.pdf")

# τ
plot()
density(hmc_3_param[:τ], lw=2, label=chain_label, margin=10mm)
plot!(title="τ", xlabel="Parameter values", ylabel = "Density")
vline!([30.], color=black, lw=3, line=:dash, label="Ground truth", legend=:topleft, alpha=0.7)
savefig("hmc_3_param_tau_inference_density.pdf")

vline!([mle_estimate_3_param.values.array[3]], color=TolVibrantOrange, lw=2, line=:dashdot, label="MLE", alpha=0.7)
savefig("hmc_3_param_tau_inference_density_mle.pdf")

