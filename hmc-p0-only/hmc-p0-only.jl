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

p = [3407.99, 5.17, log(2)/30, log(2)/90, 15.86, 1.27, 30.]; # choosing a specific set of parameters to solve equation
tspan=(0.,1720.);

prob = SDDEProblem(hes_model_drift, hes_model_noise, [30.,500.], h, tspan, p; saveat=10);
sol = solve(prob,RKMilCommute());

unobserved_data = Array(sol)[:,100:end];
measurement_std = 0.1*mean(unobserved_data[2,:])

protein = unobserved_data[2,:] +
    measurement_std*randn(length(unobserved_data[2,:]));

times = 0:10:730
protein_observations = hcat(times,protein)

@model function infer_repression(data, times, repression_mean, measurement_variance)
    P₀ ~ truncated(Normal(repression_mean, 500^2); lower=100., upper=repression_mean*2)

    _, distributions = kalman_filter(
        hcat(times, data),
        [P₀, 5.17, log(2)/30, log(2)/90, 15.86, 1.27, 30.],
        measurement_variance
    )
    data ~ MvNormal(distributions[:,1], diagm(distributions[:,2]))
end


model = infer_repression(
    protein_observations[:,2],
    protein_observations[:,1],
    mean(protein_observations[:,2]),
    measurement_std^2
)

#chain
hmc_p0_3000 = sample(model, NUTS(0.8), MCMCThreads(), 3000, 4) # summary statistics

# just the generated values
A = Array(hmc_p0_3000, append_chains=false)
f = open("hmc_p0_only.jls", "w")
serialize(f, A)
close(f)

# the complete chain
g = h5open("hmc_p0_only.h5", "w")
write(g, hmc_p0_3000)
close(g)

#mle
mle_estimate_p0 = optimize(model, MLE()) # maximum likelihood estimate (MLE)
 
for i in 1:10
	mle_loop = optimize(model, MLE())
		if (mle_loop.lp >= mle_estimate_p0.lp)
			mle_estimate_p0 = mle_loop
		end
end
writedlm("p0_only_mle.csv", mle_estimate_p0.values.array)

#traceplot
hmc_p0_traceplot = plot(hmc_p0_3000, seriestype = :traceplot, margin = 10mm)
savefig(hmc_p0_traceplot, "hmc_p0_inference_traceplot.pdf")

#densityplot
plot()
chain_label = ["Chain 1" "Chain 2" "Chain 3" "Chain 4"]

hmc_p0_densityplot = density(hmc_p0_3000[:P₀], lw=2, label=chain_label, margin=10mm)
plot!(title="P₀", xlabel="Parameter values", ylabel = "Density")
vline!([3407.99], color=TolVibrantBlue, lw=3, line=:dashdot, label="Ground truth", legend=:topright, alpha=0.7)
savefig("hmc_p0_inference_density.pdf")
vline!([mle_estimate_p0.values.array], color=TolVibrantOrange, lw=3, line=:dashdot, label="MLE", alpha=0.7)
savefig("hmc_p0_inference_density_mle.pdf")