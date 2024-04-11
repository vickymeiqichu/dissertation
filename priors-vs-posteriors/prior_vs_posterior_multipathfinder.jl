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

repression_mean = mean(protein_observations[:,2])

f = h5open("multipathfinder_1_cells.h5", "r")
multipathfinder_1_cells = read(f, Chains)

p0_dist = truncated(Normal(repression_mean, 500^2); lower=100., upper=repression_mean*2)  
h_dist = truncated(Normal(4, 0.5); lower=2., upper=6.)
tau_dist = truncated(Normal(18, 10); lower=5., upper=40.)

plot()
prior_samples = rand(p0_dist, 1000)
histogram(prior_samples, alpha = 0.8, label = "Prior", color="teal")
histogram!(multipathfinder_1_cell[:P₀], nbins=20, label = "Posterior", alpha = 0.6, color="mediumpurple1")  
plot!(xlabel = "P₀", ylabel = "frequency")
savefig("prior_vs_posterior_multipathfinder_p0.pdf")

plot()
prior_samples = rand(h_dist, 1000)
histogram(prior_samples, label = "Prior", color="teal")
histogram!(multipathfinder_1_cell[:h], nbins=40, label = "Posterior", alpha = 0.6, color="mediumpurple1")
plot!(xlabel = "h", ylabel = "frequency")
savefig("prior_vs_posterior_multipathfinder_h.pdf")

plot()
prior_samples = rand(tau_dist, 1000)
histogram(prior_samples, label = "Prior", color="teal")
histogram!(multipathfinder_1_cell[:τ], nbins=40, label = "Posterior", alpha = 0.6, color="mediumpurple1")
plot!(xlabel = "τ", ylabel = "frequency")
savefig("prior_vs_posterior_multipathfinder_tau.pdf")