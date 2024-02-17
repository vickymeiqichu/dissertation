using Plots, Random, StatsPlots, Bijectors
using Measures, DelayedKalmanFilter, StochasticDelayDiffEq
using Statistics, Turing, LinearAlgebra, Optim
using DelimitedFiles, CSV, DataFrames, Serialization
using DynamicPPL: getlogp, getval, reconstruct, vectorize, setval!, settrans!!

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

hmc_p0_3000 = sample(model, NUTS(0.8), MCMCThreads(), 3000, 4) # summary statistics

f = open("hmc_p0_only.jls", "w")
serialize(f, hmc_3_param)
close(f)

hmc_mle_estimate_p0 = optimize(model, MLE()) # maximum likelihood estimate (MLE)
write(string("hmc_p0_only_mle.txt"), hmc_mle_estimate_p0.values.array)

hmc_p0_only_plot = plot(hmc_p0_3000,  margin=10mm)
savefig(hmc_p0_only_plot, "hmc_p0_only.pdf")