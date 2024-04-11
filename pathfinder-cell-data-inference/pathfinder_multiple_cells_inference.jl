using Plots, Random, StatsPlots, Bijectors
using Measures, DelayedKalmanFilter, StochasticDelayDiffEq
using Statistics, Turing, LinearAlgebra, Pathfinder, Optim
using DelimitedFiles, CSV, DataFrames, Serialization
using DifferentialEquations, BenchmarkTools
using MCMCChains, MCMCChainsStorage, HDF5
using DynamicPPL: getlogp, getval, reconstruct, vectorize, setval!, settrans!!

begin
    black = RGBA{Float64}(colorant"#000000")
    TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
    TolVibrantBlue = RGBA{Float64}(colorant"#0077BB")
    TolVibrantMagenta = RGBA{Float64}(colorant"#EE3377")
end;

hillr(X, v, K, n) = v * (K^n) / (X^n + K^n)

function hes_model_drift(du,u,h,p,t)
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

p = [3407.99, 5.17, log(2)/30, log(2)/90, 15.86, 1.27, 30.];
tspan=(0.,1720.);

function make_protein(measurement_std)
    prob = SDDEProblem(hes_model_drift, hes_model_noise, [30.,500.], h, tspan, p; saveat=10);
    sol = solve(prob,RKMilCommute());

    unobserved_data = Array(sol)[:,100:end];
    # measurement_std = 0.1*mean(unobserved_data[2,:])

    protein = unobserved_data[2,:] + measurement_std*randn(length(unobserved_data[2,:]));
    times = 0:10:730

    return hcat(times,protein)
end

# inference 

@model function kf_multiple(datasets, repression_mean, measurement_variance)
    P₀ ~ truncated(Normal(repression_mean, 500^2); lower=1000., upper=repression_mean*2)
    h ~ truncated(Normal(4, 2); lower=2., upper=6.)
    τ ~ truncated(Normal(18, 10); lower=5., upper=40.)# τ ~ Categorical(40)
    
    n_steps = ceil(Int, τ/10.0) + 1

    for data in datasets
        _, distributions = kalman_filter(
            data,
            [P₀, h, log(2)/30, log(2)/90, 15.86, 1.27, τ],
            measurement_variance;
            alg=RK4(),
            off_diagonal_steps = n_steps,
            euler_dt=10.0
        )

        Turing.@addlogprob! logpdf(MvNormal(distributions[:,1], diagm(distributions[:,2])), data[:,2])
    end
end

measurement_std = 600.0
#datasets = [make_protein(measurement_std) for _ in 1:40]
#repression_mean = mean(last.(mean.(datasets, dims=1)))

#f = open("datasets.jls", "w")
#serialize(f, datasets)
#close(f)

#writedlm("repression_mean_cells.csv", repression_mean)

g = open("datasets.jls", "r")
datasets = deserialize(g)
close(g)
repression_mean = readdlm("repression_mean_cells.csv")[]

datasets_30 = datasets[1:30]
repression_mean_30 = mean(last.(mean.(datasets_30, dims=1)))
#writedlm("repression_mean_30_cells.csv", repression_mean_30)

datasets_20 = datasets[1:20]
repression_mean_20 = mean(last.(mean.(datasets_20, dims=1)))
#writedlm("repression_mean_20_cells.csv", repression_mean_20)

datasets_10 = datasets[1:10]
repression_mean_10 = mean(last.(mean.(datasets_10, dims=1)))
#writedlm("repression_mean_10_cells.csv", repression_mean_10)

datasets_1 = [datasets[1]]
repression_mean_1 = mean(last.(mean.(datasets_1, dims=1)))
#writedlm("repression_mean_1_cells.csv", repression_mean_1)

multiple_model = kf_multiple(
    datasets,
    repression_mean,
    measurement_std^2
)

multiple_model_30 = kf_multiple(
    datasets_30,
    repression_mean_30,
    measurement_std^2
)

multiple_model_20 = kf_multiple(
    datasets_20,
    repression_mean_20,
    measurement_std^2
)

multiple_model_10 = kf_multiple(
    datasets_10,
    repression_mean_10,
    measurement_std^2
)

multiple_model_1 = kf_multiple(
    datasets_1,
    repression_mean_1,
    measurement_std^2
)

@time pathfinder_40_cells = pathfinder(multiple_model; ndraws=2_000, optimizer=LBFGS(m=6), init_scale=10)
@time pathfinder_30_cells = pathfinder(multiple_model_30; ndraws=2_000, optimizer=LBFGS(m=6), init_scale=10)
@time pathfinder_20_cells = pathfinder(multiple_model_20; ndraws=2_000, optimizer=LBFGS(m=6), init_scale=10)
@time pathfinder_10_cells = pathfinder(multiple_model_10; ndraws=2_000, optimizer=LBFGS(m=6), init_scale=10)
@time pathfinder_1_cells = pathfinder(multiple_model_1; ndraws=2_000, optimizer=LBFGS(m=6), init_scale=10)

g = h5open("pathfinder_40_cells.h5", "w")
write(g, pathfinder_40_cells.draws_transformed)
close(g)

g = h5open("pathfinder_30_cells.h5", "w")
write(g, pathfinder_30_cells.draws_transformed)
close(g)

g = h5open("pathfinder_20_cells.h5", "w")
write(g, pathfinder_20_cells.draws_transformed)
close(g)

g = h5open("pathfinder_10_cells.h5", "w")
write(g, pathfinder_10_cells.draws_transformed)
close(g)

g = h5open("pathfinder_1_cells.h5", "w")
write(g, pathfinder_1_cells.draws_transformed)
close(g)

# P₀
plot()
density(pathfinder_1_cells.draws_transformed[:P₀], label="1 cell", lw=2)
density!(pathfinder_10_cells.draws_transformed[:P₀], label="10 cells", lw=2)
density!(pathfinder_20_cells.draws_transformed[:P₀], label="20 cells", lw=2)
density!(pathfinder_30_cells.draws_transformed[:P₀], label="30 cells", lw=2)
density!(pathfinder_40_cells.draws_transformed[:P₀], label="40 cells", lw=2)
vline!([3407.99], color=:black, lw=2, line=:dash, label="Ground truth", legend=:topleft)
plot!(xlabel="Sample value", ylabel="Density",title="P₀")
savefig("compare_multiple_cells_three_params_p0.pdf")

# h
plot()
density(pathfinder_1_cells.draws_transformed[:h], label="1 cell", lw=2)
density!(pathfinder_10_cells.draws_transformed[:h], label="10 cells", lw=2)
density!(pathfinder_20_cells.draws_transformed[:h], label="20 cells", lw=2)
density!(pathfinder_30_cells.draws_transformed[:h], label="30 cells", lw=2)
density!(pathfinder_40_cells.draws_transformed[:h], label="40 cells", lw=2)
vline!([5.17], color=:black, lw=2, line=:dash, label="Ground truth", legend=:topleft)
plot!(xlabel="Sample value", ylabel="Density",title="h")
savefig("compare_multiple_cells_three_params_h.pdf")

# τ
plot()
density(pathfinder_1_cells.draws_transformed[:τ], label="1 cell", lw=2)
density!(pathfinder_10_cells.draws_transformed[:τ], label="10 cells", lw=2)
density!(pathfinder_20_cells.draws_transformed[:τ], label="20 cells", lw=2)
density!(pathfinder_30_cells.draws_transformed[:τ], label="30 cells", lw=2)
density!(pathfinder_40_cells.draws_transformed[:τ], label="40 cells", lw=2)
vline!([30.0], color=:black, lw=2, line=:dash, label="Ground truth", legend=:topleft)
plot!(xlabel="Sample value", ylabel="Density",title="τ")
savefig("compare_multiple_cells_three_params_tau.pdf")