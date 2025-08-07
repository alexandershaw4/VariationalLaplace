using Pkg
Pkg.activate(".")

push!(LOAD_PATH, "/Users/alexandershaw/Library/CloudStorage/Dropbox/code/VariationalLaplace/src")
using VariationalLaplace
using Random, LinearAlgebra, Plots

# Seed RNG for reproducibility
Random.seed!(1234)

# Define a simple nonlinear model function
function test_model(m::Vector{Float64})
    return fill(sin(m[1]) + 0.5 * m[2]^2, 20)  # Constant output
end

# Generate synthetic observations
true_m = [1.2, 0.6]
y_true = test_model(true_m)
y_obs = y_true .+ 0.1 .* randn(length(y_true))  # Add noise

# Set up initial guess and prior covariance
m0 = [0.0, 0.0]
S0 = Matrix(I, 2, 2)

# Fit using VL
m_est, V, D, logL, allm, all_elbo = fit_variational_laplace_thermo(
    y_obs, test_model, m0, S0;
    max_iter=50, tol=1e-4, doplot=false
)

println("\nEstimated posterior mean:")
println(m_est)

# Plot ELBO progression
plot(all_elbo, label="ELBO", lw=2)
xlabel!("Iteration")
ylabel!("Free Energy")
title!("Variational Laplace Convergence")
