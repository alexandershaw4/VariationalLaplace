# run_example.jl

using Pkg

# --- Step 1: Activate the VariationalLaplace project ---
Pkg.activate("/Users/alexandershaw/Library/CloudStorage/Dropbox/code/VariationalLaplace")

# --- Step 2: Develop the ThalamoCorticalModel package ---
Pkg.develop(raw"/Users/alexandershaw/Library/CloudStorage/Dropbox/code/ThalamoCorticalModel")

# --- Step 3: Import both packages ---
using VariationalLaplace
using ThalamoCorticalModel

# --- Step 4: Run example (replace with your actual call) ---
println("Running example...")

# Dummy example (replace this with your real model + data)
y = [0.0, 0.1, 0.2, 0.1, 0.0]
f(m) = 0.5 .* m
m0 = zeros(5)
S0 = 1e-3 .* I(5)

m, V, D, logL = fit_variational_laplace_thermo(y, f, m0, S0)

println("Posterior mean: ", m)