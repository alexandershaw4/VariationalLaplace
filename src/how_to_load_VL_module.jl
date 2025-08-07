
# script showing how to load the VL module
using Pkg
Pkg.activate(".")  # or wherever your environment is
push!(LOAD_PATH, "path/to/VariationalLaplace/src")
using VariationalLaplace

# Now you can call:
m, V, D, logL, allm, all_elbo = fit_variational_laplace_thermo(y, f, m0, S0)