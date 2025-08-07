module VariationalLaplace

using LinearAlgebra
using Statistics
using Printf
using .Threads

# Include all source files (but not new modules!)
include("fit_variational_laplace_thermo.jl")
include("compute_jacobian.jl")
include("compute_smooth_covariance.jl")
include("kl_between_posteriors_precision.jl")

# Export only what you want available externally
export fit_variational_laplace_thermo

end