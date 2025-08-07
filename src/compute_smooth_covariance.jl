using LinearAlgebra

function compute_smooth_covariance(x::Vector{Float64}, lengthscale::Float64)
    n = length(x)
    K = zeros(n, n)
    for i in 1:n
        for j in 1:n
            sqdist = (x[i] - x[j])^2
            K[i, j] = exp(-sqdist / (2 * lengthscale^2))
        end
    end
    K += 1e-6 * I # Add jitter
    return K
end
