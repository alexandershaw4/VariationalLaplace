function compute_jacobian(f::Function, x::Vector{Float64}, m::Int; epsilon::Float64=1e-6)
    n = length(x)
    J = zeros(m, n)
    Threads.@threads for i in 1:n
        x_forward = copy(x)
        x_backward = copy(x)
        x_forward[i] += epsilon
        x_backward[i] -= epsilon
        J[:, i] = (f(x_forward) - f(x_backward)) / (2 * epsilon)
    end
    return J
end
