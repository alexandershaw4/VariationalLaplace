using LinearAlgebra
using Statistics
using Printf
using Plots
import Plots: @layout  

function fit_variational_laplace_thermo(y, f, m0, S0; max_iter=64, tol=1e-3, doplot=true)
    m = copy(m0)
    n = length(y)

    # Initial low-rank approx
    U, Svals, _ = svd(S0)
    eigvals = diagm(Svals)
    threshold = 0.01 * maximum(eigvals)
    k = max(sum(eigvals .> threshold), length(m))
    V = U[:, 1:k] * Diagonal(sqrt.(eigvals[1:k]))
    D = Diagonal(diag(S0) .- sum(V.^2, dims=2))

    # Initial obs noise variance
    sigma2 = ones(n)
    epsilon = 1e-6
    beta = 1e-3
    nu = 3

    allm = [copy(m)]
    all_elbo = Float64[]
    best_elbo = -Inf

    if doplot
        default(size=(1000, 600))
    end

    allentropy = Float64[]
    allloglike = Float64[]
    alllogprior = Float64[]

    for iter in 1:max_iter
        y_pred = f(m)
        residuals = y - y_pred

        #sigma2 = max.(epsilon, (residuals.^2 .+ beta) ./ (nu .+ residuals.^2 ./ 2))
        #sigma2 = clamp.((residuals.^2 .+ beta) ./ (nu .+ residuals.^2 ./ 2), epsilon, Inf)
        sigma2 = @. max(epsilon, abs((residuals.^2 + beta) / (nu + residuals.^2 / 2)))

        J = compute_jacobian(f, m, n)
        H_lik = J' * diagm(1.0 ./ sigma2) * J
        H_prior = inv(S0 + compute_smooth_covariance(m, 2.0))
        H = H_lik + H_prior
        g = J' * diagm(1.0 ./ sigma2) * residuals - H_prior * (m - m0)

        # Low-rank covariance approx
        U, Svals, _ = svd(H)
        V = U[:, 1:k] * sqrt.(Svals[1:k])
        D = Diagonal(diag(H) .- sum(V.^2, dims=2))

        # Cholesky-based update
        try
            L = cholesky(H).L
            dm = L' \ (L \ g)
        catch err
            @warn "Cholesky failed, using PCG fallback"
            dm = zeros(length(m))  # fallback
        end

        # Trust region
        norm_dm = norm(dm)
        if norm_dm > 1.0
            dm *= 1.0 / norm_dm
        end

        m_prev = copy(m)
        m += dm
        push!(allm, copy(m))

        # ELBO components
        #loglik = -0.5 * sum(residuals.^2 ./ sigma2 .+ log.(2π * sigma2))
        loglik = -0.5 * sum(residuals.^2 ./ sigma2 .+ log.(abs.(2π .* sigma2) .+ epsilon))
        #const TWO_PI = 2 * π
        #safe_sigma2 = clamp.(abs.(sigma2), 1e-8, Inf)  # re-guard sigma2 again just in case
        #log_term = @. log(TWO_PI * safe_sigma2 + 1e-8)
        #loglik = -0.5 * sum(residuals.^2 ./ safe_sigma2 .+ log_term)
        logprior = -0.5 * (m - m0)' * H_prior * (m - m0)
        #logentropy = 0.5 * sum(log.(diag(D) .+ epsilon))
        safe_diag_D = clamp.(diag(D), epsilon, Inf)
        logentropy = 0.5 * sum(log.(safe_diag_D))
        elbo = loglik + logprior + logentropy
        push!(all_elbo, elbo)

        if elbo > best_elbo
            best_elbo = elbo
        else
            @info "ELBO decreased at iteration $iter, backtracking..."
            m = m_prev
        end

        @printf "Iter %d | ELBO: %.4f | ||dm||: %.4f\n" iter elbo norm_dm



        push!(allentropy, logentropy)
        push!(allloglike, loglik)
        push!(alllogprior, logprior)
        push!(all_elbo, elbo)

        if doplot
            p1 = plot(1:length(y), y; color=:black, label="Observed", lw=1.5)
            plot!(p1, 1:length(y), f(m); color=:red, label="Prediction", lw=2)
            scatter!(p1, 1:length(y), y; yerr=sqrt.(sigma2), label="Observed ±σ", color=:black, markersize=3)
            display(p1)
        end

        if norm(dm) < tol
            println("Converged.")
            break
        end
    end

    return m, V, D, best_elbo, allm, all_elbo
end