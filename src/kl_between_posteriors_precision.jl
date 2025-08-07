function kl_between_posteriors_precision(m2::Vector, L_H2::Matrix, m1::Vector, L_H1::Matrix)
    d = length(m1)

    # log|Σ| = -log|H| where H = L_H*L_H'
    logdetΣ1 = -2 * sum(log.(diag(L_H1)))
    logdetΣ2 = -2 * sum(log.(diag(L_H2)))

    H1 = L_H1 * transpose(L_H1)
    T = L_H2 \ (transpose(L_H2) \ H1) # trace(H1 * Σ2)
    tr_term = tr(T)

    dm = m2 - m1
    Hdmdm = transpose(L_H2) * dm
    quad_term = dot(Hdmdm, Hdmdm)

    return 0.5 * (tr_term + quad_term - d + (logdetΣ1 - logdetΣ2))
end
