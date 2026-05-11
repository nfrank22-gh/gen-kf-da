const alpha = Float32[0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748, 1]
const beta  = Float32[0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257]
const gamma = Float32[0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681]

function kf_step(rhs::KfRhs, omega_hat, dt)
    u = omega_hat
    h = zero(omega_hat)
    for i in 1:5
        g, _, _ = explicit_term(rhs, u)
        h = g .+ beta[i] .* h
        mu = 0.5f0 * dt * (alpha[i+1] - alpha[i])
        imp_rhs = u .+ gamma[i] .* dt .* h .+ mu .* implicit_term(rhs, u)
        u = implicit_solve(rhs, imp_rhs, mu)
    end
    return u
end

function kf_step_particles(rhs::KfRhs, omega_hat, xp, yp, dt)
    u    = omega_hat
    h    = zero(omega_hat)
    h_xp = zero(xp)
    h_yp = zero(yp)
    for i in 1:5
        g, u_grid, v_grid = explicit_term(rhs, u)
        h = g .+ beta[i] .* h
        mu = 0.5f0 * dt * (alpha[i+1] - alpha[i])
        imp_rhs = u .+ gamma[i] .* dt .* h .+ mu .* implicit_term(rhs, u)
        u = implicit_solve(rhs, imp_rhs, mu)
        u_p = bilinear_interp_periodic(u_grid, xp, yp)
        v_p = bilinear_interp_periodic(v_grid, xp, yp)
        xp, yp, h_xp, h_yp = tracer_substep(xp, yp, h_xp, h_yp, u_p, v_p, beta[i], gamma[i], dt)
    end
    xp = mod.(xp, Float32(L))
    yp = mod.(yp, Float32(L))
    return u, xp, yp
end
