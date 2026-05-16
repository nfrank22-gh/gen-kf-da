
L = 2pi

function get_K(N; T=Float32)
    dx = L / N
    ky = T.(2pi .* rfftfreq(N, 1/dx))
    kx = T.(2pi .* fftfreq(N, 1/dx))
    KY = reshape(ky, :, 1) .* ones(T, 1, N)
    KX = ones(T, N÷2+1, 1) .* reshape(kx, 1, :)
    return KX, KY
end

function get_dealias_mask(N; T=Float32)
    my_r = (fftfreq(N) .* N)[1:N÷2+1]
    mx   = fftfreq(N) .* N
    MY = reshape(my_r, :, 1) .* ones(1, N)
    MX = ones(N÷2+1, 1) .* reshape(mx, 1, :)
    M  = (abs.(MX) .<= N/3) .& (abs.(MY) .<= N/3)
    return T.(M)
end

function kolmogorov_forcing(L, N, n; T=Float32)
    y = T.((0:N-1) .* (L/N))
    Y = reshape(y, :, 1) .* ones(T, 1, N)
    return T(-n) .* cos.(T(n) .* Y)
end

struct KfRhs{T, RA, CA}
    M::RA
    dyOp::CA
    dxOp::CA
    diffOp::CA
    laplacianOp::CA
    forcingHat::CA


    function KfRhs(Re, n, N; T=Float32)
        KX, KY = get_K(N; T=T)
        dxOp = im .* KX
        dyOp = im .* KY
        laplacian = dxOp.^2 .+ dyOp.^2
        diffOp = laplacian ./ T(Re)
        laplacianOp = copy(laplacian)
        laplacianOp[1, 1] = one(Complex{T})
        M = get_dealias_mask(N; T=T)
        forcing = kolmogorov_forcing(L, N, n; T=T)
        forcingHat = rfft(forcing)
        new{T, typeof(M), typeof(forcingHat)}(M, dyOp, dxOp, diffOp, laplacianOp, forcingHat)
    end
end

function vort_hat_2_vel_hat(rhs::KfRhs, omega_hat)
    psi_hat = omega_hat ./ rhs.laplacianOp
    u_hat   = rhs.dyOp .* psi_hat
    v_hat   = .-rhs.dxOp .* psi_hat
    return u_hat, v_hat
end

function explicit_term(rhs::KfRhs, omega_hat)
    N = size(rhs.M, 2)
    u_hat, v_hat = vort_hat_2_vel_hat(rhs, omega_hat)
    u = irfft(u_hat, N)
    v = irfft(v_hat, N)
    dw_dx = irfft(rhs.dxOp .* omega_hat, N)
    dw_dy = irfft(rhs.dyOp .* omega_hat, N)
    adv_hat = rfft(.-(u .* dw_dx .+ v .* dw_dy))
    return adv_hat .* rhs.M .+ rhs.forcingHat, u, v
end

function implicit_term(rhs::KfRhs, omega_hat)
    return rhs.diffOp .* omega_hat
end

function implicit_solve(rhs::KfRhs, omega_hat, mu)
    return omega_hat ./ (1 .- mu .* rhs.diffOp)
end
