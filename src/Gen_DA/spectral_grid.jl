using FFTW

struct SpectralGrid
    N::Int
    kx::Matrix{Float32}   # (1, N)     — x wavenumbers, broadcast shape
    ky::Matrix{Float32}   # (N÷2+1, 1) — y wavenumbers, broadcast shape
    lap::Matrix{Float32}  # -(kx² + ky²), [1,1] = 1 to avoid division by zero at DC
end

function SpectralGrid(N::Int)
    L  = Float32(2π)
    dx = L / N
    ky_1d = Float32.(2π .* rfftfreq(N, 1/dx))
    kx_1d = Float32.(2π .* fftfreq(N, 1/dx))
    kx  = reshape(kx_1d, 1, N)
    ky  = reshape(ky_1d, N÷2+1, 1)
    lap = -(kx.^2 .+ ky.^2)
    lap[1, 1] = 1f0
    SpectralGrid(N, kx, ky, lap)
end

# Zero-pad a spectral array from its current resolution to N_out.
# Non-mutating (uses cat) for Zygote/ChainRules compatibility.
function spectral_pad(omega_hat, N_out)
    nfreq_in  = size(omega_hat, 1)
    N_in      = size(omega_hat, 2)
    trailing  = size(omega_hat)[3:end]
    nfreq_out = N_out ÷ 2 + 1
    half_in   = N_in ÷ 2
    flat      = reshape(omega_hat, nfreq_in, N_in, :)
    batch     = size(flat, 3)
    T         = eltype(omega_hat)
    low_x    = flat[1:half_in, 1:half_in, :]
    high_x   = flat[1:half_in, N_in-half_in+1:N_in, :]
    top_rows = cat(low_x, zeros(T, half_in, N_out - 2*half_in, batch), high_x; dims=2)
    padded   = cat(top_rows, zeros(T, nfreq_out - half_in, N_out, batch); dims=1)
    return reshape(padded, nfreq_out, N_out, trailing...)
end

# Compute physical-space velocity (u, v) from the stream function in spectral space.
# Uses u = ∂ψ/∂y = iky·ψ̂  and  v = -∂ψ/∂x = -ikx·ψ̂.
# Result is at grid.N resolution.
function velocity_from_psi_hat(grid::SpectralGrid, psi_hat)
    dxOp = complex.(zero(grid.kx), grid.kx)
    dyOp = complex.(zero(grid.ky), grid.ky)
    u = irfft(dyOp   .* psi_hat, grid.N)
    v = irfft(.-dxOp .* psi_hat, grid.N)
    return u, v
end

# Variant for when psi_hat is at a lower resolution than N_out (spectral upsampling).
function velocity_from_psi_hat(grid::SpectralGrid, psi_hat, N_out::Int)
    dxOp = complex.(zero(grid.kx), grid.kx)
    dyOp = complex.(zero(grid.ky), grid.ky)
    u = irfft(spectral_pad(dyOp   .* psi_hat, N_out), N_out, 1:2)
    v = irfft(spectral_pad(.-dxOp .* psi_hat, N_out), N_out, 1:2)
    return u, v
end
