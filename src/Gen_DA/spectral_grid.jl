using FFTW

struct SpectralGrid
    N::Int
    kx::Matrix{Float32}      # (1, N)     — x wavenumbers, broadcast shape
    ky::Matrix{Float32}      # (N÷2+1, 1) — y wavenumbers, broadcast shape
    lap::Matrix{Float32}     # -(kx² + ky²), [1,1] = 1 to avoid division by zero at DC
    dc_mask::Matrix{Float32} # 1 everywhere except 0 at [1,1]; used to zero mean flow
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
    dc_mask = ones(Float32, N÷2+1, N)
    dc_mask[1, 1] = 0f0
    SpectralGrid(N, kx, ky, lap, dc_mask)
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

# Leray (Helmholtz-Hodge) projection: given free spectral coefficients (u_hat, v_hat),
# subtract the irrotational component to enforce ∇·u = 0, then zero the DC mode.
# u_hat, v_hat may have arbitrary trailing dimensions (e.g. batch, channel).
function leray_project(grid::SpectralGrid, u_hat, v_hat)
    ikx = complex.(zero(grid.kx), grid.kx)   # (1, N)
    iky = complex.(zero(grid.ky), grid.ky)   # (N÷2+1, 1)
    inv_lap = -1f0 ./ grid.lap               # 1/|k|²; DC slot = -1 but div=0 there
    div_hat = ikx .* u_hat .+ iky .* v_hat
    u_proj  = (u_hat .- ikx .* (div_hat .* inv_lap)) .* grid.dc_mask
    v_proj  = (v_hat .- iky .* (div_hat .* inv_lap)) .* grid.dc_mask
    return u_proj, v_proj
end

# Convert physical-space (u, v) to vorticity ω = ∂v/∂x − ∂u/∂y via spectral curl.
# u, v must have shape (grid.N, grid.N, ...) with arbitrary trailing dimensions.
function vorticity_from_vel(grid::SpectralGrid, u, v)
    ikx = complex.(zero(grid.kx), grid.kx)
    iky = complex.(zero(grid.ky), grid.ky)
    u_hat   = rfft(u, 1:2)
    v_hat   = rfft(v, 1:2)
    omega_hat = ikx .* v_hat .- iky .* u_hat
    return irfft(omega_hat, grid.N, 1:2)
end

# Compute physical-space velocity (u, v) from the stream function in spectral space.
# Uses u = ∂ψ/∂y = iky·ψ̂  and  v = -∂ψ/∂x = -ikx·ψ̂.
# Result is at grid.N resolution.
function velocity_from_psi_hat(grid::SpectralGrid, psi_hat)
    dxOp = complex.(zero(grid.kx), grid.kx)
    dyOp = complex.(zero(grid.ky), grid.ky)
    u = irfft(dyOp   .* psi_hat, grid.N, 1:2)
    v = irfft(.-dxOp .* psi_hat, grid.N, 1:2)
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
