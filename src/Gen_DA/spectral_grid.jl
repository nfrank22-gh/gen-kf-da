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
