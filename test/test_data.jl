using Test
using Gen_DA.NN
using Random

@testset "DataPipeline.split_trajectory" begin
    N = 4
    n_snaps = 10
    traj = Float32.(reshape(1:N*N*n_snaps, N, N, n_snaps))
    dt = 0.1f0
    save_every = 1
    T_train = 0.5  # snapshots at t=0,0.1,...,0.5 → indices 1..6 in training

    train, eval_snaps = NN.DataPipeline.split_trajectory(traj, dt, save_every, T_train)

    @test size(train, 3) + size(eval_snaps, 3) == n_snaps
    @test size(train, 3) == 6
    @test size(eval_snaps, 3) == 4
    @test train == traj[:, :, 1:6]
    @test eval_snaps == traj[:, :, 7:end]
end

@testset "DataPipeline.make_sensor_array" begin
    N = 16
    n_meas_space = 10
    rng = MersenneTwister(42)
    sensor_ci = NN.DataPipeline.make_sensor_array(N, n_meas_space, rng)

    @test length(sensor_ci) == n_meas_space
    @test all(1 <= ci[1] <= N && 1 <= ci[2] <= N for ci in sensor_ci)
    @test length(unique(sensor_ci)) == n_meas_space

    rng2 = MersenneTwister(42)
    sensor_ci2 = NN.DataPipeline.make_sensor_array(N, n_meas_space, rng2)
    @test sensor_ci == sensor_ci2
end

@testset "DataPipeline.batch_partition" begin
    n_train = 10
    batch_size = 3
    rng = MersenneTwister(1)
    batches = NN.DataPipeline.batch_partition(n_train, batch_size, rng)

    all_indices = sort(vcat(batches...))
    @test all_indices == collect(1:n_train)
    @test length(batches) == 4           # ceil(10/3) = 4
    @test length(batches[end]) == 1      # last batch has remainder
end

@testset "DataPipeline.extract_vorticity_spectral" begin
    using FFTW
    N = 16
    nfreq = 4
    NDOF = 2 * nfreq - 1   # 7
    half = NDOF ÷ 2        # 3

    rng = MersenneTwister(11)
    snaps = randn(rng, Float32, N, N, 3)
    indices = [1, 3]

    oh_re, oh_im = NN.DataPipeline.extract_vorticity_spectral(snaps, indices, nfreq, NDOF, N)

    @test size(oh_re) == (nfreq, NDOF, 2)
    @test size(oh_im) == (nfreq, NDOF, 2)

    for (b, idx) in enumerate(indices)
        oh = rfft(snaps[:, :, idx])
        # positive-kx window
        @test oh_re[:, 1:half, b]      ≈ real.(oh[1:nfreq, 1:half])
        @test oh_im[:, 1:half, b]      ≈ imag.(oh[1:nfreq, 1:half])
        # negative-kx window
        @test oh_re[:, half+2:NDOF, b] ≈ real.(oh[1:nfreq, N-half+1:N])
        @test oh_im[:, half+2:NDOF, b] ≈ imag.(oh[1:nfreq, N-half+1:N])
        # Nyquist column stays zero
        @test all(iszero, oh_re[:, half+1, b])
        @test all(iszero, oh_im[:, half+1, b])
    end
end

@testset "DataPipeline.extract_observations" begin
    using FFTW
    N = 8

    rng = MersenneTwister(7)
    omega = randn(rng, Float32, N, N)

    grid = SpectralGrid(N)

    # Hand-compute expected u, v using SpectralGrid operators
    omega_hat  = rfft(omega)
    psi_hat    = omega_hat ./ complex.(grid.lap)
    u_expected = irfft(complex.(zero(grid.ky), grid.ky) .* psi_hat, N)
    v_expected = irfft(complex.(zero(grid.kx), .-grid.kx) .* psi_hat, N)

    sensor_ci = [CartesianIndex(2, 3), CartesianIndex(5, 7)]
    snaps = reshape(omega, N, N, 1)
    u_meas, v_meas = NN.DataPipeline.extract_observations(snaps, [1], sensor_ci, grid)

    @test size(u_meas) == (2, 1)
    @test size(v_meas) == (2, 1)
    @test u_meas[1, 1] ≈ u_expected[2, 3] rtol=1e-4
    @test u_meas[2, 1] ≈ u_expected[5, 7] rtol=1e-4
    @test v_meas[1, 1] ≈ v_expected[2, 3] rtol=1e-4
    @test v_meas[2, 1] ≈ v_expected[5, 7] rtol=1e-4
end
