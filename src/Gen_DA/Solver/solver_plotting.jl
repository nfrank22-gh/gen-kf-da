using CairoMakie

function plot_vorticity(omega; title="Vorticity", filename=nothing)
    fig = Figure()
    ax  = Axis(fig[1, 1]; title=title, xlabel="x", ylabel="y", aspect=DataAspect())
    lim = maximum(abs, omega)
    hm  = heatmap!(ax, omega; colormap=:RdBu, colorrange=(-lim, lim))
    Colorbar(fig[1, 2], hm)
    isnothing(filename) || save(filename, fig)
    return fig
end

function vort_to_vel(omega::Matrix{T}) where T
    N  = size(omega, 1)
    dx = T(2pi / N)
    ky = T.(2pi .* rfftfreq(N, 1/dx))
    kx = T.(2pi .* fftfreq(N, 1/dx))
    KY = reshape(ky, :, 1) .* ones(T, 1, N)
    KX = ones(T, N÷2+1, 1) .* reshape(kx, 1, :)
    laplacian      = -(KX.^2 .+ KY.^2)
    laplacian[1,1] = one(T)
    omega_hat = rfft(omega)
    psi_hat   = omega_hat ./ laplacian
    u = irfft(im .* KY .* psi_hat, N)
    v = irfft(.-(im .* KX .* psi_hat), N)
    return u, v
end

function animate_particles(trajectory, xp_traj, yp_traj;
                            filename="animation.mp4", fps=30, slowdown=6, skip=8)
    N       = size(trajectory, 1)
    n_saves = size(trajectory, 3)
    x_grid  = LinRange(0f0, Float32(2pi), N + 1)[1:end-1]
    idx     = 1:skip:N
    spacing = Float32(x_grid[1 + skip] - x_grid[1])

    println("  Precomputing velocity fields...")
    u_traj = similar(trajectory)
    v_traj = similar(trajectory)
    for i in 1:n_saves
        u_traj[:,:,i], v_traj[:,:,i] = vort_to_vel(trajectory[:,:,i])
    end

    n_sub     = length(idx)
    arrow_tails = [Point2f(x_grid[idx[jx]], x_grid[idx[jy]])
                   for jy in 1:n_sub for jx in 1:n_sub]

    frame      = Observable(1)
    part_pts   = @lift(Point2f.(xp_traj[:, $frame], yp_traj[:, $frame]))
    arrow_dirs = map(frame) do f
        u = u_traj[idx, idx, f]
        v = v_traj[idx, idx, f]
        m = max(maximum(sqrt.(u.^2 .+ v.^2)), 1f-6)
        s = spacing * 0.7f0 / m
        [Vec2f(u[jy,jx]*s, v[jy,jx]*s)
         for jy in 1:n_sub for jx in 1:n_sub]
    end

    fig = Figure()
    ax  = Axis(fig[1,1]; xlabel="x", ylabel="y", aspect=DataAspect(),
               limits=(0, 2pi, 0, 2pi))
    arrows2d!(ax, arrow_tails, arrow_dirs; color=:black, tiplength=12, tipwidth=8, shaftwidth=2)
    scatter!(ax, part_pts; color=:red, markersize=5, strokewidth=0)

    all_frames = [i for i in 1:n_saves for _ in 1:slowdown]
    CairoMakie.record(fig, filename, all_frames; framerate=fps) do i
        frame[] = i
    end
    return filename
end
