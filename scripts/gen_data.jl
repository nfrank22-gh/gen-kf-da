using Gen_DA.Solver
using JLD2
using Reactant

Reactant.set_default_backend("cuda")



function run_spinup(step, rhs, omega_hat, dt, T_spinup)
    n_spinup = round(Int, T_spinup / dt)
    println("Spinning up for $n_spinup steps...")
    for i in 1:n_spinup
        omega_hat = step(rhs, omega_hat, dt)
    end
    println("Spinup complete.")
    return omega_hat
end

function run_particles(step_p, rhs, omega_hat, dt, T_data, save_every, npart, Re, n, N; animate=false)
    n_steps = round(Int, T_data / dt)
    dir     = "data/particles/Re$(Re)_N$(N)_npart$(npart)"
    mkpath(dir)

    xp_cpu, yp_cpu = random_particles_ic(npart)
    xp = ConcreteRArray(xp_cpu)
    yp = ConcreteRArray(yp_cpu)

    println("Integrating with particles for $n_steps steps...")
    traj, _, xp_traj, yp_traj = integrate(step_p, rhs, omega_hat, xp, yp, dt, n_steps; save_every)
    println("Done. $(size(traj, 3)) snapshots collected.")

    jldsave("$dir/trajectory.jld2"; trajectory=traj, xp_traj, yp_traj, Re, n, N, dt, save_every, npart)
    println("Saved to $dir/trajectory.jld2")

    if animate
        println("Rendering animation...")
        animate_particles(traj, xp_traj, yp_traj; filename="$dir/animation.mp4", fps=80)
        println("Animation saved to $dir/animation.mp4")
    end
end

function run_no_particles(step, rhs, omega_hat, dt, T_data, save_every, Re, n, N)
    n_steps = round(Int, T_data / dt)
    dir     = "data/no_particles/Re=$(Re)_N=$(N)_dt=$(dt)_T=$(T_data)"
    mkpath(dir)

    println("Integrating without particles for $n_steps steps...")
    traj, _ = integrate(step, rhs, omega_hat, dt, n_steps; save_every)
    println("Done. $(size(traj, 3)) snapshots collected.")

    jldsave("$dir/trajectory.jld2"; trajectory=traj, Re, n, N, dt, save_every)
    println("Saved to $dir/trajectory.jld2")

    plot_vorticity(traj[:, :, end]; title="Final vorticity", filename="$dir/vorticity.png")
    println("Plot saved to $dir/vorticity.png")
end

function main()
    Re         = 100
    n          = 4
    N          = 128
    dt         = Float32(0.01)
    T_spinup   = 50
    T_data     = 10000
    save_every = 100
    npart      = 40
    run_with_particles    = false
    run_without_particles = true
    animate               = false

    rhs =  KfRhs(Re, n, N)
    omega_hat = Reactant.to_rarray(random_ic(N))

    # Compile step functions once; shape/type must match all future calls
    step   = @compile kf_step(rhs, omega_hat, dt)
    xp_dummy, yp_dummy = random_particles_ic(npart)
    step_p = @compile kf_step_particles(rhs, omega_hat,
                                         ConcreteRArray(xp_dummy),
                                         ConcreteRArray(yp_dummy), dt)

    omega_hat = run_spinup(step, rhs, omega_hat, dt, T_spinup)

    run_with_particles    && run_particles(step_p, rhs, omega_hat, dt, T_data, save_every, npart, Re, n, N; animate)
    run_without_particles && run_no_particles(step, rhs, omega_hat, dt, T_data, save_every, Re, n, N)
    


end

main()
