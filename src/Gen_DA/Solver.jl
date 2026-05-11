module Solver

export KfRhs, explicit_term, implicit_term, implicit_solve, kf_step, kf_step_particles
export random_ic, random_particles_ic, integrate
export bilinear_interp_periodic
export plot_vorticity, animate_particles

using AbstractFFTs

include("Solver/rhs.jl")
include("Solver/particles.jl")
include("Solver/update_fn.jl")
include("Solver/integrate.jl")
include("Solver/solver_plotting.jl")

end
