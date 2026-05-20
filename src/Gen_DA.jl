module Gen_DA
  include("Gen_DA/spectral_grid.jl")
  include("Gen_DA/Solver.jl")
  include("Gen_DA/NN.jl")
  export Solver, NN, SpectralGrid, spectral_pad, velocity_from_psi_hat, vorticity_from_vel
end
