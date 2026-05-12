using Test
using Gen_DA.NN
using Lux, Lux.Training, Optimisers, Random
using Enzyme

@testset "build_optimizer" begin
    opt = NN.build_optimizer(1f-3)
    @test opt isa Optimisers.Adam
end

