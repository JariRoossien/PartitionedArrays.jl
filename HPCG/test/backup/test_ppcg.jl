using Test
using LinearAlgebra
using PartitionedArrays
using HPCG

function test_ppcg()
    # Create a distributed environment
    parts = (2,)
    ranks = LinearIndices(parts)[:]
    comm = MPIComm(ranks)
    
    # Create a simple test problem
    n = 100
    A_local = spdiagm(0 => 2.0*ones(n), 1 => -1.0*ones(n-1), -1 => -1.0*ones(n-1))
    x_local = zeros(n)
    b_local = ones(n)
    
    # Create partitioned arrays
    A = PMatrix(A_local, comm, (n,n))
    x = PVector(x_local, comm, n)
    b = PVector(b_local, comm, n)
    
    # Solve using pipelined CG
    timing_data = [0.0]
    x_sol, timing, res0, res, iters = ref_pipelined_cg!(
        copy(x), A, b, timing_data, tolerance=1e-8, maxiter=1000)
    
    println("PPCG completed in $iters iterations")
    println("Initial residual: $res0")
    println("Final residual: $res")
    println("Time taken: $(timing[1]) seconds")
    
    # Verify solution by checking residual
    r = similar(b)
    mul!(r, A, x_sol)
    r .= b .- r
    final_res = norm(r)
    
    @test isapprox(final_res, res, rtol=1e-6)
    @test final_res/res0 <= 1e-8
    
    return true
end

# Run the test
test_ppcg()
