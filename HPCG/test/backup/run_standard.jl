using LinearAlgebra
using SparseArrays
using Random

# Include the standard pipelined CG implementation
include("../src/pipelined_cg.jl")

# Define a function to create a test problem
function create_test_problem(n::Int)
    # Create a sparse matrix with some structure
    # Using a 5-point stencil for a 2D Poisson problem
    A = spdiagm(
        0 => 4.0*ones(n*n),
        1 => -ones(n*n-1),
        -1 => -ones(n*n-1),
        n => -ones(n*n-n),
        -n => -ones(n*n-n)
    )
    
    # Remove connections between grid boundaries
    for i in 1:n-1
        A[i*n, i*n+1] = 0.0
        A[i*n+1, i*n] = 0.0
    end
    
    # Create a random right-hand side
    Random.seed!(42)  # For reproducibility
    b = rand(n*n)
    
    # Initial guess is all zeros
    x0 = zeros(n*n)
    
    return A, b, x0
end

# Parse command line arguments
n = parse(Int, ARGS[1])

# Create test problem
A, b, x0 = create_test_problem(n)

# Run standard pipelined CG
timing_data_std = [0.0]
x_std, _, res0_std, res_std, iters_std = ref_pipelined_cg!(
    copy(x0), A, b, timing_data_std, tolerance=1e-8, maxiter=1000)

# Print results
println("STANDARD_RESULTS:")
println(iters_std)
println(res0_std)
println(res_std)
println(norm(x_std))
