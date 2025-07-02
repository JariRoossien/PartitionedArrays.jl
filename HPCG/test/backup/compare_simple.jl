using LinearAlgebra
using SparseArrays
using Random

# Include the standard pipelined CG implementation
include("../src/pipelined_cg.jl")

# Include the simple non-blocking implementation
include("../src/pipelined_cg_non_blocking_simple.jl")

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

# Function to compare iterations between implementations
function compare_iterations(n::Int)
    println("Comparing iterations with $(n)x$(n) grid ($(n*n)x$(n*n) matrix):")
    
    # Create test problem
    A, b, x0 = create_test_problem(n)
    
    # Standard pipelined CG
    timing_data_std = [0.0]
    x_std, _, res0_std, res_std, iters_std = ref_pipelined_cg!(
        copy(x0), A, b, timing_data_std, tolerance=1e-8, maxiter=1000)
    
    # Simple non-blocking pipelined CG
    x_nb, res0_nb, res_nb, iters_nb = non_blocking_pipelined_cg_simple!(
        copy(x0), A, b, tolerance=1e-8, maxiter=1000)
    
    # Compare results
    println("Standard Pipelined CG:")
    println("  Iterations: $iters_std")
    println("  Initial residual: $res0_std")
    println("  Final residual: $res_std")
    println("  Solution norm: $(norm(x_std))")
    
    println("\nSimple Non-blocking Pipelined CG:")
    println("  Iterations: $iters_nb")
    println("  Initial residual: $res0_nb")
    println("  Final residual: $res_nb")
    println("  Solution norm: $(norm(x_nb))")
    
    # Compare solutions
    solution_diff = norm(x_std - x_nb) / norm(x_std)
    println("\nSolution difference (relative): $solution_diff")
    
    # Compare residuals
    residual_diff = abs(res_std - res_nb) / res_std
    println("Residual difference (relative): $residual_diff")
    
    # Check if the iterations are the same
    if iters_std == iters_nb
        println("\nBoth implementations take the same number of iterations!")
    else
        println("\nThe implementations take different numbers of iterations.")
        println("Difference: $(iters_nb - iters_std) iterations")
    end
    
    return iters_std, iters_nb, solution_diff, residual_diff
end

# Run the comparison with a 64x64 grid (4096x4096 matrix)
compare_iterations(64)
