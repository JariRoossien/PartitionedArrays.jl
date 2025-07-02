using LinearAlgebra
using SparseArrays
using Random

# Include the pipelined CG implementation
include("../src/pipelined_cg.jl")

# Define a simple reference CG implementation
function simple_cg(A, b, x0; tol=1e-8, maxiter=1000)
    x = copy(x0)
    r = b - A * x
    p = copy(r)
    rsold = dot(r, r)
    res0 = sqrt(rsold)
    
    for i = 1:maxiter
        Ap = A * p
        alpha = rsold / dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = dot(r, r)
        res = sqrt(rsnew)
        
        if res < tol * res0
            return x, res0, res, i
        end
        
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    end
    
    res = sqrt(rsold)
    return x, res0, res, maxiter
end

# Define a simple test problem
function create_test_problem(n::Int)
    # Create a simple tridiagonal matrix
    A = spdiagm(0 => 2.0*ones(n), 1 => -1.0*ones(n-1), -1 => -1.0*ones(n-1))
    # Create a random right-hand side
    Random.seed!(42)  # For reproducibility
    b = rand(n)
    # Initial guess is all zeros
    x0 = zeros(n)
    return A, b, x0
end

# Function to compare the results of different CG implementations
function compare_cg_implementations()
    n = 100
    A, b, x0 = create_test_problem(n)
    
    # Solve using simple reference CG
    x_ref, res0_ref, res_ref, iters_ref = simple_cg(
        A, b, x0, tol=1e-8, maxiter=1000)
    
    # Solve using pipelined CG
    timing_data_ppcg = [0.0]
    x_ppcg, _, res0_ppcg, res_ppcg, iters_ppcg = non_blocking_pipelined_cg!(
        copy(x0), A, b, timing_data_ppcg, tolerance=1e-8, maxiter=1000)
    
    # Compare results
    println("Reference CG:")
    println("  Iterations: $iters_ref")
    println("  Initial residual: $res0_ref")
    println("  Final residual: $res_ref")
    
    println("\nPipelined CG:")
    println("  Iterations: $iters_ppcg")
    println("  Initial residual: $res0_ppcg")
    println("  Final residual: $res_ppcg")
    
    # Compare solutions
    solution_diff = norm(x_ref - x_ppcg) / norm(x_ref)
    println("\nSolution difference (relative): $solution_diff")
    
    # Compare residuals
    residual_diff = abs(res_ref - res_ppcg) / res_ref
    println("Residual difference (relative): $residual_diff")
    
    # Check if the solutions are close enough
    if solution_diff < 1e-6 && residual_diff < 1e-6
        println("\nThe implementations produce equivalent results!")
        return true
    else
        println("\nThe implementations produce different results.")
        return false
    end
end

# Run the comparison
compare_cg_implementations()
