using LinearAlgebra
using SparseArrays
using Random
using Statistics
using Printf

# Mock implementations of non-blocking dot product functions for timing tests
# This version simulates communication delay in dot products
struct FakeFuture{T}
    result::T
    delay::Float64  # Simulated communication delay in seconds
end

function Base.fetch(future::FakeFuture)
    # Simulate communication delay
    if future.delay > 0
        sleep(future.delay)
    end
    return future.result
end

function setup_non_blocking_dot(a, b)
    return nothing
end

# Configurable delay for dot products (to simulate network latency)
const DOT_PRODUCT_DELAY = 0.001  # 1 millisecond delay

function non_blocking_dot(a, b, setup)
    # Compute the dot product immediately
    result = dot(a, b)
    # Return a future with simulated delay
    return FakeFuture(result, DOT_PRODUCT_DELAY)
end

# Function to create a test problem
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

# Function to run standard pipelined CG
function run_standard_ppcg(A, b, x0; tolerance=1e-8, maxiter=1000)
    # Include the standard pipelined CG implementation
    include("../src/pipelined_cg.jl")
    
    timing_data_std = [0.0]
    start_time = time()
    x_std, _, _, _, std_iters = ref_pipelined_cg!(
        copy(x0), A, b, timing_data_std, tolerance=tolerance, maxiter=maxiter)
    std_time = time() - start_time
    
    return std_time, std_iters
end

# Function to run non-blocking pipelined CG
function run_nonblocking_ppcg(A, b, x0; tolerance=1e-8, maxiter=1000)
    # Include the non-blocking pipelined CG implementation
    include("../src/pipelined_cg_non_blocking.jl")
    
    timing_data_nb = [0.0]
    start_time = time()
    x_nb, _, _, _, nb_iters = non_blocking_pipelined_cg!(
        copy(x0), A, b, timing_data_nb, tolerance=tolerance, maxiter=maxiter)
    nb_time = time() - start_time
    
    return nb_time, nb_iters
end

# Run timing tests for different grid sizes
function run_timing_tests(grid_sizes::Vector{Int}, num_runs::Int=5)
    println("Running timing tests for different grid sizes...")
    println("Using simulated dot product delay of $(DOT_PRODUCT_DELAY*1000) ms")
    println()
    
    # Print header
    @printf("%-10s %-15s %-15s %-15s %-15s %-15s\n", 
            "Grid Size", "Standard Time", "Non-blocking", "Speedup", "Std Iters", "NB Iters")
    println("-" ^ 80)
    
    for n in grid_sizes
        # Create test problem
        A, b, x0 = create_test_problem(n)
        
        # Arrays to store timing results
        standard_times = zeros(num_runs)
        nonblocking_times = zeros(num_runs)
        standard_iters = zeros(Int, num_runs)
        nonblocking_iters = zeros(Int, num_runs)
        
        for i in 1:num_runs
            # Run standard pipelined CG
            standard_times[i], standard_iters[i] = run_standard_ppcg(A, b, x0)
            
            # Run non-blocking pipelined CG
            nonblocking_times[i], nonblocking_iters[i] = run_nonblocking_ppcg(A, b, x0)
        end
        
        # Calculate statistics
        std_mean_time = mean(standard_times)
        nb_mean_time = mean(nonblocking_times)
        std_mean_iters = mean(standard_iters)
        nb_mean_iters = mean(nonblocking_iters)
        speedup = std_mean_time / nb_mean_time
        
        # Print results
        @printf("%-10d %-15.6f %-15.6f %-15.6f %-15d %-15d\n", 
                n, std_mean_time, nb_mean_time, speedup, Int(std_mean_iters), Int(nb_mean_iters))
    end
end

# Run timing tests for different grid sizes
grid_sizes = [16, 32, 64, 96, 128]
run_timing_tests(grid_sizes)
