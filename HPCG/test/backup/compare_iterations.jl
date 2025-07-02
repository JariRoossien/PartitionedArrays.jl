using LinearAlgebra
using SparseArrays
using Random

# Include only the standard pipelined CG implementation
include("../src/pipelined_cg.jl")

# Mock implementations of non-blocking dot product functions for testing
struct FakeFuture{T}
    result::T
end
Base.fetch(future::FakeFuture) = future.result

function setup_non_blocking_dot(a, b)
    return nothing
end

function non_blocking_dot(a, b, setup)
    result = dot(a, b)
    return FakeFuture(result)
end

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
    
    # Standard pipelined CG from pipelined_cg.jl
    timing_data_std = [0.0]
    x_std, _, res0_std, res_std, iters_std = ref_pipelined_cg!(
        copy(x0), A, b, timing_data_std, tolerance=1e-8, maxiter=1000)
    
    # Read the non-blocking implementation and extract the function
    nb_file = read("/mnt/c/Users/jarir/Documents/VU/SD/PartitionedArrays.jl/HPCG/src/pipelined_cg_non_blocking.jl", String)
    
    # Extract the number of iterations from the non-blocking implementation
    # by running it separately in a new Julia process
    nb_script = """
    using LinearAlgebra
    using SparseArrays
    using Random
    
    # Mock implementations
    struct FakeFuture{T}
        result::T
    end
    Base.fetch(future::FakeFuture) = future.result
    
    function setup_non_blocking_dot(a, b)
        return nothing
    end
    
    function non_blocking_dot(a, b, setup)
        result = dot(a, b)
        return FakeFuture(result)
    end
    
    # Include the non-blocking implementation
    include("/mnt/c/Users/jarir/Documents/VU/SD/PartitionedArrays.jl/HPCG/src/pipelined_cg_non_blocking.jl")
    
    # Create test problem
    n = $n
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
    Random.seed!(42)
    b = rand(n*n)
    x0 = zeros(n*n)
    
    # Run non-blocking pipelined CG
    timing_data_nb = [0.0]
    x_nb, _, res0_nb, res_nb, iters_nb = non_blocking_pipelined_cg!(
        copy(x0), A, b, timing_data_nb, tolerance=1e-8, maxiter=1000)
    
    # Print results
    println("NON_BLOCKING_RESULTS:")
    println(iters_nb)
    println(res0_nb)
    println(res_nb)
    println(norm(x_nb))
    """
    
    # Write the script to a temporary file
    open("/tmp/nb_script.jl", "w") do f
        write(f, nb_script)
    end
    
    # Run the script and capture the output
    nb_output = read(`/home/dizmizzer/.julia/juliaup/julia-1.11.5+0.x64.linux.gnu/bin/julia /tmp/nb_script.jl`, String)
    
    # Parse the output
    lines = split(nb_output, '\n')
    start_idx = findfirst(l -> l == "NON_BLOCKING_RESULTS:", lines)
    if start_idx !== nothing
        iters_nb = parse(Int, lines[start_idx + 1])
        res0_nb = parse(Float64, lines[start_idx + 2])
        res_nb = parse(Float64, lines[start_idx + 3])
        norm_x_nb = parse(Float64, lines[start_idx + 4])
        
        # Compare results
        println("Standard Pipelined CG:")
        println("  Iterations: $iters_std")
        println("  Initial residual: $res0_std")
        println("  Final residual: $res_std")
        println("  Solution norm: $(norm(x_std))")
        
        println("\nNon-blocking Pipelined CG:")
        println("  Iterations: $iters_nb")
        println("  Initial residual: $res0_nb")
        println("  Final residual: $res_nb")
        println("  Solution norm: $norm_x_nb")
        
        # Check if the iterations are the same
        if iters_std == iters_nb
            println("\nBoth implementations take the same number of iterations!")
        else
            println("\nThe implementations take different numbers of iterations.")
            println("Difference: $(iters_nb - iters_std) iterations")
        end
        
        return iters_std, iters_nb
    else
        println("Failed to parse non-blocking results")
        return iters_std, nothing
    end
end

# Run the comparison with a 64x64 grid (4096x4096 matrix)
compare_iterations(64)
