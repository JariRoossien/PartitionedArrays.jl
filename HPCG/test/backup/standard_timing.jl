using LinearAlgebra
using SparseArrays
using Random

# Override the dot product to add delay
const DOT_PRODUCT_DELAY = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.001

function delayed_dot(a, b)
    result = dot(a, b)
    sleep(DOT_PRODUCT_DELAY)  # Add delay to simulate network latency
    return result
end

# Include the standard pipelined CG implementation
include("../src/pipelined_cg.jl")

# Override the dot function to use our delayed version
import LinearAlgebra: dot
dot(a::AbstractVector, b::AbstractVector) = delayed_dot(a, b)

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

# Parse command line arguments
n = parse(Int, ARGS[1])
num_runs = parse(Int, ARGS[2])

println("Using dot product delay of $(DOT_PRODUCT_DELAY*1000) ms")

# Create test problem
A, b, x0 = create_test_problem(n)

# Run standard pipelined CG multiple times
times = Float64[]
iters = Int[]

for i in 1:num_runs
    timing_data = [0.0]
    start_time = time()
    _, _, _, _, iter = ref_pipelined_cg!(
        copy(x0), A, b, timing_data, tolerance=1e-8, maxiter=1000)
    end_time = time()
    
    push!(times, end_time - start_time)
    push!(iters, iter)
end

# Print results
println("STANDARD_RESULTS:")
println(join(times, ","))
println(join(iters, ","))
