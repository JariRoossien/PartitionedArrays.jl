using LinearAlgebra

# Function to compare iterations between implementations
function compare_iterations(n::Int)
    println("Comparing iterations with $(n)x$(n) grid ($(n*n)x$(n*n) matrix):")
    
    # Run standard pipelined CG in a separate process
    std_output = read(`/home/dizmizzer/.julia/juliaup/julia-1.11.5+0.x64.linux.gnu/bin/julia --project=. HPCG/test/run_standard.jl $n`, String)
    
    # Run non-blocking pipelined CG in a separate process
    nb_output = read(`/home/dizmizzer/.julia/juliaup/julia-1.11.5+0.x64.linux.gnu/bin/julia --project=. HPCG/test/run_nonblocking.jl $n`, String)
    
    # Parse standard results
    std_lines = split(std_output, '\n')
    std_start_idx = findfirst(l -> l == "STANDARD_RESULTS:", std_lines)
    if std_start_idx !== nothing
        iters_std = parse(Int, std_lines[std_start_idx + 1])
        res0_std = parse(Float64, std_lines[std_start_idx + 2])
        res_std = parse(Float64, std_lines[std_start_idx + 3])
        norm_x_std = parse(Float64, std_lines[std_start_idx + 4])
    else
        error("Failed to parse standard results")
    end
    
    # Parse non-blocking results
    nb_lines = split(nb_output, '\n')
    nb_start_idx = findfirst(l -> l == "NONBLOCKING_RESULTS:", nb_lines)
    if nb_start_idx !== nothing
        iters_nb = parse(Int, nb_lines[nb_start_idx + 1])
        res0_nb = parse(Float64, nb_lines[nb_start_idx + 2])
        res_nb = parse(Float64, nb_lines[nb_start_idx + 3])
        norm_x_nb = parse(Float64, nb_lines[nb_start_idx + 4])
    else
        error("Failed to parse non-blocking results")
    end
    
    # Compare results
    println("Standard Pipelined CG:")
    println("  Iterations: $iters_std")
    println("  Initial residual: $res0_std")
    println("  Final residual: $res_std")
    println("  Solution norm: $norm_x_std")
    
    println("\nFixed Non-blocking Pipelined CG:")
    println("  Iterations: $iters_nb")
    println("  Initial residual: $res0_nb")
    println("  Final residual: $res_nb")
    println("  Solution norm: $norm_x_nb")
    
    # Compare solutions (approximately)
    solution_diff_ratio = abs(norm_x_std - norm_x_nb) / norm_x_std
    println("\nSolution norm difference (relative): $solution_diff_ratio")
    
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
    
    return iters_std, iters_nb, solution_diff_ratio, residual_diff
end

# Run the comparison with a 64x64 grid (4096x4096 matrix)
compare_iterations(64)
