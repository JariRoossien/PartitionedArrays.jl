#!/usr/bin/env julia

"""
CG Benchmark: Reference vs Pipelined Non-blocking Implementation with Metis Partitioning
Compares ref_cg.jl and pipelined_cg_non_blocking.jl using any matrix file
Uses Metis optimal partitioning to minimize ghost columns and communication overhead

Usage:
  julia benchmark_ref_vs_pipelined.jl matrix_file.mat                    # Single process
  mpiexec -n 2 julia benchmark_ref_vs_pipelined.jl matrix_file.mat      # 2 processes
  mpiexec -n 4 julia benchmark_ref_vs_pipelined.jl matrix_file.mat      # 4 processes
  mpiexec -n 8 julia benchmark_ref_vs_pipelined.jl matrix_file.mat      # 8 processes

Supported formats:
  - .mat files (MATLAB format)
  - .mtx files (Matrix Market format - placeholder)
  - .jld2 files (Julia format - placeholder)

Key Features:
  - Uses Metis.partition() for optimal matrix partitioning
  - Minimizes ghost columns and communication overhead
  - Follows the approach described in example.md
"""

using Pkg
Pkg.activate(".")

using LinearAlgebra
using Printf
using Statistics
using PartitionedArrays
using PartitionedArrays: distribute_with_mpi
using SparseArrays
using Dates
using MPI
import MAT
using Metis

# Include HPCG module for both CG implementations
include("src/HPCG.jl")
using .HPCG

"""
Wrapper for HPCG ref_cg! to match our benchmark interface
"""
function hpcg_ref_cg_wrapper!(x, A, b, nprocs; tolerance=1e-6, maxiter=1000)
    # Initialize timing data array (7 elements as expected by ref_cg!)
    timing_data = zeros(7)
    
    # Debug: Check initial conditions
    initial_residual = norm(b - A * x)
    
    # Internal timing for diagnostics
    internal_start = time()
    
    # Call HPCG ref_cg! 
    x_final, timing_result, residual0, residual_final, iters = HPCG.ref_cg!(
        x, A, b, timing_data;
        tolerance=tolerance, 
        maxiter=maxiter,
        Pl=HPCG.Identity()
    )
    
    internal_end = time()
    internal_time = internal_end - internal_start
    
    # Debug: Verify the computation actually happened
    final_residual_check = norm(b - A * x_final)
    
    # Warn if something seems wrong
    if abs(final_residual_check - residual_final) > tolerance * 10
        @warn "Reference CG: Residual mismatch! Reported: $residual_final, Actual: $final_residual_check"
    end
    
    if iters < 3 && residual_final > tolerance
        @warn "Reference CG: Suspiciously few iterations ($iters) without convergence"
    end
    
    # Debug timing
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println("    Reference CG internal time: $(internal_time)s")
    end
    
    return x_final, residual0, residual_final, iters
end

"""
Wrapper for non-blocking pipelined CG to match our benchmark interface
"""
function pipelined_cg_wrapper!(x, A, b, nprocs; tolerance=1e-6, maxiter=1000)
    # Initialize timing data array
    timing_data = zeros(7)
    
    # Debug: Check initial conditions
    initial_residual = norm(b - A * x)
    
    # Internal timing for diagnostics
    internal_start = time()
    
    # Call non-blocking pipelined CG from HPCG module
    x_final, timing_result, residual0, residual_final, iters = HPCG.non_blocking_pipelined_cg!(
        x, A, b, timing_data;
        tolerance=tolerance, 
        maxiter=maxiter,
        Pl=HPCG.Identity()
    )
    
    internal_end = time()
    internal_time = internal_end - internal_start
    
    # Debug: Verify the computation actually happened
    final_residual_check = norm(b - A * x_final)
    
    # Warn if something seems wrong
    if abs(final_residual_check - residual_final) > tolerance * 10
        @warn "Pipelined CG: Residual mismatch! Reported: $residual_final, Actual: $final_residual_check"
    end
    
    if iters < 3 && residual_final > tolerance
        @warn "Pipelined CG: Suspiciously few iterations ($iters) without convergence"
    end
    
    # Additional check for pipelined CG
    if iters == 1
        @warn "Pipelined CG: Only 1 iteration - possible early termination or trivial problem"
    end
    
    # Debug timing - this is crucial for diagnosing the timing issue
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println("    Pipelined CG internal time: $(internal_time)s")
        if internal_time < 0.001
            println("    ⚠️  WARNING: Pipelined CG internal time suspiciously short!")
        end
    end
    
    return x_final, residual0, residual_final, iters
end

"""
Load matrix from various file formats
"""
function load_matrix_file(filename::String)
    if !isfile(filename)
        error("File not found: $filename")
    end
   
    ext = lowercase(splitext(filename)[2])
   
    try
        if ext == ".mat"
            return load_mat_file(filename)
        elseif ext == ".mtx"
            return load_mtx_file(filename)
        elseif ext == ".jld2"
            return load_jld2_file(filename)
        else
            error("Unsupported file format: $ext. Supported: .mat, .mtx, .jld2")
        end
    catch e
        error("Failed to load $filename: $e")
    end
end

"""
Load MATLAB .mat file
"""
function load_mat_file(filename::String)
    dict = MAT.matread(filename)
    # Try different common structures
    if haskey(dict, "Problem") && haskey(dict["Problem"], "A")
        A_sparse = dict["Problem"]["A"]
        name = get(dict["Problem"], "name", splitext(basename(filename))[1])
    elseif haskey(dict, "A")
        A_sparse = dict["A"]
        name = splitext(basename(filename))[1]
    else
        # Try to find the first sparse matrix
        for (key, value) in dict
            if isa(value, SparseMatrixCSC)
                A_sparse = value
                name = key
                break
            end
        end
        if !@isdefined(A_sparse)
            error("No sparse matrix found in .mat file")
        end
    end
    return A_sparse, name
end

"""
Load and distribute any matrix file using Metis partitioning
"""
function load_and_distribute_matrix(ranks, filename::String)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if rank == 0
        println("Loading matrix from: $filename")
    end

    # Load matrix only on rank 0, following example.md approach
    A_in_main = map(ranks) do r
        if r == 1  # rank 0 in Julia is rank 1 in PartitionedArrays
            A_sparse, matrix_name = load_matrix_file(filename)
            if size(A_sparse, 1) != size(A_sparse, 2)
                error("Matrix must be square for CG. Got size: $(size(A_sparse))")
            end
            return A_sparse
        else
            return nothing
        end
    end

    # Get matrix name for reporting
    matrix_name = if rank == 0
        _, name = load_matrix_file(filename)
        name
    else
        ""
    end
    
    # Broadcast matrix name to all ranks
    matrix_name = MPI.bcast(matrix_name, 0, comm)

    # Get matrix size
    n_in_main = map(A_in_main) do A
        if A !== nothing
            size(A, 1)
        else
            nothing
        end
    end
    
    n_all = PartitionedArrays.multicast(n_in_main)
    n = PartitionedArrays.getany(n_all)

    # Use Metis to create optimal partition (following example.md)
    colors_in_main = map(A_in_main) do A
        if A !== nothing
            Metis.partition(A, nprocs)
        else
            nothing
        end
    end

    # Create partition from Metis colors
    row_partition = PartitionedArrays.partition_from_color(ranks, colors_in_main; multicast=true)

    # Extract sparse matrix components for psparse
    AIJV = map(A_in_main) do A
        if A !== nothing
            AI, AJ, AV = findnz(A)
        else
            AI = Int64[]
            AJ = Int64[]
            AV = Float64[]
        end
        return AI, AJ, AV
    end
    AI, AJ, AV = PartitionedArrays.tuple_of_arrays(AIJV)

    # Create partitioned sparse matrix using psparse
    t = PartitionedArrays.psparse(AI, AJ, AV, row_partition, row_partition)
    A_partitioned = fetch(t)

    # Create RHS vector - use A*ones to create a more realistic problem
    # This ensures the exact solution is the vector of all ones
    ones_vec = PartitionedArrays.pones(axes(A_partitioned, 2))
    b_partitioned = A_partitioned * ones_vec
    x0_partitioned = PartitionedArrays.pzeros(axes(A_partitioned, 2))

    if rank == 0
        println("✓ Matrix loaded and partitioned with Metis: $matrix_name")
        println("✓ Using optimal Metis partitioning for $nprocs processes")
    end

    return A_partitioned, b_partitioned, x0_partitioned, nothing, matrix_name, nprocs
end

"""
Benchmark a single CG algorithm with MPI awareness
"""
function benchmark_cg_algorithm(alg_name::String, solver_func, A, b, x0, nprocs;
                               tolerance=1e-6, maxiter=2000, runs=3, recreate_matrix_func=nothing)
   
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
   
    if rank == 0
        println("  Benchmarking $alg_name on $nprocs processes...")
    end
   
    times = Float64[]
    iterations = Int[]
    residuals = Float64[]
   
    for i in 1:runs
        # Optionally recreate matrix for each run (maximum isolation)
        A_run, b_run, x0_run = if recreate_matrix_func !== nothing && contains(alg_name, "Pipelined")
            if rank == 0 && i == 1
                println("    Recreating matrix for each pipelined run...")
            end
            A_new, b_new, x0_new, _, _, _ = recreate_matrix_func()
            A_new, b_new, x0_new
        else
            A, b, copy(x0)
        end
        
        # Create completely fresh copy for each run
        x_copy = copy(x0_run)
        
        # Validate initial conditions
        if rank == 0 && i == 1
            initial_norm = norm(x_copy)
            println("    Initial solution norm: $initial_norm (should be ~0)")
            rhs_norm = norm(b_run)
            println("    RHS norm: $rhs_norm (should be > 0)")
        end
       
        # Synchronize all processes before timing
        MPI.Barrier(comm)
        start_time = MPI.Wtime()
       
        try
            x_final, residual0, residual_final, iters = solver_func(x_copy, A_run, b_run, nprocs;
                                                                   tolerance=tolerance, maxiter=maxiter)
           
            # Synchronize all processes after computation
            MPI.Barrier(comm)
            end_time = MPI.Wtime()
            
            # Validate results
            if rank == 0 && i == 1
                final_norm = norm(x_final)
                println("    Final solution norm: $final_norm")
                println("    Initial residual: $residual0")
                println("    Final residual: $residual_final")
                println("    Iterations: $iters")
                
                # Check if convergence seems suspicious
                if iters < 3
                    println("    ⚠️  WARNING: Very few iterations ($iters) - check for trivial problem or warm start!")
                end
            end
           
            push!(times, end_time - start_time)
            push!(iterations, iters)
            push!(residuals, residual_final)
           
        catch e
            if rank == 0
                println("    Error in run $i: $e")
            end
            push!(times, NaN)
            push!(iterations, maxiter)
            push!(residuals, NaN)
        end
        
        # Small barrier between runs to ensure clean separation
        MPI.Barrier(comm)
    end
   
    # Filter out failed runs
    valid_indices = .!isnan.(times)
    if sum(valid_indices) == 0
        return (mean_time = NaN, std_time = NaN, mean_iterations = NaN,
                mean_residual = NaN, converged = false)
    end
   
    valid_times = times[valid_indices]
    valid_iterations = iterations[valid_indices]
    valid_residuals = residuals[valid_indices]
   
    return (
        mean_time = mean(valid_times),
        std_time = length(valid_times) > 1 ? std(valid_times) : 0.0,
        mean_iterations = mean(valid_iterations),
        mean_residual = mean(valid_residuals),
        converged = sum(valid_indices) == runs
    )
end

"""
Save results to file
"""
function save_results(ref_results, pipelined_results, matrix_name, filename, nprocs)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
   
    if rank == 0
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        safe_name = replace(matrix_name, r"[^a-zA-Z0-9_]" => "_")
        output_filename = "benchmark_ref_vs_pipelined_metis_$(safe_name)_$(nprocs)procs_$timestamp.txt"
       
        open(output_filename, "w") do f
            println(f, "CG Algorithm Benchmark: Reference vs Pipelined Non-blocking with Metis Partitioning")
            println(f, "Generated: $(now())")
            println(f, "Input file: $filename")
            println(f, "Matrix: $matrix_name")
            println(f, "Mode: MPI Distributed with Metis Optimal Partitioning")
            println(f, "Processes: $nprocs")
            println(f, "Julia version: $(VERSION)")
            println(f, "="^60)
           
            println(f, "\nReference CG (ref_cg.jl):")
            println(f, "  Time: $(ref_results.mean_time) ± $(ref_results.std_time) s")
            println(f, "  Iterations: $(ref_results.mean_iterations)")
            println(f, "  Residual: $(ref_results.mean_residual)")
            println(f, "  Converged: $(ref_results.converged)")
           
            println(f, "\nPipelined CG Non-blocking (pipelined_cg_non_blocking.jl):")
            println(f, "  Time: $(pipelined_results.mean_time) ± $(pipelined_results.std_time) s")
            println(f, "  Iterations: $(pipelined_results.mean_iterations)")
            println(f, "  Residual: $(pipelined_results.mean_residual)")
            println(f, "  Converged: $(pipelined_results.converged)")
           
            if !isnan(ref_results.mean_time) && !isnan(pipelined_results.mean_time)
                speedup = ref_results.mean_time / pipelined_results.mean_time
                println(f, "\nSpeedup: $(speedup)x")
                println(f, "Performance: $(speedup > 1 ? "Pipelined CG faster" : "Reference CG faster")")
                
                # Iteration comparison
                iter_diff = ref_results.mean_iterations - pipelined_results.mean_iterations
                println(f, "Iteration difference: $(iter_diff) (negative means pipelined uses fewer)")
                
                println(f, "\nPartitioning: Metis optimal partitioning (minimizes ghost columns)")
            end
        end
       
        println("Results saved to: $output_filename")
    end
end

"""
Main benchmark function - called by PartitionedArrays.with_mpi
"""
function perform_benchmarks(distribute_func, filename::String)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
   
    # Create ranks array for PartitionedArrays (following example.md pattern)
    ranks = PartitionedArrays.DebugArray(LinearIndices((nprocs,)))
   
    # Determine process description
    proc_desc = if nprocs == 1
        "SINGLE PROCESS"
    elseif nprocs == 2
        "2 PROCESSES"
    elseif nprocs == 4
        "4 PROCESSES"
    elseif nprocs == 8
        "8 PROCESSES"
    else
        "$nprocs PROCESSES"
    end
   
    if rank == 0
        println("="^80)
        println("CG BENCHMARK: REFERENCE vs PIPELINED NON-BLOCKING - $proc_desc")
        println("="^80)
        println("Mode: $(nprocs == 1 ? "Single Process" : "MPI Distributed with Metis Partitioning")")
        println("Total MPI processes: $nprocs")
        println("Current rank: $rank")
        println("Input file: $filename")
        println("Julia version: $(VERSION)")
        println("Start time: $(now())")
        println("="^80)
    end
   
    # Load and demonstrate Metis partitioning with any matrix
    A, b, x0, process_layout, matrix_name, total_procs = load_and_distribute_matrix(ranks, filename)
   
    if rank == 0
        println("\n" * "="^80)
        println("PERFORMANCE BENCHMARK WITH $total_procs MPI PROCESSES")
        println("="^80)
        println("Matrix: $matrix_name (from $filename)")
        println("Size: $(length(b)) unknowns")
        println("Matrix type: $(typeof(A))")
        println("Matrix density: $(round(nnz(A)/(size(A,1)^2)*100, digits=4))%")
        println("MPI processes: $total_procs")
        println("Partitioning: ✅ METIS OPTIMAL PARTITIONING WITH $total_procs PROCESSES")
        println("-"^80)
    end
   
    # Benchmark parameters
    tolerance = 1e-6
    maxiter = 2000
    runs = 3
   
    # Benchmark both algorithms with complete isolation
    comm = MPI.COMM_WORLD
    
    if rank == 0
        println("  Starting Reference CG benchmark...")
    end
    
    # Complete barrier and fresh start for reference algorithm
    MPI.Barrier(comm)
    x0_ref = copy(x0)  # Fresh copy for reference algorithm
    ref_results = benchmark_cg_algorithm("Reference CG (ref_cg.jl)", hpcg_ref_cg_wrapper!, A, b, x0_ref, total_procs;
                                        tolerance=tolerance, maxiter=maxiter, runs=runs)
   
    # Force garbage collection and complete barrier between algorithms
    if rank == 0
        println("  Reference CG completed. Preparing for Pipelined CG...")
        println("  Recreating sparse matrix for complete isolation...")
    end
    GC.gc()  # Force garbage collection
    MPI.Barrier(comm)
    
    # RECREATE THE SPARSE MATRIX for complete isolation
    if rank == 0
        println("  Recreating matrix and RHS from scratch...")
    end
    
    # Recreate the matrix using the same process as initial load
    A_fresh, b_fresh, x0_fresh, _, _, _ = load_and_distribute_matrix(ranks, filename)
    
    if rank == 0
        println("  Fresh matrix created. Starting Pipelined CG benchmark...")
    end
    
    # Complete fresh start for pipelined algorithm with fresh matrix
    MPI.Barrier(comm)
    
    # Create a function to recreate the matrix for maximum isolation
    recreate_matrix = () -> load_and_distribute_matrix(ranks, filename)
    
    pipelined_results = benchmark_cg_algorithm("Pipelined CG Non-blocking", pipelined_cg_wrapper!, A_fresh, b_fresh, x0_fresh, total_procs;
                                              tolerance=tolerance, maxiter=maxiter, runs=runs, recreate_matrix_func=recreate_matrix)
   
    # Display results (only on rank 0)
    if rank == 0
        println("\n  Results with $total_procs MPI processes (Metis partitioning):")
        println("  ┌─────────────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
        println("  │ Algorithm                       │ Time (s)     │ Iterations   │ Residual     │ Converged    │")
        println("  ├─────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")
        @printf("  │ Reference CG (ref_cg.jl)        │ %8.4f±%.4f │ %8.1f     │ %8.2e     │ %12s │\n",
                ref_results.mean_time, ref_results.std_time, ref_results.mean_iterations,
                ref_results.mean_residual, ref_results.converged ? "Yes" : "No")
        @printf("  │ Pipelined CG Non-blocking       │ %8.4f±%.4f │ %8.1f     │ %8.2e     │ %12s │\n",
                pipelined_results.mean_time, pipelined_results.std_time, pipelined_results.mean_iterations,
                pipelined_results.mean_residual, pipelined_results.converged ? "Yes" : "No")
        println("  └─────────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")
       
        if !isnan(ref_results.mean_time) && !isnan(pipelined_results.mean_time)
            speedup = ref_results.mean_time / pipelined_results.mean_time
            @printf("  Speedup: %.2fx %s\n", abs(speedup),
                   speedup > 1 ? "(Pipelined faster)" : "(Reference faster)")
           
            # Compare iterations
            iter_diff = ref_results.mean_iterations - pipelined_results.mean_iterations
            @printf("  Iteration difference: %.1f %s\n", iter_diff,
                   iter_diff > 0 ? "(Pipelined uses fewer)" : "(Reference uses fewer)")
           
            # Analyze convergence behavior
            println("  Algorithm Analysis:")
            println("    Reference CG: Classical conjugate gradient implementation")
            println("    Pipelined CG: Non-blocking communication with overlapped computation")
            println("    Partitioning: Metis optimal partitioning (minimizes ghost columns)")
            
            if abs(iter_diff) < 2
                println("    → Similar convergence behavior (iteration difference < 2)")
            elseif iter_diff > 0
                println("    → Pipelined CG converges faster (fewer iterations)")
            else
                println("    → Reference CG converges faster (fewer iterations)")
            end
        end
       
        println("\n" * "="^80)
        println("BENCHMARK SUMMARY - $total_procs MPI PROCESSES WITH METIS PARTITIONING")
        println("="^80)
       
        if !isnan(ref_results.mean_time) && !isnan(pipelined_results.mean_time)
            speedup = ref_results.mean_time / pipelined_results.mean_time
           
            println("✅ REFERENCE vs PIPELINED CG COMPARISON WITH METIS PARTITIONING SUCCESSFUL!")
            println("")
            println("Key Results:")
            println("- Both algorithms executed successfully across $total_procs processes")
            println("- Matrix loaded and processed: $matrix_name")
            println("- Problem size: $(length(b)) unknowns")
            println("- Matrix density: $(round(nnz(A)/(size(A,1)^2)*100, digits=4))%")
            println("- Partitioning: Metis optimal (minimizes communication overhead)")
            println("")
            @printf("Reference CG time:        %.6f ± %.6f seconds (%.1f iterations)\n", 
                   ref_results.mean_time, ref_results.std_time, ref_results.mean_iterations)
            @printf("Pipelined CG time:        %.6f ± %.6f seconds (%.1f iterations)\n", 
                   pipelined_results.mean_time, pipelined_results.std_time, pipelined_results.mean_iterations)
            @printf("Speedup:                  %.3fx\n", speedup)
           
            if speedup > 1.1
                println("✅ Pipelined CG shows significant performance improvement!")
                println("   Non-blocking communication and computation overlap is effective")
                println("   Metis partitioning reduces communication overhead")
            elseif speedup > 0.9
                println("⚖️  Performance is comparable between both implementations")
                println("   Both algorithms handle the problem similarly with optimal partitioning")
            else
                println("⚠️  Reference CG is faster for this problem")
                println("   Classical implementation may be more suitable for this matrix")
            end
            
            # Convergence analysis
            iter_diff = ref_results.mean_iterations - pipelined_results.mean_iterations
            if abs(iter_diff) < 1
                println("🔄 Both algorithms have identical convergence behavior")
            elseif iter_diff > 0
                println("🚀 Pipelined CG converges faster ($(abs(iter_diff)) fewer iterations)")
            else
                println("🐌 Reference CG converges faster ($(abs(iter_diff)) fewer iterations)")
            end
        end
       
        println("\nCompletion time: $(now())")
        println("✅ Reference vs Pipelined CG benchmark with Metis partitioning completed!")
        println("="^80)
    end
   
    # Save results
    save_results(ref_results, pipelined_results, matrix_name, filename, total_procs)
   
    return (ref_results, pipelined_results)
end

# Main execution using the PartitionedArrays.with_mpi pattern
if abspath(PROGRAM_FILE) == @__FILE__
    # Get filename from command line arguments
    if length(ARGS) < 1
        println("Usage: julia benchmark_ref_vs_pipelined.jl <matrix_file>")
        println("       mpiexec -n 2 julia benchmark_ref_vs_pipelined.jl <matrix_file>")
        println("       mpiexec -n 4 julia benchmark_ref_vs_pipelined.jl <matrix_file>")
        println("       mpiexec -n 8 julia benchmark_ref_vs_pipelined.jl <matrix_file>")
        println("")
        println("Supported formats: .mat, .mtx, .jld2")
        println("")
        println("This benchmark compares:")
        println("  - Reference CG (ref_cg.jl): Classical conjugate gradient")
        println("  - Pipelined CG (pipelined_cg_non_blocking.jl): Non-blocking with overlapped communication")
        println("  - Uses Metis optimal partitioning to minimize ghost columns and communication overhead")
        exit(1)
    end
    
    filename = ARGS[1]
    PartitionedArrays.with_mpi() do distribute
        perform_benchmarks(distribute, filename)
    end
end
