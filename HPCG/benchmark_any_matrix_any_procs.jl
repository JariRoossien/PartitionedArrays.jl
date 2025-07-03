#!/usr/bin/env julia

"""
General CG Benchmark for Any Matrix File - Any Number of Processes
Based on the working benchmark pattern, supports 1, 2, 4, 8, or any number of processes

Usage:
  julia benchmark_any_matrix_any_procs.jl matrix_file.mat                    # Single process
  mpiexec -n 2 julia benchmark_any_matrix_any_procs.jl matrix_file.mat      # 2 processes
  mpiexec -n 4 julia benchmark_any_matrix_any_procs.jl matrix_file.mat      # 4 processes
  mpiexec -n 8 julia benchmark_any_matrix_any_procs.jl matrix_file.mat      # 8 processes

Supported formats:
  - .mat files (MATLAB format)
  - .mtx files (Matrix Market format - placeholder)
  - .jld2 files (Julia format - placeholder)
"""

using Pkg
Pkg.activate("/mnt/d/Users/Jari/Documents/VU/SD/PartitionedArrays.jl")

using LinearAlgebra
using Printf
using Statistics
using PartitionedArrays
using SparseArrays
using Dates
using MPI
import MAT

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
        error("Failed to load matrix file: $e")
    end
end

"""
Load MATLAB .mat file
"""
function load_mat_file(filename::String)
    try
        data = MAT.matread(filename)
        
        # Try to find the matrix in the file
        matrix_key = nothing
        for key in keys(data)
            if isa(data[key], AbstractMatrix)
                matrix_key = key
                break
            end
        end
        
        if matrix_key === nothing
            error("No matrix found in MAT file")
        end
        
        A = data[matrix_key]
        
        # Convert to sparse if not already
        if !isa(A, SparseMatrixCSC)
            A = sparse(A)
        end
        
        # Ensure the matrix is square and symmetric/positive definite for CG
        n, m = size(A)
        if n != m
            error("Matrix must be square for CG solver")
        end
        
        # Create right-hand side vector
        b = ones(n)
        x0 = zeros(n)
        
        return A, b, x0, "$(basename(filename)) ($matrix_key)"
        
    catch e
        error("Failed to load MAT file: $e")
    end
end

"""
Load Matrix Market .mtx file (placeholder)
"""
function load_mtx_file(filename::String)
    error("MTX file support not yet implemented")
end

"""
Load Julia .jld2 file (placeholder)
"""
function load_jld2_file(filename::String)
    error("JLD2 file support not yet implemented")
end

"""
Mock implementations for non-blocking operations
"""
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

# Include the CG implementations
include("src/pipelined_cg_non_blocking.jl")

"""
Simple reference CG implementation for comparison
"""
function reference_cg_solve(x, A, b; tolerance=1e-6, maxiter=1000)
    r = b - A * x
    p = copy(r)
    rsold = dot(r, r)
    res0 = sqrt(rsold)
    
    for i in 1:maxiter
        Ap = A * p
        alpha = rsold / dot(p, Ap)
        x .+= alpha .* p
        r .-= alpha .* Ap
        rsnew = dot(r, r)
        
        if sqrt(rsnew) / res0 < tolerance
            return x, i, sqrt(rsnew), res0
        end
        
        beta = rsnew / rsold
        p .= r .+ beta .* p
        rsold = rsnew
    end
    
    return x, maxiter, sqrt(rsold), res0
end

"""
Benchmark both algorithms
"""
function benchmark_algorithms(A, b, x0, problem_name; tolerance=1e-6, maxiter=1000, runs=3)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    
    if rank == 0
        println("Benchmarking: $problem_name")
        println("  Matrix size: $(size(A)), nnz: $(nnz(A))")
        println("  Processes: $nprocs")
    end
    
    results = Dict()
    
    # Test Reference CG
    if rank == 0
        println("  Running Reference CG...")
    end
    
    ref_times = Float64[]
    ref_iterations = Int[]
    
    for run in 1:runs
        x_copy = copy(x0)
        
        MPI.Barrier(MPI.COMM_WORLD)
        start_time = MPI.Wtime()
        
        try
            x_result, iters, final_res, res0 = reference_cg_solve(x_copy, A, b; tolerance=tolerance, maxiter=maxiter)
            
            MPI.Barrier(MPI.COMM_WORLD)
            end_time = MPI.Wtime()
            
            elapsed = end_time - start_time
            push!(ref_times, elapsed)
            push!(ref_iterations, iters)
            
            if rank == 0
                println("    Run $run: $(elapsed:.4f)s, $iters iterations, residual: $(final_res:.2e)")
            end
            
        catch e
            if rank == 0
                println("    Run $run failed: $e")
            end
        end
    end
    
    # Test Pipelined CG
    if rank == 0
        println("  Running Pipelined CG...")
    end
    
    pip_times = Float64[]
    pip_iterations = Int[]
    
    for run in 1:runs
        x_copy = copy(x0)
        timing_data = [0.0]
        
        MPI.Barrier(MPI.COMM_WORLD)
        start_time = MPI.Wtime()
        
        try
            x_result, timing, res0, final_res, iters = non_blocking_pipelined_cg!(
                x_copy, A, b, timing_data; tolerance=tolerance, maxiter=maxiter)
            
            MPI.Barrier(MPI.COMM_WORLD)
            end_time = MPI.Wtime()
            
            elapsed = end_time - start_time
            push!(pip_times, elapsed)
            push!(pip_iterations, iters)
            
            if rank == 0
                println("    Run $run: $(elapsed:.4f)s, $iters iterations, residual: $(final_res:.2e)")
            end
            
        catch e
            if rank == 0
                println("    Run $run failed: $e")
            end
        end
    end
    
    # Summary
    if rank == 0 && !isempty(ref_times) && !isempty(pip_times)
        ref_avg = mean(ref_times)
        pip_avg = mean(pip_times)
        speedup = ref_avg / pip_avg
        
        println("  Results:")
        println("    Reference CG: $(ref_avg:.4f)s ± $(std(ref_times):.4f)s")
        println("    Pipelined CG: $(pip_avg:.4f)s ± $(std(pip_times):.4f)s")
        println("    Speedup: $(speedup:.2f)x")
        println("    Iterations match: $(abs(mean(ref_iterations) - mean(pip_iterations)) < 0.1)")
        
        results = (
            problem_name = problem_name,
            matrix_size = size(A),
            nnz = nnz(A),
            processes = nprocs,
            ref_time = ref_avg,
            pip_time = pip_avg,
            speedup = speedup,
            ref_iterations = mean(ref_iterations),
            pip_iterations = mean(pip_iterations)
        )
    end
    
    return results
end

"""
Main benchmark function
"""
function run_benchmark()
    MPI.Init()
    
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    
    if rank == 0
        println("=" ^ 60)
        println("CG ALGORITHM BENCHMARK - ANY MATRIX, ANY PROCESSES")
        println("=" ^ 60)
        println("Processes: $nprocs")
        println("Julia version: $(VERSION)")
        println("Start time: $(Dates.now())")
        println("=" ^ 60)
    end
    
    # Get matrix filename from command line
    if length(ARGS) < 1
        if rank == 0
            println("Usage: julia benchmark_any_matrix_any_procs.jl <matrix_file>")
            println("Example: julia benchmark_any_matrix_any_procs.jl 494_bus.mat")
        end
        MPI.Finalize()
        return
    end
    
    matrix_file = ARGS[1]
    
    try
        # Load matrix
        if rank == 0
            println("Loading matrix from: $matrix_file")
        end
        
        A, b, x0, problem_name = load_matrix_file(matrix_file)
        
        # Run benchmark
        results = benchmark_algorithms(A, b, x0, problem_name)
        
        # Save results
        if rank == 0 && !isempty(results)
            timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
            filename = "benchmark_$(splitext(basename(matrix_file))[1])_$(nprocs)procs_$(timestamp).txt"
            
            open(filename, "w") do file
                println(file, "CG Algorithm Benchmark Results")
                println(file, "Generated: $(Dates.now())")
                println(file, "Matrix file: $matrix_file")
                println(file, "Problem: $(results.problem_name)")
                println(file, "Matrix size: $(results.matrix_size)")
                println(file, "Non-zeros: $(results.nnz)")
                println(file, "Processes: $(results.processes)")
                println(file, "")
                println(file, "Results:")
                println(file, "  Reference CG time: $(results.ref_time:.4f)s")
                println(file, "  Pipelined CG time: $(results.pip_time:.4f)s")
                println(file, "  Speedup: $(results.speedup:.2f)x")
                println(file, "  Reference iterations: $(results.ref_iterations:.1f)")
                println(file, "  Pipelined iterations: $(results.pip_iterations:.1f)")
            end
            
            println("Results saved to: $filename")
        end
        
    catch e
        if rank == 0
            println("Benchmark failed: $e")
        end
    end
    
    if rank == 0
        println("\n" * "=" ^ 60)
        println("BENCHMARK COMPLETED")
        println("End time: $(Dates.now())")
        println("=" ^ 60)
    end
    
    MPI.Finalize()
end

# Run the benchmark
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
