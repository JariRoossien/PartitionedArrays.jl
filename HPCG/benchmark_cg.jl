#!/usr/bin/env julia

"""
Clean benchmark comparing Reference CG vs Pipelined CG algorithms
using PartitionedArrays for distributed computing.

Usage:
  julia benchmark_cg.jl                    # Single-threaded benchmark
  mpiexec -n 4 julia benchmark_cg.jl      # MPI distributed benchmark
"""

using Pkg
Pkg.activate("..")

using LinearAlgebra
using Printf
using Statistics
using PartitionedArrays
using SparseArrays
using Dates

# Check if MPI is available and initialize if needed
const USE_MPI = try
    using MPI
    MPI.Init()
    true
catch
    false
end

# Mock implementations for non-blocking operations
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
include("src/ref_cg.jl")
include("src/pipelined_cg_non_blocking.jl")

# Define mul_no_lat! for ref_cg compatibility
function mul_no_lat!(y, A, x)
    mul!(y, A, x)
end

"""
Create test problems - works with both single-threaded and MPI backends
"""
function create_test_problem(backend, problem_size::Int)
    if USE_MPI
        # MPI distributed problem
        nprocs = length(backend)
        parts_per_dir = (nprocs,)
        nodes_per_dir = (problem_size,)
        args = laplacian_fdm(nodes_per_dir, parts_per_dir, backend)
        A = psparse(args...) |> fetch
        b = pones(axes(A, 2))
        x0 = pzeros(axes(A, 2))
    else
        # Single-threaded problem using DebugArray
        parts_per_dir = (length(backend),)
        nodes_per_dir = (problem_size,)
        args = laplacian_fdm(nodes_per_dir, parts_per_dir, backend)
        A = psparse(args...) |> fetch
        b = pones(axes(A, 2))
        x0 = pzeros(axes(A, 2))
    end
    
    return A, b, x0
end

"""
Simple reference CG implementation for comparison
"""
function simple_reference_cg(x, A, b; tolerance=1e-6, maxiter=1000)
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
Benchmark a single algorithm
"""
function benchmark_algorithm(alg_name::String, solver_func, A, b, x0; 
                            tolerance=1e-6, maxiter=1000, runs=3)
    
    rank = USE_MPI ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
    
    if rank == 0
        println("  Benchmarking $alg_name...")
    end
    
    times = Float64[]
    iterations = Int[]
    residuals = Float64[]
    
    for i in 1:runs
        x_copy = copy(x0)
        
        # Synchronize if using MPI
        if USE_MPI
            MPI.Barrier(MPI.COMM_WORLD)
            start_time = MPI.Wtime()
        else
            start_time = time()
        end
        
        try
            if alg_name == "Pipelined CG"
                timing_data = [0.0]
                x_result, timing, res0, final_residual, iters = solver_func(x_copy, A, b, timing_data; tolerance=tolerance, maxiter=maxiter)
            else
                x_result, iters, final_residual, res0 = solver_func(x_copy, A, b; tolerance=tolerance, maxiter=maxiter)
            end
            
            if USE_MPI
                MPI.Barrier(MPI.COMM_WORLD)
                end_time = MPI.Wtime()
            else
                end_time = time()
            end
            
            elapsed_time = end_time - start_time
            push!(times, elapsed_time)
            push!(iterations, iters)
            push!(residuals, final_residual)
            
            if rank == 0
                println("    Run $i: $(elapsed_time:.4f)s, $iters iterations, residual: $(final_residual:.2e)")
            end
            
        catch e
            if rank == 0
                println("    Run $i failed: $e")
            end
            return nothing
        end
    end
    
    if rank == 0 && !isempty(times)
        avg_time = mean(times)
        std_time = std(times)
        avg_iters = mean(iterations)
        avg_residual = mean(residuals)
        
        println("    Average: $(avg_time:.4f)s ± $(std_time:.4f)s")
        println("    Iterations: $(avg_iters:.1f), Final residual: $(avg_residual:.2e)")
    end
    
    return (times=times, iterations=iterations, residuals=residuals)
end

"""
Main benchmark function
"""
function run_benchmark()
    rank = USE_MPI ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
    nprocs = USE_MPI ? MPI.Comm_size(MPI.COMM_WORLD) : 1
    
    if rank == 0
        println("=" ^ 60)
        println("CG ALGORITHM BENCHMARK")
        println("=" ^ 60)
        println("Mode: $(USE_MPI ? "MPI Distributed" : "Single-threaded")")
        println("Processes: $nprocs")
        println("Julia version: $(VERSION)")
        println("Start time: $(Dates.now())")
        println("=" ^ 60)
    end
    
    # Set up backend
    if USE_MPI
        backend = with_mpi() do distribute
            distribute(LinearIndices((nprocs,)))
        end
    else
        backend = DebugArray(1)
    end
    
    # Test problem sizes
    problem_sizes = [16, 64, 256]
    
    results = Dict()
    
    for problem_size in problem_sizes
        if rank == 0
            println("\nTesting problem size: $problem_size")
            println("-" ^ 40)
        end
        
        try
            # Create test problem
            A, b, x0 = create_test_problem(backend, problem_size)
            
            if rank == 0
                println("Matrix size: $(size(A))")
                println("Matrix type: $(typeof(A))")
            end
            
            # Test algorithms
            ref_result = benchmark_algorithm("Reference CG", simple_reference_cg, A, b, x0)
            pip_result = benchmark_algorithm("Pipelined CG", non_blocking_pipelined_cg!, A, b, x0)
            
            if rank == 0 && ref_result !== nothing && pip_result !== nothing
                println("✓ Both algorithms completed successfully")
                
                # Compare performance
                ref_avg_time = mean(ref_result.times)
                pip_avg_time = mean(pip_result.times)
                speedup = ref_avg_time / pip_avg_time
                
                # Compare iterations (should be identical)
                ref_avg_iters = mean(ref_result.iterations)
                pip_avg_iters = mean(pip_result.iterations)
                
                println("Performance comparison:")
                println("  Reference CG: $(ref_avg_time:.4f)s, $(ref_avg_iters:.1f) iterations")
                println("  Pipelined CG: $(pip_avg_time:.4f)s, $(pip_avg_iters:.1f) iterations")
                println("  Speedup: $(speedup:.2f)x $(speedup > 1.0 ? "(Pipelined faster)" : "(Reference faster)")")
                
                if abs(ref_avg_iters - pip_avg_iters) < 0.1
                    println("  ✓ Iteration counts match (algorithms are equivalent)")
                else
                    println("  ⚠ Iteration counts differ (potential algorithm difference)")
                end
                
                results[problem_size] = (ref=ref_result, pip=pip_result, speedup=speedup)
            else
                if rank == 0
                    println("✗ One or both algorithms failed")
                end
            end
            
        catch e
            if rank == 0
                println("✗ Problem size $problem_size failed: $e")
            end
        end
    end
    
    # Summary
    if rank == 0
        println("\n" * "=" ^ 60)
        println("BENCHMARK SUMMARY")
        println("=" ^ 60)
        
        if !isempty(results)
            println("Problem Size | Ref Time | Pip Time | Speedup | Iterations")
            println("-" ^ 55)
            for (size, result) in sort(collect(results))
                ref_time = mean(result.ref.times)
                pip_time = mean(result.pip.times)
                speedup = result.speedup
                iters = mean(result.ref.iterations)
                @printf("%11d | %8.4f | %8.4f | %7.2fx | %10.1f\n", size, ref_time, pip_time, speedup, iters)
            end
            
            println("\nKey Insights:")
            println("- Both algorithms should converge in the same number of iterations")
            println("- Performance differences depend on problem size and system characteristics")
            println("- Pipelined CG is designed to overlap communication with computation in MPI settings")
            
            # Save results
            timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
            filename = "cg_benchmark_mpi_$(nprocs)procs_$(timestamp).txt"
            
            open(filename, "w") do file
                println(file, "CG Algorithm Benchmark Results")
                println(file, "Generated: $(Dates.now())")
                println(file, "Mode: $(USE_MPI ? "MPI Distributed" : "Single-threaded")")
                println(file, "Processes: $nprocs")
                println(file, "Julia version: $(VERSION)")
                println(file, "")
                for (size, result) in sort(collect(results))
                    println(file, "Problem size: $size")
                    println(file, "  Reference CG: $(mean(result.ref.times):.4f)s, $(mean(result.ref.iterations):.1f) iterations")
                    println(file, "  Pipelined CG: $(mean(result.pip.times):.4f)s, $(mean(result.pip.iterations):.1f) iterations")
                    println(file, "  Speedup: $(result.speedup:.2f)x")
                    println(file, "")
                end
            end
            
            println("Results saved to: $filename")
        else
            println("No successful benchmarks completed.")
        end
        
        println("\nEnd time: $(Dates.now())")
        println("=" ^ 60)
    end
end

# Run the benchmark
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
