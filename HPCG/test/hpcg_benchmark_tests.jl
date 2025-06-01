# Save this as benchmark_solvers.jl
# To run: mpiexec -n 4 julia benchmark_solvers.jl

# --- Required Packages ---
using MPI
using PartitionedArrays
# using PartitionedArrays.FEM # Removed as FEM is not found as a direct submodule in this context
using SparseArrays # For local sparse matrix construction if needed
using LinearAlgebra # For norm, dot - Identity should come from HPCG

# --- Import your HPCG module ---
# Ensure HPCG.jl is in your JULIA_LOAD_PATH or in the same directory,
# or Pkg.develop'd
using HPCG

# --- Function containing the core benchmark logic ---
function perform_benchmarks(distribute_func)
    # MPI is initialized by with_mpi.
    # We can get comm, rank, and nprocs directly from MPI API.
    comm = MPI.COMM_WORLD
    rank_0_idx = MPI.Comm_rank(comm) # 0-indexed
    nprocs = MPI.Comm_size(comm)

    # --- Problem Setup ---
    N_global_dim = 100 # Example: 100x100 grid for Laplacian
    # N_global = N_global_dim * N_global_dim # Total number of unknowns

    # 1. Create a PData object representing the 1D distribution of processes
    # This is the pattern like `ranks = distribute(LinearIndices((np,)))`
    # `distribute_func` is the function passed by `with_mpi`.
    process_layout_pdata = distribute_func(LinearIndices((nprocs,)))

    # 2. Create distributed matrix A
    # Using PartitionedArrays.poisson_matrix directly.
    # This function typically takes (ElementType, GlobalGridDims, ProcessLayoutPData)
    A_dist = PartitionedArrays.poisson_matrix(Float64, (N_global_dim, N_global_dim), process_layout_pdata)
    # Previous attempts that might cause UndefVarError in some environments/versions:
    # poisson_problem = FEM.Poisson((N_global_dim, N_global_dim), Float64, process_layout_pdata) # Caused UndefVarError for FEM
    # A_dist = FEM.matrix(poisson_problem)
    # A_dist = PartitionedArrays.laplacian_matrix(N_global_dim, N_global_dim, Float64, process_layout_pdata) # Caused UndefVarError for laplacian_matrix

    # 3. Define row partitioning for vectors based on the distributed matrix
    row_partition = A_dist.rows # This is an IndexSet or similar

    # Create distributed vectors
    xtrue_dist = PVector(undef, row_partition)
    map_parts(local_values(xtrue_dist)) do local_xtrue_vals # local_values gets the underlying local array
        fill!(local_xtrue_vals, 1.0)
    end
    # consistent!(xtrue_dist) |> wait # Often not needed after local fill if no overlap

    b_dist = A_dist * xtrue_dist # Parallel matrix-vector product
    # consistent!(b_dist) |> wait # Result of A*x should be consistent

    x_initial_dist = PVector(0.0, row_partition) # Creates a PVector filled with 0.0
    # consistent!(x_initial_dist) |> wait # Not needed for scalar fill

    # Solver parameters
    tol = 1e-8
    max_iters = 2000

    # --- Benchmark ref_cg! (from HPCG module) ---
    if rank_0_idx == 0
        println("--- Benchmarking HPCG.ref_cg! ---")
    end
    x_cg = PVector(0.0, row_partition) # Initialize with zeros
    copy!(x_cg, x_initial_dist)        # Or copy from a defined initial state
    # consistent!(x_cg) |> wait

    timing_data_cg = [0.0] # For the @elapsed time within the solver (local to each rank)

    # Assuming CGStateVariables is defined and exported by HPCG
    # and its constructor matches this usage with PVectors.
    # zero(PVector) and similar(PVector) are well-defined.
    cg_state_vars = HPCG.CGStateVariables(zero(x_cg), similar(x_cg), similar(x_cg))

    MPI.Barrier(comm) # Synchronize all processes before starting the timer
    time_cg_start = MPI.Wtime()
    
    x_sol_cg, _, res0_cg, res_final_cg, iters_cg = HPCG.ref_cg!(
        x_cg, A_dist, b_dist, timing_data_cg;
        tolerance = tol,
        maxiter = max_iters,
        statevars = cg_state_vars,
        Pl = HPCG.Identity() # Assuming Identity is defined and exported by HPCG
    )
    
    MPI.Barrier(comm) # Synchronize all processes before stopping the timer
    time_cg_end = MPI.Wtime()
    elapsed_time_cg = time_cg_end - time_cg_start

    if rank_0_idx == 0
        println("HPCG.ref_cg! completed.")
        println("  Iterations: $iters_cg")
        println("  Initial Residual (from solver): $res0_cg")
        println("  Final Residual: $res_final_cg")
        println("  Wall Time: $elapsed_time_cg seconds")
        # timing_data_cg[1] is local to rank 0 here.
        println("  Internal @elapsed (rank 0): $(timing_data_cg[1]) seconds")
    end

    # --- Benchmark ref_pipelined_cg! (from HPCG module) ---
    if rank_0_idx == 0
        println("\n--- Benchmarking HPCG.ref_pipelined_cg! ---")
    end
    x_ppcg = PVector(0.0, row_partition) # Initialize with zeros
    copy!(x_ppcg, x_initial_dist)       # Or copy from a defined initial state
    # consistent!(x_ppcg) |> wait

    timing_data_ppcg = [0.0]

    MPI.Barrier(comm)
    time_ppcg_start = MPI.Wtime()

    x_sol_ppcg, _, res0_ppcg, res_final_ppcg, iters_ppcg = HPCG.ref_pipelined_cg!(
        x_ppcg, A_dist, b_dist, timing_data_ppcg;
        tolerance = tol,
        maxiter = max_iters,
        Pl = HPCG.Identity() # Assuming Identity is defined and exported by HPCG
    )

    MPI.Barrier(comm)
    time_ppcg_end = MPI.Wtime()
    elapsed_time_ppcg = time_ppcg_end - time_ppcg_start

    if rank_0_idx == 0
        println("HPCG.ref_pipelined_cg! completed.")
        println("  Iterations: $iters_ppcg")
        println("  Initial Residual: $res0_ppcg")
        println("  Final Residual: $res_final_ppcg")
        println("  Wall Time: $elapsed_time_ppcg seconds")
        # timing_data_ppcg[1] is local to rank 0 here.
        println("  Internal @elapsed (rank 0): $(timing_data_ppcg[1]) seconds")
    end

    # --- Comparison ---
    if rank_0_idx == 0
        println("\n--- Comparison (Wall Times) ---")
        println("HPCG.ref_cg!:          $elapsed_time_cg seconds")
        println("HPCG.ref_pipelined_cg!: $elapsed_time_ppcg seconds")
        if elapsed_time_cg < elapsed_time_ppcg
            println("HPCG.ref_cg! was faster.")
        elseif elapsed_time_ppcg < elapsed_time_cg
            println("HPCG.ref_pipelined_cg! was faster.")
        else
            println("Both solvers took approximately the same time.")
        end
    end
    # MPI.Finalize() is handled by with_mpi
end

# --- Main Execution Block ---
if abspath(PROGRAM_FILE) == @__FILE__
    # PartitionedArrays.with_mpi will initialize and finalize MPI.
    # It passes a `distribute` function to the lambda.
    PartitionedArrays.with_mpi() do distribute_func_arg
        perform_benchmarks(distribute_func_arg)
    end
end
