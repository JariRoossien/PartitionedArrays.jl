# Save this as benchmark_solvers.jl
# To run: mpiexec -n 4 julia benchmark_solvers.jl

# --- Required Packages ---
using MPI
using PartitionedArrays
using SparseArrays # For local sparse matrix construction if needed
using LinearAlgebra # For norm, dot - Identity should come from HPCG

# --- Import your HPCG module ---
# Ensure HPCG.jl is in your JULIA_LOAD_PATH or in the same directory,
# or Pkg.develop'd
using HPCG

# --- Function containing the core benchmark logic ---
function perform_benchmarks(distribute_func)
    # MPI is initialized by with_mpi.
    comm = MPI.COMM_WORLD
    rank_0_idx = MPI.Comm_rank(comm) # 0-indexed
    nprocs = MPI.Comm_size(comm)

    # --- Problem Setup ---
    # Define global problem dimensions (inspired by HPCG benchmark structures)
    global_nx = 64
    global_ny = 64
    global_nz = 32 # Example values from user snippet, make configurable if needed

    # Determine processor grid dimensions (px, py, pz)
    px, py, pz = (0,0,0) # Initialize
    if isdefined(HPCG, :compute_optimal_xyz) && nprocs > 0 # nprocs=0 check for safety
        px, py, pz = HPCG.compute_optimal_xyz(nprocs)
        if rank_0_idx == 0
            println("Using HPCG.compute_optimal_xyz for processor grid: ($px, $py, $pz)")
        end
    else
        if rank_0_idx == 0 && nprocs > 0
            @warn "HPCG.compute_optimal_xyz not found or nprocs=0. Using fallback logic for processor grid."
        end
        if nprocs == 1; px,py,pz = (1,1,1)
        elseif nprocs == 2; px,py,pz = (2,1,1) # Or (1,2,1) or (1,1,2)
        elseif nprocs == 4; px,py,pz = (2,2,1) # As in user snippet
        elseif nprocs == 8; px,py,pz = (2,2,2) # Common 3D decomposition
        elseif nprocs == 16; px,py,pz = (4,2,2) # Common 3D decomposition
        elseif nprocs > 0; px,py,pz = (nprocs,1,1); # Default to 1D
        else # nprocs == 0 or negative, problematic
            if rank_0_idx == 0; @error "Invalid nprocs: $nprocs"; end
            return # Cannot proceed
        end
    end

    if nprocs > 0 && px*py*pz != nprocs
        if rank_0_idx == 0
            @error "Processor grid dimensions ($px, $py, $pz) from compute_optimal_xyz or fallback (total $(px*py*pz)) do not multiply to nprocs ($nprocs). Overriding to 1D decomposition."
        end
        px = nprocs; py = 1; pz = 1;
    end

    # Calculate local dimensions per process
    local_nx, local_ny, local_nz = 0,0,0
    if nprocs > 0 && px > 0 && py > 0 && pz > 0
        if global_nx % px != 0 || global_ny % py != 0 || global_nz % pz != 0
            if rank_0_idx == 0
                @error "Global dimensions ($global_nx, $global_ny, $global_nz) are not evenly divisible by processor grid ($px, $py, $pz). Results may be incorrect or errors may occur."
            end
            # This is a critical issue for HPCG-like domain decomposition.
            # Forcing integer division, but this implies problem setup might be inconsistent.
        end
        local_nx = div(global_nx, px)
        local_ny = div(global_ny, py)
        local_nz = div(global_nz, pz)
    elseif nprocs > 0 # px,py,pz might be zero if compute_optimal_xyz had issues
         if rank_0_idx == 0; @error "Processor grid dimensions are invalid ($px,$py,$pz) for nprocs $nprocs"; end
         return
    end


    # 1. Create a PData object representing the 1D distribution of processes (ranks)
    process_layout_pdata = distribute_func(LinearIndices((nprocs,)))

    if rank_0_idx == 0
        println("Attempting to build distributed matrix and vector using HPCG.build_p_matrix with:")
        println("  Global Dims: ($global_nx, $global_ny, $global_nz)")
        println("  Local Dims (calculated): ($local_nx, $local_ny, $local_nz) per process")
        println("  Processor Grid: ($px, $py, $pz) for $nprocs processes")
    end

    # 2. Create distributed matrix A and vector b using HPCG.build_p_matrix
    # Ensure HPCG.build_p_matrix is exported and matches this assumed signature:
    # (ranks_PData, local_nx, local_ny, local_nz, global_nx, global_ny, global_nz, px, py, pz)
    A_dist, b_dist = HPCG.build_p_matrix(process_layout_pdata,
                                        local_nx, local_ny, local_nz,
                                        global_nx, global_ny, global_nz,
                                        px, py, pz)

    # 3. Define row partitioning for vectors based on the distributed matrix
    # Use axes(A_dist, 1) to get the row partitioning (AbstractIndexSet)
    # This is still useful for understanding the partitioning, but not directly for the PVector constructor that failed.
    # row_partition = axes(A_dist, 1) # Not strictly needed if using similar(b_dist)

    # Initial guess (distributed vector of zeros)
    # Create x_initial_dist by using similar on b_dist (which should be a correctly partitioned PVector)
    # and then fill it. This is more robust than PVector(0.0, row_partition).
    x_initial_dist = similar(b_dist) 
    x_initial_dist .= 0.0 # Fill with zeros using broadcasting

    # Solver parameters
    tol = 1e-8
    max_iters = 2000 # Default, can be adjusted

    # --- Benchmark ref_cg! (from HPCG module) ---
    if rank_0_idx == 0
        println("--- Benchmarking HPCG.ref_cg! ---")
    end
    x_cg = similar(b_dist) # Create with same partitioning as b_dist
    copy!(x_cg, x_initial_dist) # Copy the zeroed initial state

    timing_data_cg = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # zero(x_cg) and similar(x_cg) will correctly create PVectors with the same partitioning as x_cg
    cg_state_vars = HPCG.CGStateVariables(zero(x_cg), similar(x_cg), similar(x_cg))

    MPI.Barrier(comm)
    time_cg_start = MPI.Wtime()
    
    x_sol_cg, _, res0_cg, res_final_cg, iters_cg = HPCG.ref_cg!(
        x_cg, A_dist, b_dist, timing_data_cg;
        tolerance = tol,
        maxiter = max_iters,
        statevars = cg_state_vars,
        Pl = HPCG.Identity()
    )
    
    MPI.Barrier(comm)
    time_cg_end = MPI.Wtime()
    elapsed_time_cg = time_cg_end - time_cg_start

    if rank_0_idx == 0
        println("HPCG.ref_cg! completed.")
        println("  Iterations: $iters_cg")
        println("  Initial Residual (from solver): $res0_cg")
        println("  Final Residual: $res_final_cg")
        println("  Wall Time: $elapsed_time_cg seconds")
        println("  Internal @elapsed (rank 0): $(timing_data_cg[1]) seconds")
    end

    # --- Benchmark ref_pipelined_cg! (from HPCG module) ---
    if rank_0_idx == 0
        println("\n--- Benchmarking HPCG.ref_pipelined_cg! ---")
    end
    x_ppcg = similar(b_dist) # Create with same partitioning as b_dist
    copy!(x_ppcg, x_initial_dist) # Copy the zeroed initial state

    timing_data_ppcg = [0.0]

    MPI.Barrier(comm)
    time_ppcg_start = MPI.Wtime()

    x_sol_ppcg, _, res0_ppcg, res_final_ppcg, iters_ppcg = HPCG.ref_pipelined_cg!(
        x_ppcg, A_dist, b_dist, timing_data_ppcg;
        tolerance = tol,
        maxiter = max_iters,
        Pl = HPCG.Identity()
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
end

# --- Main Execution Block ---
if abspath(PROGRAM_FILE) == @__FILE__
    PartitionedArrays.with_mpi() do distribute_func_arg
        perform_benchmarks(distribute_func_arg)
    end
end
