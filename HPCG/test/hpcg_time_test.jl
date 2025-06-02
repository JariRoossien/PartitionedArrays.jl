using MPI
using PartitionedArrays
using SparseArrays
using LinearAlgebra

using HPCG

function perform_benchmarks(distribute_func)

    comm = MPI.COMM_WORLD
    rank_0_idx = MPI.Comm_rank(comm) 
    nprocs = MPI.Comm_size(comm)

    global_nx = 64
    global_ny = 64
    global_nz = 32

    if nprocs == 1; px,py,pz = (1,1,1)
    elseif nprocs == 2; px,py,pz = (2,1,1) 
    elseif nprocs == 4; px,py,pz = (2,2,1)
    elseif nprocs == 8; px,py,pz = (2,2,2) 
    elseif nprocs > 0; px,py,pz = (nprocs,1,1);

    px = nprocs; py = 1; pz = 1;

    local_nx, local_ny, local_nz = 0,0,0
    if nprocs > 0 && px > 0 && py > 0 && pz > 0
        local_nx = div(global_nx, px)
        local_ny = div(global_ny, py)
        local_nz = div(global_nz, pz)
    end


    process_layout_pdata = distribute_func(LinearIndices((nprocs,)))

    if rank_0_idx == 0
        println("Attempting to build distributed matrix and vector using HPCG.build_p_matrix with:")
        println("  Global Dims: ($global_nx, $global_ny, $global_nz)")
        println("  Local Dims (calculated): ($local_nx, $local_ny, $local_nz) per process")
        println("  Processor Grid: ($px, $py, $pz) for $nprocs processes")
    end

    A_dist, b_dist = HPCG.build_p_matrix(process_layout_pdata,
                                        local_nx, local_ny, local_nz,
                                        global_nx, global_ny, global_nz,
                                        px, py, pz)

    x_initial_dist = similar(b_dist) 
    x_initial_dist .= 0.0 

    # Solver parameters
    tol = 1e-8
    max_iters = 2000

    # --- Benchmark ref_cg! (from HPCG module) ---
    if rank_0_idx == 0
        println("--- Benchmarking HPCG.ref_cg! ---")
    end
    x_cg = similar(b_dist)
    copy!(x_cg, x_initial_dist)

    timing_data_cg = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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
    x_ppcg = similar(b_dist)
    copy!(x_ppcg, x_initial_dist)

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

if abspath(PROGRAM_FILE) == @__FILE__
    PartitionedArrays.with_mpi() do distribute_func_arg
        perform_benchmarks(distribute_func_arg)
    end
end
