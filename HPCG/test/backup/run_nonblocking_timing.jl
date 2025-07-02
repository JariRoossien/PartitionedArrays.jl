# Run non-blocking pipelined CG
include("../src/pipelined_cg_non_blocking.jl")

timing_data_nb = [0.0]
start_time = time()
x_nb, _, _, _, nb_iters = non_blocking_pipelined_cg!(
    copy(x0), A, b, timing_data_nb, tolerance=1e-8, maxiter=1000)
nb_time = time() - start_time
