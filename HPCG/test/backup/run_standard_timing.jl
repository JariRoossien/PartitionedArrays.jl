# Run standard pipelined CG
include("../src/pipelined_cg.jl")

timing_data_std = [0.0]
start_time = time()
x_std, _, _, _, std_iters = ref_pipelined_cg!(
    copy(x0), A, b, timing_data_std, tolerance=1e-8, maxiter=1000)
std_time = time() - start_time
