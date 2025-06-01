module HPCGMPITests

using PartitionedArrays

include(joinpath("..", "..", "hpcg_time_test.jl"))

with_mpi(perform_benchmarks)

end # module
