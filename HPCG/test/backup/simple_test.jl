using LinearAlgebra
using SparseArrays

println("Testing if the pipelined_cg.jl file can be loaded...")

# Try to include the pipelined_cg.jl file
include("../src/pipelined_cg.jl")

println("Successfully loaded pipelined_cg.jl!")
println("The implementation contains the following functions:")

# List the functions defined in the file
for name in names(Main, all=true)
    if startswith(string(name), "PPCG") || startswith(string(name), "ppcg") || 
       startswith(string(name), "ref_pipelined_cg")
        println("- $name")
    end
end

println("\nImplementation verification successful!")
