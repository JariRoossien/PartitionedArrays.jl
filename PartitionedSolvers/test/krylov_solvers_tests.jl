module KrylovSolversTests

using Test
import PartitionedSolvers as PS
import PartitionedArrays as PA
import IterativeSolvers

n = 10
nodes_per_dir = (n,n,n)
args = PA.laplacian_fem(nodes_per_dir)
A = PA.sparse_matrix(args...)
x = ones(size(A,2))
b = A*x

y = similar(x)
y .= 0
p = PS.linear_problem(y,A,b)
Pl = PS.preconditioner(PS.amg,p)
s = PS.cg(p;verbose=true,Pl)
s = PS.solve(s)
@test x ≈ PS.solution(p)
s = PS.update(s,matrix=2*A)
s = PS.solve(s)
@test x ≈ 2*PS.solution(p)

z = IterativeSolvers.cg(A,b;verbose=true,Pl)

parts_per_dir = (2,2,2)
ranks = PA.DebugArray(LinearIndices((prod(parts_per_dir),)))

args = PA.laplacian_fem(nodes_per_dir,parts_per_dir,ranks)
A = PA.psparse(args...) |> fetch
x = PA.pones(axes(A,2))
b = A*x

y = similar(x)
y .= 0
p = PS.linear_problem(y,A,b)
Pl = PS.preconditioner(PS.amg,p)
s = PS.cg(p;verbose=true,Pl)
s = PS.solve(s)
@test x ≈ PS.solution(p)
s = PS.update(s,matrix=2*A)
s = PS.solve(s)
@test x ≈ 2*PS.solution(p)

end # module
