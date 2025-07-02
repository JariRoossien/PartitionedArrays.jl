using LinearAlgebra
using SparseArrays
using Random

# Define a simple reference CG implementation
function simple_cg(A, b, x0; tol=1e-8, maxiter=1000)
    x = copy(x0)
    r = b - A * x
    p = copy(r)
    rsold = dot(r, r)
    res0 = sqrt(rsold)
    
    for i = 1:maxiter
        Ap = A * p
        alpha = rsold / dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = dot(r, r)
        res = sqrt(rsnew)
        
        if res < tol * res0
            return x, res0, res, i
        end
        
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    end
    
    res = sqrt(rsold)
    return x, res0, res, maxiter
end

# Define a simplified pipelined CG implementation based on our changes
function simplified_pipelined_cg(A, b, x0; tol=1e-8, maxiter=1000)
    x = copy(x0)
    r = b - A * x
    u = similar(x)
    w = similar(x)
    m = similar(x)
    n = similar(x)
    z = similar(x)
    q = similar(x)
    p = similar(x)
    
    res0 = norm(r)
    res = res0
    
    # Initialize
    ldiv!(u, I, r)  # u₀ = M⁻¹r₀ (with identity preconditioner)
    mul!(w, A, u)   # w₀ = A u₀
    ldiv!(m, I, w)  # m₀ = M⁻¹w₀
    
    # g₀ = (w₀, u₀)
    dot_wu = dot(w, u)
    # d₀ = (m₀, w₀)
    dot_mw = dot(m, w)
    
    mul!(n, A, m)   # n₀ = A m₀
    
    g_prev = dot_wu
    
    # α₀ = g₀ / d₀
    a_prev = dot_wu / dot_mw
    
    # Initialize q₀, z₀ and p₀
    copyto!(q, m)   # q₀ = m₀
    copyto!(z, n)   # z₀ = n₀
    copyto!(p, u)   # p₀ = u₀
    
    # x₁ = x₀ + α₀ p₀
    x .+= a_prev .* p
    # u₁ = u₀ - α₀ q₀
    u .-= a_prev .* q
    # w₁ = w₀ - α₀ z₀
    w .-= a_prev .* z
    
    # Update residual
    mul!(r, A, x)   # r_temp = A*x
    r .= b .- r     # r = b - A*x
    res = norm(r)
    
    iter = 1
    while iter < maxiter && res > tol * res0
        # mᵢ = M⁻¹wᵢ
        ldiv!(m, I, w)
        
        # gᵢ = (wᵢ, uᵢ)
        dot_wu = dot(w, u)
        # dᵢ = (mᵢ, wᵢ)
        dot_mw = dot(m, w)
        
        # nᵢ = A mᵢ
        mul!(n, A, m)
        
        # βᵢ = gᵢ / g_{i-1}
        b_i = dot_wu / g_prev
        
        # Denominator for αᵢ: dᵢ - βᵢ * gᵢ / α_{i-1}
        a_i_denominator = dot_mw - b_i * dot_wu / a_prev
        
        # αᵢ = gᵢ / (dᵢ - βᵢ * gᵢ / α_{i-1})
        a_i = dot_wu / a_i_denominator
        
        # Update search directions p, q, z
        # pᵢ = uᵢ + βᵢ p_{i-1}
        p .= u .+ b_i .* p
        # qᵢ = mᵢ + βᵢ q_{i-1}
        q .= m .+ b_i .* q
        # zᵢ = nᵢ + βᵢ z_{i-1}
        z .= n .+ b_i .* z
        
        # x_{i+1} = xᵢ + αᵢ pᵢ
        x .+= a_i .* p
        # u_{i+1} = uᵢ - αᵢ qᵢ
        u .-= a_i .* q
        # w_{i+1} = wᵢ - αᵢ zᵢ
        w .-= a_i .* z
        
        # Store current g and α for next iteration
        g_prev = dot_wu
        a_prev = a_i
        
        # Update residual
        mul!(r, A, x)   # r_temp = A*x
        r .= b .- r     # r = b - A*x
        res = norm(r)
        
        iter += 1
    end
    
    return x, res0, res, iter
end

# Define a simple test problem
function create_test_problem(n::Int)
    # Create a simple tridiagonal matrix
    A = spdiagm(0 => 2.0*ones(n), 1 => -1.0*ones(n-1), -1 => -1.0*ones(n-1))
    # Create a random right-hand side
    Random.seed!(42)  # For reproducibility
    b = rand(n)
    # Initial guess is all zeros
    x0 = zeros(n)
    return A, b, x0
end

# Function to compare the results of different CG implementations
function compare_cg_implementations()
    n = 100
    A, b, x0 = create_test_problem(n)
    
    # Solve using simple reference CG
    x_ref, res0_ref, res_ref, iters_ref = simple_cg(
        A, b, x0, tol=1e-8, maxiter=1000)
    
    # Solve using simplified pipelined CG
    x_ppcg, res0_ppcg, res_ppcg, iters_ppcg = simplified_pipelined_cg(
        A, b, x0, tol=1e-8, maxiter=1000)
    
    # Compare results
    println("Reference CG:")
    println("  Iterations: $iters_ref")
    println("  Initial residual: $res0_ref")
    println("  Final residual: $res_ref")
    
    # Compute true residual for reference CG
    true_res_ref = norm(b - A * x_ref)
    println("  True residual: $true_res_ref")
    
    println("\nPipelined CG:")
    println("  Iterations: $iters_ppcg")
    println("  Initial residual: $res0_ppcg")
    println("  Final residual: $res_ppcg")
    
    # Compute true residual for pipelined CG
    true_res_ppcg = norm(b - A * x_ppcg)
    println("  True residual: $true_res_ppcg")
    
    # Compare solutions
    solution_diff = norm(x_ref - x_ppcg) / norm(x_ref)
    println("\nSolution difference (relative): $solution_diff")
    
    # Compare residuals
    residual_diff = abs(res_ref - res_ppcg) / res_ref
    println("Residual difference (relative): $residual_diff")
    
    # Check if the solutions are close enough
    if solution_diff < 1e-6 && residual_diff < 1e-6
        println("\nThe implementations produce equivalent results!")
        return true
    else
        println("\nThe implementations produce different results.")
        return false
    end
end

# Run the comparison
compare_cg_implementations()
