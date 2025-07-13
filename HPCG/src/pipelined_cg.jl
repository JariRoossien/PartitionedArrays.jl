import LinearAlgebra: mul!, dot, norm, ldiv!

# Identity preconditioner fallback
struct Identity end

LinearAlgebra.ldiv!(dst::AbstractVector, ::Identity, src::AbstractVector) = copyto!(dst, src)
LinearAlgebra.ldiv!(::Identity, src::AbstractVector) = copy(src)


mutable struct PPCGIterable{precT, matT, solT, vecT, numT <: Real}
    Pl::precT         # Preconditioner
    A::matT           # System matrix
    x::solT           # Solution vector
    b::vecT           # Right-hand side vector
    r::vecT           # Residual vector (r = b - Ax)
    u::vecT           # Preconditioned residual or related vector
    w::vecT           # A * u or related vector
    m::vecT           # M^{-1}w or related vector
    n::vecT           # A * m or related vector
    s::vecT           # A * p or related vector
    z::vecT           # Search direction related to n
    q::vecT           # Search direction related to m
    p::vecT           # Search direction for x
    
    gamma_prev::numT  # Previous gamma_i = (r_i, u_i)
    alpha_prev::numT  # Previous step length alpha_i
    tol::numT         # Tolerance for convergence
    residual0::numT   # Initial residual norm
    residual::numT    # Current residual norm
    maxiter::Int      # Maximum number of iterations
    timing_data::Vector{Float64} # timing

    # Setup for non-blocking dot products
    dot_gamma_setup::Any # Setup for (r, u) dot product for gamma_i
    dot_d_setup::Any     # Setup for (w, u) dot product for d

    # Current dot product futures
    dot_gamma_future::Any # Future for current (r, u) dot product
    dot_d_future::Any     # Future for current (w, u) dot product
end

@inline function converged(it::PPCGIterable)
    if it.residual0 == zero(it.residual0)
        return true 
    end
    return it.residual / it.residual0 <= it.tol
end

@inline start(it::PPCGIterable) = 0

@inline function done(it::PPCGIterable, iteration::Int)
    if converged(it)
        return true
    end
    if iteration >= it.maxiter
        return true
    end
    return false
end

function Base.iterate(it::PPCGIterable, iteration::Int = start(it))
    if done(it, iteration)
        return nothing
    end

    num_type = eltype(it.x)
    eps_val = eps(num_type)

    it.timing_data[1] += @elapsed begin
        if iteration == 0
            # Line 1: r_0 := b - Ax_0; u_0 := M^{-1}r_0; w_0 := Au_0
            # r_0 is already computed and stored in it.r
            ldiv!(it.u, it.Pl, it.r) # u_0 = M^{-1}r_0
            mul!(it.w, it.A, it.u)   # w_0 = A u_0

            # Initialize non-blocking dot product setups
            it.dot_gamma_setup = setup_non_blocking_dot(it.r, it.u)
            it.dot_d_setup = setup_non_blocking_dot(it.w, it.u)

            # Start non-blocking dot products for gamma_0 and d
            it.dot_gamma_future = non_blocking_dot(it.r, it.u, it.dot_gamma_setup)
            # Fetch dot products
            gamma_i = fetch(it.dot_gamma_future)
            

            it.dot_d_future = non_blocking_dot(it.w, it.u, it.dot_d_setup)
            d = fetch(it.dot_d_future)

            # Line 5: m_0 := M^{-1}w_0 (compute while dots are in progress)
            ldiv!(it.m, it.Pl, it.w)
            # Line 6: n_0 := Am_0 (compute while dots are in progress)
            mul!(it.n, it.A, it.m)

            
            # Line 11: b_i := 0 (for i = 0)
            b_i = zero(num_type)

            # Line 12: a_i := gamma_i / d
            local a_i
            if abs(d) < eps_val # Avoid division by zero/small number
                a_i = zero(num_type)
            else
                a_i = gamma_i / d
            end
            
            # Initialize search directions for i=0
            # p_0 = u_0
            copyto!(it.p, it.u)
            # s_0 = w_0 (from (A p_0))
            copyto!(it.s, it.w)
            # q_0 = m_0
            copyto!(it.q, it.m)
            # z_0 = n_0
            copyto!(it.z, it.n)

            # Store current gamma and alpha for next iteration
            it.gamma_prev = gamma_i
            it.alpha_prev = a_i

        else # iteration > 0
            # Lines 3 and 4: gamma_i := (r_i, u_i); d := (w_i, u_i)
            # gamma_{i+1} = (r_{i+1}, u_{i+1})
            it.dot_gamma_future = non_blocking_dot(it.r, it.u, it.dot_gamma_setup)
            gamma_i = fetch(it.dot_gamma_future)
            # d_{i+1} = (w_{i+1}, u_{i+1})
            it.dot_d_future = non_blocking_dot(it.w, it.u, it.dot_d_setup)
            d = fetch(it.dot_d_future)

            # Line 5: m_i := M^{-1}w_i 
            ldiv!(it.m, it.Pl, it.w)
            
            # Line 6: n_i := Am_i (compute while dot products are in progress for next iter)
            mul!(it.n, it.A, it.m)
            
            # Line 8: b_i := gamma_i / gamma_{i-1}
            local b_i
            if abs(it.gamma_prev) < eps_val
                b_i = zero(num_type) # Avoid division by zero
            else
                b_i = gamma_i / it.gamma_prev
            end

            # Line 9: a_i := gamma_i / (d - b_i * gamma_i / a_{i-1})
            local a_i_denominator
            if abs(it.alpha_prev) < eps_val
                a_i_denominator = d # Simplified case if alpha_prev is tiny
            else
                a_i_denominator = d - b_i * gamma_i / it.alpha_prev
            end

            local a_i
            if abs(a_i_denominator) < eps_val
                a_i = zero(num_type) # Avoid division by zero
            else
                a_i = gamma_i / a_i_denominator
            end

            # Update search directions using current b_i
            # Line 13: z_i := n_i + b_i * z_{i-1}
            @. it.z = it.n + b_i * it.z
            # Line 14: q_i := m_i + b_i * q_{i-1}
            @. it.q = it.m + b_i * it.q
            # Line 15: s_i := w_i + b_i * s_{i-1}
            @. it.s = it.w + b_i * it.s
            # Line 16: p_i := u_i + b_i * p_{i-1}
            @. it.p = it.u + b_i * it.p

            # Store current gamma and alpha for next iteration
            it.gamma_prev = gamma_i
            it.alpha_prev = a_i
        end

        # Update solution and vectors
        # These updates use a_i and the most recent search directions
        # Line 17: x_{i+1} := x_i + a_i * p_i
        @. it.x += it.alpha_prev * it.p # Use it.alpha_prev which is the current a_i

        # Line 18: r_{i+1} := r_i - a_i * s_i
        @. it.r -= it.alpha_prev * it.s
        # Line 19: u_{i+1} := u_i - a_i * q_i
        @. it.u -= it.alpha_prev * it.q
        # Line 20: w_{i+1} := w_i - a_i * z_i
        @. it.w -= it.alpha_prev * it.z

        # Update residual norm (norm of r_{i+1})
        it.residual = norm(it.r)


    end

    return it.residual, iteration + 1
end

function ppcg_iterator!(x, A, b, timing_data, Pl = Identity();
    tolerance::Float64 = 1e-6, 
    maxiter::Int = size(A, 2))

    # Initial residual r₀ = b - Ax₀
    r = similar(x)
    copyto!(r, b)
    tmp = similar(x)
    mul!(tmp, A, x) # tmp = Ax₀
    r .-= tmp        # r = b - Ax₀

    residual0 = norm(r)
    current_residual = residual0
    
    # Allocate workspace vectors
    u = similar(x)
    w = similar(x)
    m = similar(x)
    n = similar(x) 
    s = similar(x)
    z = similar(x) # Allocated
    q = similar(x) # Allocated
    p = similar(x)

    gamma_prev_init = zero(eltype(x))
    alpha_prev_init = one(eltype(x))

    # Setup for non-blocking dot products (same for all iterations)
    dot_gamma_setup = setup_non_blocking_dot(r, u)
    dot_d_setup = setup_non_blocking_dot(w, u)
    
    # Initialize futures as nothing
    dot_gamma_future = nothing
    dot_d_future = nothing

    # Corrected constructor call: pass 'z' and 'q' vectors
    return PPCGIterable(Pl, A, x, b, r, u, w, m, n, s, z, q, p,
        gamma_prev_init, alpha_prev_init,
        tolerance, residual0, current_residual,
        maxiter, timing_data, dot_gamma_setup, dot_d_setup,
        dot_gamma_future, dot_d_future)
end

function pipelined_cg!(x, A, b, timing_data;
    tolerance::Float64 = 1e-6,
    maxiter::Int = size(A, 2),
    Pl = Identity())

    timing_data[1] = 0.0

    iterable = ppcg_iterator!(x, A, b, timing_data, Pl;
        tolerance = tolerance, maxiter = maxiter)
    
    iters = 0
    for res_norm in iterable
        iters += 1
    end

    return iterable.x, iterable.timing_data, iterable.residual0, iterable.residual, iters
end