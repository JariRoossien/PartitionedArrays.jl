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
    z::vecT           # Search direction related to w
    q::vecT           # Search direction related to u
    p::vecT           # Search direction for x
    g_prev::numT      # Previous g = (w, u)
    a_prev::numT      # Previous step length alpha
    tol::numT         # Tolerance for convergence
    residual0::numT   # Initial residual norm
    residual::numT    # Current residual norm
    maxiter::Int      # Maximum number of iterations
    timing_data::Vector{Float64} # timing
    # Setup for non-blocking dot products
    dot_wu_setup::Any  # Setup for (w, u) dot product
    dot_mw_setup::Any  # Setup for (m, w) dot product
    # Current dot product futures
    dot_wu_future::Any  # Future for current (w, u) dot product
    dot_mw_future::Any  # Future for current (m, w) dot product
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
            # (r₀ is already computed and stored in it.r)
            # u₀ = M⁻¹r₀
            ldiv!(it.u, it.Pl, it.r)
            # w₀ = A u₀
            mul!(it.w, it.A, it.u)
            
            # m₀ = M⁻¹w₀
            ldiv!(it.m, it.Pl, it.w)

            # Initialize non-blocking dot product setups
            it.dot_wu_setup = setup_non_blocking_dot(it.w, it.u)
            it.dot_mw_setup = setup_non_blocking_dot(it.m, it.w)

            # Start non-blocking dot products
            # g₀ = (w₀, u₀)
            it.dot_wu_future = non_blocking_dot(it.w, it.u, it.dot_wu_setup)
            # d₀ = (m₀, w₀) 
            it.dot_mw_future = non_blocking_dot(it.m, it.w, it.dot_mw_setup)

            # n₀ = A m₀ (compute this while dot products are in progress)
            mul!(it.n, it.A, it.m)            
            
            # Initialize q₀, z₀ and p₀ (more computation while dot products are in progress)
            # q₀ = m₀
            copyto!(it.q, it.m)
            # z₀ = n₀
            copyto!(it.z, it.n)
            # p₀ = u₀
            copyto!(it.p, it.u)
            
            # Now we need the dot product results
            dot_wu = fetch(it.dot_wu_future)
            dot_mw = fetch(it.dot_mw_future)
            
            it.g_prev = dot_wu

            # α₀ = g₀ / d₀
            if abs(dot_mw) < eps_val # Avoid division by zero/small number
                it.a_prev = zero(num_type)
            else
                it.a_prev = dot_wu / dot_mw
            end

            a = it.a_prev # This is α₀

            # x₁ = x₀ + α₀ p₀
            @. it.x += a * it.p
            # u₁ = u₀ - α₀ q₀
            @. it.u -= a * it.q
            # w₁ = w₀ - α₀ z₀
            @. it.w -= a * it.z
        else
            # mᵢ = M⁻¹wᵢ (do this first!)
            ldiv!(it.m, it.Pl, it.w)
            
            # Start non-blocking dot products for the current iteration
            # gᵢ = (wᵢ, uᵢ)
            it.dot_wu_future = non_blocking_dot(it.w, it.u, it.dot_wu_setup)
            # dᵢ = (mᵢ, wᵢ)
            it.dot_mw_future = non_blocking_dot(it.m, it.w, it.dot_mw_setup)
            
            # nᵢ = A mᵢ (compute this while dot products are in progress)
            mul!(it.n, it.A, it.m)
            
            # Get the results of the dot products
            dot_wu = fetch(it.dot_wu_future)
            dot_mw = fetch(it.dot_mw_future)
            
            # βᵢ = gᵢ / g_{i-1}
            local b_i 
            if abs(it.g_prev) < eps_val
                b_i = zero(num_type) # Avoid division by zero
            else
                b_i = dot_wu / it.g_prev
            end

            # Denominator for αᵢ: dᵢ - βᵢ * gᵢ / α_{i-1}
            local a_i_denominator
            if abs(it.a_prev) < eps_val 
                a_i_denominator = dot_mw 
            else
                a_i_denominator = dot_mw - b_i * dot_wu / it.a_prev
            end

            # αᵢ = gᵢ / (dᵢ - βᵢ * gᵢ / α_{i-1})
            local a_i
            if abs(a_i_denominator) < eps_val
                a_i = zero(num_type) # Avoid division by zero
            else
                a_i = dot_wu / a_i_denominator
            end

            # Update search directions p, q, z
            # pᵢ = uᵢ + βᵢ p_{i-1}
            @. it.p = it.u + b_i * it.p
            # qᵢ = mᵢ + βᵢ q_{i-1}
            @. it.q = it.m + b_i * it.q
            # zᵢ = nᵢ + βᵢ z_{i-1}
            @. it.z = it.n + b_i * it.z
            
            # x_{i+1} = xᵢ + αᵢ pᵢ
            @. it.x += a_i * it.p
            # u_{i+1} = uᵢ - αᵢ qᵢ
            @. it.u -= a_i * it.q
            # w_{i+1} = wᵢ - αᵢ zᵢ
            @. it.w -= a_i * it.z

            # Store current g and α for next iteration
            it.g_prev = dot_wu
            it.a_prev = a_i
        end

        # Update residual: r = b - Ax
        mul!(it.r, it.A, it.x) # r_temp = A*x
        @. it.r = it.b - it.r   # r = b - A*x
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
    r .-= tmp       # r = b - Ax₀

    residual0 = norm(r)
    current_residual = residual0
    
    # Allocate workspace vectors
    u = similar(x)
    w = similar(x)
    m = similar(x)
    n = similar(x) 
    z = similar(x)
    q = similar(x)
    p = similar(x)

    g_prev_init = zero(eltype(x))
    a_prev_init = one(eltype(x)) # Using one to avoid potential 0/0

    # Setup for non-blocking dot products
    dot_wu_setup = setup_non_blocking_dot(w, u)
    dot_mw_setup = setup_non_blocking_dot(m, w)
    
    # Initialize futures as nothing
    dot_wu_future = nothing
    dot_mw_future = nothing

    return PPCGIterable(Pl, A, x, b, r, u, w, m, n, z, q, p,
        g_prev_init, a_prev_init,
        tolerance, residual0, current_residual,
        maxiter, timing_data, dot_wu_setup, dot_mw_setup,
        dot_wu_future, dot_mw_future)
end

function non_blocking_pipelined_cg!(x, A, b, timing_data;
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
