import LinearAlgebra: mul!, dot, norm, ldiv!

# Identity preconditioner fallback
struct Identity end

# Define how ldiv! (M^{-1} * v) works for the Identity preconditioner
LinearAlgebra.ldiv!(dst::AbstractVector, ::Identity, src::AbstractVector) = copyto!(dst, src)
LinearAlgebra.ldiv!(::Identity, src::AbstractVector) = copy(src) # for out-of-place


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
    timing_data::Vector{Float64} # For timing
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
            # Initial setup:
            # u₀ = M⁻¹r₀ (r₀ is already computed and stored in it.r)
            ldiv!(it.u, it.Pl, it.r)
            # w₀ = A u₀
            mul!(it.w, it.A, it.u)
            
            # p₀ = u₀
            copyto!(it.p, it.u)

            # m₀ = M⁻¹w₀
            ldiv!(it.m, it.Pl, it.w)
            # n₀ = A m₀
            mul!(it.n, it.A, it.m)

            # Initialize search directions q₀ and z₀
            # q₀ = m₀
            copyto!(it.q, it.m)
            # z₀ = n₀
            copyto!(it.z, it.n)

            # g₀ = (w₀, u₀)
            dot_wu = dot(it.w, it.u)
            # d₀ = (m₀, w₀)  (Note: it.m is M⁻¹w₀)
            dot_mw = dot(it.m, it.w)
            
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
            # mᵢ = M⁻¹wᵢ 
            ldiv!(it.m, it.Pl, it.w)
            # nᵢ = A mᵢ
            mul!(it.n, it.A, it.m)
            
            # gᵢ = (wᵢ, uᵢ) 
            dot_wu = dot(it.w, it.u)
            # dᵢ = (mᵢ, wᵢ)
            dot_mw = dot(it.m, it.w)

            # βᵢ = gᵢ / g_{i-1}
            local b_i # Ensure b_i is defined
            if abs(it.g_prev) < eps_val
                b_i = zero(num_type) # Avoid division by zero, implies breakdown or convergence
            else
                b_i = dot_wu / it.g_prev
            end

            # Denominator for αᵢ: dᵢ - βᵢ * gᵢ / α_{i-1}
            local a_i_denominator
            if abs(it.a_prev) < eps_val 
                a_i_denominator = dot_mw # Or handle as breakdown
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
            
            # Update solution and auxiliary vectors
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
    nvec = similar(x) # 'n' in PPCGIterable
    z = similar(x)
    q = similar(x)
    p = similar(x)

    g_prev_init = zero(eltype(x))
    a_prev_init = one(eltype(x)) # Using one to avoid potential 0/0 if g_prev_init was also 0 and used early.

    return PPCGIterable(Pl, A, x, b, r, u, w, m, nvec, z, q, p,
        g_prev_init, a_prev_init,
        tolerance, residual0, current_residual,
        maxiter, timing_data)
end

function ref_pipelined_cg!(x, A, b, timing_data;
    tolerance::Float64 = 1e-6, # Default tolerance
    maxiter::Int = size(A, 2),
    Pl = Identity())

    if !(timing_data isa Vector{Float64}) || isempty(timing_data)
        println("Warning: timing_data not properly initialized. Using default [0.0].")
        timing_data = [0.0] 
    else
        timing_data[1] = 0.0 # Reset timer
    end

    iterable = ppcg_iterator!(x, A, b, timing_data, Pl;
        tolerance = tolerance, maxiter = maxiter)
    
    iters = 0
    for res_norm in iterable # Iterate consumes the iterable
        iters += 1
    end

    # Return final solution, timing data, initial residual, final residual, and iteration count
    return iterable.x, iterable.timing_data, iterable.residual0, iterable.residual, iters
end

