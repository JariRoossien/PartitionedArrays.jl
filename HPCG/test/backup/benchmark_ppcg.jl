using LinearAlgebra
using SparseArrays
using Random
using Statistics

# Include mock non-blocking implementations
include("mock_non_blocking.jl")

# Define the Identity preconditioner
struct Identity end
LinearAlgebra.ldiv!(dst::AbstractVector, ::Identity, src::AbstractVector) = copyto!(dst, src)
LinearAlgebra.ldiv!(::Identity, src::AbstractVector) = copy(src)

# Define the PPCGIterable struct
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
    # Futures for dot products started in the previous iteration
    next_dot_wu_future::Any  # Future for next (w, u) dot product
    next_dot_mw_future::Any  # Future for next (m, w) dot product
    # Flag to indicate which iterate function to use
    use_non_blocking::Bool   # Whether to use non-blocking dot products
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
        if it.use_non_blocking
            # Non-blocking implementation
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

                # Start non-blocking dot products early
                # g₀ = (w₀, u₀)
                dot_wu_future = non_blocking_dot(it.w, it.u, it.dot_wu_setup)
                # d₀ = (m₀, w₀) 
                dot_mw_future = non_blocking_dot(it.m, it.w, it.dot_mw_setup)

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
                dot_wu = fetch(dot_wu_future)
                dot_mw = fetch(dot_mw_future)
                
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
                
                # Start the dot products for the next iteration early
                # This will overlap with the residual computation below
                next_dot_wu_future = non_blocking_dot(it.w, it.u, it.dot_wu_setup)
                next_dot_mw_future = non_blocking_dot(it.m, it.w, it.dot_mw_setup)
                
                # Store these futures for the next iteration
                it.next_dot_wu_future = next_dot_wu_future
                it.next_dot_mw_future = next_dot_mw_future
            else
                # Get the dot product results that were started in the previous iteration
                dot_wu = fetch(it.next_dot_wu_future)
                dot_mw = fetch(it.next_dot_mw_future)
                
                # mᵢ = M⁻¹wᵢ 
                ldiv!(it.m, it.Pl, it.w)
                
                # nᵢ = A mᵢ
                mul!(it.n, it.A, it.m)
                
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
                
                # Start the dot products for the next iteration early
                # This will overlap with the residual computation below
                next_dot_wu_future = non_blocking_dot(it.w, it.u, it.dot_wu_setup)
                next_dot_mw_future = non_blocking_dot(it.m, it.w, it.dot_mw_setup)
                
                # Store these futures for the next iteration
                it.next_dot_wu_future = next_dot_wu_future
                it.next_dot_mw_future = next_dot_mw_future
            end
        else
            # Standard implementation with blocking dot products
            if iteration == 0
                # (r₀ is already computed and stored in it.r)
                # u₀ = M⁻¹r₀
                ldiv!(it.u, it.Pl, it.r)
                # w₀ = A u₀
                mul!(it.w, it.A, it.u)
                
                # m₀ = M⁻¹w₀
                ldiv!(it.m, it.Pl, it.w)

                # g₀ = (w₀, u₀)
                dot_wu = dot(it.w, it.u)
                # d₀ = (m₀, w₀) 
                dot_mw = dot(it.m, it.w)

                # n₀ = A m₀
                mul!(it.n, it.A, it.m)            
                
                it.g_prev = dot_wu

                # α₀ = g₀ / d₀
                if abs(dot_mw) < eps_val # Avoid division by zero/small number
                    it.a_prev = zero(num_type)
                else
                    it.a_prev = dot_wu / dot_mw
                end

                a = it.a_prev # This is α₀

                # Initialize q₀, z₀ and p₀
                # q₀ = m₀
                copyto!(it.q, it.m)
                # z₀ = n₀
                copyto!(it.z, it.n)
                # p₀ = u₀
                copyto!(it.p, it.u)

                # x₁ = x₀ + α₀ p₀
                @. it.x += a * it.p
                # u₁ = u₀ - α₀ q₀
                @. it.u -= a * it.q
                # w₁ = w₀ - α₀ z₀
                @. it.w -= a * it.z
            else
                # mᵢ = M⁻¹wᵢ 
                ldiv!(it.m, it.Pl, it.w)

                # gᵢ = (wᵢ, uᵢ) 
                dot_wu = dot(it.w, it.u)
                # dᵢ = (mᵢ, wᵢ)
                dot_mw = dot(it.m, it.w)
                
                # nᵢ = A mᵢ
                mul!(it.n, it.A, it.m)

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
        end

        # Update residual: r = b - Ax
        mul!(it.r, it.A, it.x) # r_temp = A*x
        @. it.r = it.b - it.r   # r = b - A*x
        it.residual = norm(it.r)
    end

    return it.residual, iteration + 1
end

# Create a standard pipelined CG implementation without non-blocking dot products
function standard_pipelined_cg!(x, A, b, timing_data;
    tolerance::Float64 = 1e-6,
    maxiter::Int = size(A, 2),
    Pl = Identity())

    timing_data[1] = 0.0

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
    a_prev_init = one(eltype(x))

    # Create the iterable with use_non_blocking = false
    iterable = PPCGIterable(Pl, A, x, b, r, u, w, m, n, z, q, p,
        g_prev_init, a_prev_init,
        tolerance, residual0, current_residual,
        maxiter, timing_data, nothing, nothing, nothing, nothing, false)
    
    iters = 0
    for res_norm in iterable
        iters += 1
    end

    return iterable.x, iterable.timing_data, iterable.residual0, iterable.residual, iters
end

# Create a non-blocking pipelined CG implementation
function non_blocking_pipelined_cg!(x, A, b, timing_data;
    tolerance::Float64 = 1e-6,
    maxiter::Int = size(A, 2),
    Pl = Identity())

    timing_data[1] = 0.0

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
    next_dot_wu_future = nothing
    next_dot_mw_future = nothing

    # Create the iterable with use_non_blocking = true
    iterable = PPCGIterable(Pl, A, x, b, r, u, w, m, n, z, q, p,
        g_prev_init, a_prev_init,
        tolerance, residual0, current_residual,
        maxiter, timing_data, dot_wu_setup, dot_mw_setup,
        next_dot_wu_future, next_dot_mw_future, true)
    
    iters = 0
    for res_norm in iterable
        iters += 1
    end

    return iterable.x, iterable.timing_data, iterable.residual0, iterable.residual, iters
end

# Define a function to create a test problem
function create_test_problem(n::Int)
    # Create a sparse matrix with some structure
    # Using a 5-point stencil for a 2D Poisson problem
    A = spdiagm(
        0 => 4.0*ones(n*n),
        1 => -ones(n*n-1),
        -1 => -ones(n*n-1),
        n => -ones(n*n-n),
        -n => -ones(n*n-n)
    )
    
    # Remove connections between grid boundaries
    for i in 1:n-1
        A[i*n, i*n+1] = 0.0
        A[i*n+1, i*n] = 0.0
    end
    
    # Create a random right-hand side
    Random.seed!(42)  # For reproducibility
    b = rand(n*n)
    
    # Initial guess is all zeros
    x0 = zeros(n*n)
    
    return A, b, x0
end

# Function to benchmark both implementations
function benchmark_cg_implementations(n::Int, num_runs::Int=10)
    println("Benchmarking with $(n)x$(n) grid ($(n*n)x$(n*n) matrix):")
    
    # Create test problem
    A, b, x0 = create_test_problem(n)
    
    # Arrays to store timing results
    standard_times = zeros(num_runs)
    nonblocking_times = zeros(num_runs)
    standard_iters = zeros(Int, num_runs)
    nonblocking_iters = zeros(Int, num_runs)
    
    # Run benchmarks
    for i in 1:num_runs
        # Standard pipelined CG
        timing_data_std = [0.0]
        _, _, _, _, iters_std = standard_pipelined_cg!(
            copy(x0), A, b, timing_data_std, tolerance=1e-8, maxiter=1000)
        standard_times[i] = timing_data_std[1]
        standard_iters[i] = iters_std
        
        # Non-blocking pipelined CG
        timing_data_nb = [0.0]
        _, _, _, _, iters_nb = non_blocking_pipelined_cg!(
            copy(x0), A, b, timing_data_nb, tolerance=1e-8, maxiter=1000)
        nonblocking_times[i] = timing_data_nb[1]
        nonblocking_iters[i] = iters_nb
    end
    
    # Calculate statistics
    std_mean_time = mean(standard_times)
    nb_mean_time = mean(nonblocking_times)
    std_mean_iters = mean(standard_iters)
    nb_mean_iters = mean(nonblocking_iters)
    speedup = std_mean_time / nb_mean_time
    
    # Print results
    println("Standard Pipelined CG:")
    println("  Average time: $(std_mean_time) seconds")
    println("  Average iterations: $(std_mean_iters)")
    
    println("\nNon-blocking Pipelined CG:")
    println("  Average time: $(nb_mean_time) seconds")
    println("  Average iterations: $(nb_mean_iters)")
    
    println("\nSpeedup: $(speedup)x")
    
    return standard_times, nonblocking_times, standard_iters, nonblocking_iters
end

# Run the benchmark with a 64x64 grid (4096x4096 matrix)
benchmark_cg_implementations(64, 5)
