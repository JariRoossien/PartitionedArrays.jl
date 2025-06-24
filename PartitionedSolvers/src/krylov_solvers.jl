
function krylov_options(p;
        iterations = length(rhs(p)),
        abstol = zero(real(eltype(rhs(p)))),
        reltol = sqrt(eps(real(eltype(rhs(p))))),
        norm = norm,
        Pl=preconditioner(identity_solver,p),
        update_Pl = true,
        verbose = false,
        output_prefix = "",
    )
    KrylovOptions(iterations,abstol,reltol,norm,Pl,update_Pl,verbose,output_prefix)
end

struct KrylovOptions{A,B,C,D,E,F,G,H} <: AbstractType
    iterations::A
    abstol::B
    reltol::C
    norm::D
    Pl::E
    update_Pl::F
    verbose::G
    output_prefix::H
end

function update(o::KrylovOptions;kwargs...)
    data = (;kwargs...)
    if hasproperty(data,:iterations)
        iterations = data.iterations
    else
        iterations = o.iterations
    end
    if hasproperty(data,:abstol)
        abstol = data.abstol
    else
        abstol = o.abstol
    end
    if hasproperty(data,:reltol)
        reltol = data.reltol
    else
        reltol = o.reltol
    end
    if hasproperty(data,:norm)
        norm = data.norm
    else
        norm = o.norm
    end
    if hasproperty(data,:Pl)
        Pl = data.Pl
    else
        Pl = o.Pl
    end
    if hasproperty(data,:update_Pl)
        update_Pl = data.update_Pl
    else
        update_Pl = o.update_Pl
    end
    if hasproperty(data,:verbose)
        verbose = data.verbose
    else
        verbose = o.verbose
    end
    if hasproperty(data,:output_prefix)
        output_prefix = data.output_prefix
    else
        output_prefix = o.output_prefix
    end
    KrylovOptions(iterations,abstol,reltol,norm,Pl,update_Pl,verbose,output_prefix)
end

function cg_state(p)
    x = solution(p)
    A = matrix(p)
    dx = similar(x,axes(A,2))
    u = similar(x)
    r = similar(x,axes(A,1))
    ρ = one(real(eltype(x)))
    iteration = 0
    current = zero(real(eltype(x)))
    target = zero(real(eltype(x)))
    PGCState(dx,r,u,A,ρ,iteration,current,target)
end

struct PGCState{A,B,C,D,E,F,G,H} <: AbstractType
    dx::A
    r::B
    u::C
    A::D
    ρ::E
    iteration::F
    current::G
    target::H
end

function update(o::PGCState;kwargs...)
    data = (;kwargs...)
    if hasproperty(data,:dx)
        dx = data.dx
    else
        dx = o.dx
    end
    if hasproperty(data,:r)
        r = data.r
    else
        r = o.r
    end
    if hasproperty(data,:u)
        u = data.u
    else
        u = o.u
    end
    if hasproperty(data,:A)
        A = data.A
    else
        A = o.A
    end
    if hasproperty(data,:ρ)
        ρ = data.ρ
    else
        ρ = o.ρ
    end
    if hasproperty(data,:iteration)
        iteration = data.iteration
    else
        iteration = o.iteration
    end
    if hasproperty(data,:current)
        current = data.current
    else
        current = o.current
    end
    if hasproperty(data,:target)
        target = data.target
    else
        target = o.target
    end
    PGCState(dx,r,u,A,ρ,iteration,current,target)
end

struct KrylovWorkspace{A,B} <: AbstractType
    options::A
    state::B
end

function update(w::KrylovWorkspace;kwargs...)
    options = update(w.options;kwargs...)
    state = update(w.state;kwargs...)
    KrylovWorkspace(options,state)
end

function done(ws::KrylovWorkspace)
    converged(ws) || tired(ws)
end

function converged(ws::KrylovWorkspace)
    (;current,target) = ws.state
    current <= target
end

function tired(ws::KrylovWorkspace)
    (;iteration) = ws.state
    (;iterations) = ws.options
    iteration >= iterations
end

function print_progress_header(a::KrylovWorkspace)
    (;output_prefix,verbose) = a.options
    s = output_prefix
    v = verbose
    c = "current"
    t = "target"
    v && @printf "%s%20s %20s\n" s "iterations" "residual"
    v && @printf "%s%10s %10s %10s %10s\n" s c t c t
end

function print_progress(a::KrylovWorkspace)
    (;output_prefix,verbose,iterations) = a.options
    (;iteration,current,target) = a.state
    s = output_prefix
    v = verbose
    v && @printf "%s%10i %10i %10.2e %10.2e\n" s iteration iterations current target
end

function cg(p;kwargs...)
    options = krylov_options(p;kwargs...)
    state = cg_state(p)
    workspace = KrylovWorkspace(options,state)
    linear_solver(cg_update,cg_step,p,workspace)
end

function cg_update(ws,A)
    (;Pl,update_Pl) = ws.options
    if update_Pl
        Pl = update(Pl,matrix=A)
    end
    iteration = 0
    update(ws;iteration,Pl,A)
end

function cg_step(x,ws,b,phase=:start;zero_guess=false,kwargs...)
    (;dx,r,u,A,ρ,iteration,current,target) = ws.state
    (;reltol,abstol,norm,Pl) = ws.options
    s = u
    if phase === :start
        iteration = 0
        phase = :advance
        copyto!(r,b)
        if ! zero_guess
            mul!(s,A,x)
            axpy!(-one(eltype(s)),s,r)
            #r .-= s
        end
        current = norm(r)
        target = max(reltol*current,abstol)
        fill!(dx,zero(eltype(dx)))
        ρ = one(eltype(x))
        ws = update(ws;iteration,ρ,current,target)
        print_progress_header(ws)
    end
    ldiv!(u,Pl,r)
    ρ_prev = ρ
    ρ = dot(r,u)
    β = ρ / ρ_prev
    axpby!(one(eltype(u)),u,β,dx)
    #dx .= u .+ β .* dx
    s = u
    mul!(s,A,dx)
    α = ρ / dot(s,dx)
    axpy!(α,dx,x)
    #x .+= α .* dx
    axpy!(-α,s,r)
    #r .-= α .* s
    iteration += 1
    current = norm(r)
    ws = update(ws;iteration,ρ,current)
    print_progress(ws)
    if done(ws)
        phase = :stop
    end
    x,ws,phase
end

