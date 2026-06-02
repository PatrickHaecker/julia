# This file is a part of Julia. License is MIT: https://julialang.org/license

# Incomplete type definitions.
#
# A top-level definition (e.g. a `struct` or method) that references a
# not-yet-defined binding in the current module raises an `UndefVarError`
# during evaluation. The C toplevel evaluator catches that error and calls
# `incomplete_defer`, which either (a) for method defs, rewrites the
# signature to admit a placeholder `DataType` and re-evaluates the def in
# place (the method is admitted dormant; later patched via
# `jl_method_resig`), or (b) for any other definitional form
# (struct/abstract/primitive/const-alias/macro), registers a zero-arg
# thunk that re-evaluates the original surface AST when the missing
# symbol later binds. Both kinds of pending work are tracked on a single
# canonical `IncompleteRef` per `(mod, name)` and drained from the
# binding-event hook `incomplete_drain_ready`. At module close,
# `incomplete_finalize` throws `IncompleteTypeError` listing every
# still-unresolved name and the count of methods / pending definitions
# waiting on each. At the REPL (Main) the close hook is not invoked, so
# unresolved entries linger until their dependency is bound.

const incomplete_lock = Threads.ReentrantLock()

# ---------------------------------------------------------------------------
# IncompleteTypeError — thrown at module close when one or more forward
# references never resolved. One error aggregates every unresolved name in
# the module so the user sees the full list of missing definitions in a
# single report (the previous design threw a `LoadError`/`CompositeException`
# of `UndefVarError`s, which lost the per-name dependent context).

struct IncompleteTypeEntry
    name::Symbol            # the unresolved binding
    srcfile::Symbol         # first reference site
    srcline::Int32
    n_methods::Int          # how many method dependents are waiting
    n_pending::Int          # how many replay finalizers are waiting
end

struct IncompleteTypeError <: Exception
    mod::Module
    entries::Vector{IncompleteTypeEntry}
end

# ---------------------------------------------------------------------------
# IncompleteRef — placeholder registry for not-yet-defined type names.
#
# When a method definition references a name that is not bound in the
# defining module, lowering substitutes a synthesised abstract `DataType`
# placeholder so the method is admitted into the table immediately (and
# `methods(f)` reports it honestly) but cannot match any concrete dispatch
# (no value can be `isa` an as-yet-undefined type). The placeholder lives
# in the method's signature; this `IncompleteRef` record is the registry
# entry that owns the placeholder, lists the methods / incomplete DataTypes
# that reference it, and is consulted when the real binding arrives to
# patch every dependent in place.
#
# One pending replay/finalizer thunk in `IncompleteRef.pending_finalizers`,
# bundled with the (optional) name of the definition that registered it and
# its source location. Used by `incomplete_resolve!` to invoke the thunk
# and by `InteractiveUtils.incomplete_definitions` to surface each pending
# definition individually at the REPL.
struct PendingFinalizer
    thunk
    defined_name::Union{Symbol,Nothing}
    srcfile::Symbol
    srcline::Int32
end

# One canonical `IncompleteRef` per `(mod, name)` pair. The registry is
# guarded by `incomplete_lock`.

mutable struct IncompleteRef
    const mod::Module
    const name::Symbol
    # Synthesised abstract `DataType` that stands in for the unresolved name
    # in method signatures and struct field types. Set when the placeholder
    # is materialised; `nothing` until then.
    placeholder::Union{DataType,Nothing}
    # Methods and incomplete `DataType`s that reference `placeholder`. Walked
    # on resolution to substitute the real type. Element type is `Any` to
    # avoid pulling forward declarations of `Method` / `DataType`.
    const dependents::Vector{Any}
    # Thunks invoked (in registration order) by `incomplete_resolve!`
    # after dependent methods have been patched. Used by struct / abstract /
    # primitive / const-alias lowering to finalise a placeholder DataType or
    # rebind a const alias once the waited-on name binds. A thunk that
    # itself raises a recoverable `UndefVarError` is re-registered on the
    # new missing name; any other exception propagates.
    const pending_finalizers::Vector{PendingFinalizer}
    # Source location of the first reference (for diagnostics).
    const srcfile::Symbol
    const srcline::Int32
end

# Per-module canonical registry: `(mod, name) -> IncompleteRef`.
const incomplete_refs = IdDict{Module,IdDict{Symbol,IncompleteRef}}()

# Reverse map: placeholder `DataType` -> owning `IncompleteRef`. Populated
# when a placeholder is materialised. Used to discover, from a method
# signature alone, which refs the method depends on (so we can register
# the method as a dependent without threading state through lowering).
const incomplete_placeholders = IdDict{DataType,IncompleteRef}()

# Look up the canonical `IncompleteRef` for `(mod, name)`, or `nothing` if
# none is registered. Read under `incomplete_lock`.
function find_incomplete_ref(mod::Module, name::Symbol)
    @lock incomplete_lock begin
        st = get(incomplete_refs, mod, nothing)
        isnothing(st) && return nothing
        return get(st, name, nothing)
    end
end

# Get the canonical `IncompleteRef` for `(mod, name)`, creating it on first
# call. `srcfile`/`srcline` are used only for the new record (existing
# records keep their original source location, so the first reference site
# is what shows up in diagnostics).
function get_or_create_incomplete_ref(mod::Module, name::Symbol,
                                      srcfile::Symbol = :none,
                                      srcline::Integer = 0)
    @lock incomplete_lock begin
        st = get!(() -> IdDict{Symbol,IncompleteRef}(), incomplete_refs, mod)
        existing = get(st, name, nothing)
        existing === nothing || return existing
        ref = IncompleteRef(mod, name, nothing, Any[], Any[], srcfile, Int32(srcline))
        st[name] = ref
        return ref
    end
end

# Remove the canonical `IncompleteRef` for `(mod, name)` from the registry.
# Called after a resolution pass has patched every dependent.
function remove_incomplete_ref!(mod::Module, name::Symbol)
    @lock incomplete_lock begin
        st = get(incomplete_refs, mod, nothing)
        isnothing(st) && return nothing
        delete!(st, name)
        isempty(st) && delete!(incomplete_refs, mod)
        return nothing
    end
end

# Return the canonical placeholder `DataType` for `(mod, name)`, materialising
# it lazily on first call. The placeholder is a fresh abstract `DataType`
# whose `TypeName` carries `name` and `mod`, so reflection (`methods(f)`,
# `MethodError` messages) prints it as `mod.name` — indistinguishable from a
# real abstract type, which matches the user's mental model that the method
# *is* defined and just waits for the type. The binding `mod.name` itself is
# **not** taken by the placeholder, so the user's later `name = ...` or
# `struct name ... end` binds the real value normally; the binding-event
# drain triggers placeholder resolution which patches every dependent
# method/struct.
#
# Holds `incomplete_lock` during creation to keep the canonicalisation
# invariant under concurrent first references.
function incomplete_placeholder(mod::Module, name::Symbol,
                                srcfile::Symbol = :none,
                                srcline::Integer = 0)
    ref = get_or_create_incomplete_ref(mod, name, srcfile, srcline)
    # Fast path: already materialised. Re-read under the lock since another
    # thread may be in the middle of `materialize_placeholder!`.
    @lock incomplete_lock begin
        p = ref.placeholder
        p === nothing || return p
        # Mirror the lowering of `abstract type name end`: create the type,
        # set its supertype to `Any`, and seal the body. Skipping any of the
        # three steps leaves the `DataType` in a partially-initialised state
        # that crashes `obviously_disjoint` during method-table intersection.
        T = Core._abstracttype(mod, name, Core.svec())
        Core._setsuper!(T, Any)
        Core._typebody!(false, T)
        ref.placeholder = T
        incomplete_placeholders[T] = ref
        return T
    end
end

# Lowering entry: resolve `name` in `mod` to a `Type`, falling back to the
# canonical placeholder if `name` is unbound. Used at method-signature
# evaluation sites: a method written as `f(x::A) = ...` evaluates `A` via
# this helper rather than a bare `getproperty(mod, :A)`, so an undefined `A`
# yields a placeholder `DataType` (admitting the method into the table in a
# dormant state) instead of an `UndefVarError`.
#
# Non-Type bindings (e.g. `f(x::1)`) propagate the original `TypeError` so
# real misuse still errors eagerly.
function incomplete_typeref(mod::Module, name::Symbol,
                            srcfile::Symbol = :none,
                            srcline::Integer = 0)
    if isdefined(mod, name)
        v = getglobal(mod, name)
        v isa Type && return v
        # Bound to a non-Type value: let the normal type assertion fire
        # downstream by returning the value unchanged.
        return v
    end
    return incomplete_placeholder(mod, name, srcfile, srcline)
end

# Walk a method signature `sig` (a `Tuple`/`UnionAll`/`Union`/`TypeVar`
# tree) and return every placeholder `DataType` referenced in it.
function incomplete_placeholders_in(@nospecialize(sig))
    found = DataType[]
    _scan_placeholders!(found, sig)
    return found
end

function _scan_placeholders!(found::Vector{DataType}, @nospecialize(t))
    if t isa DataType
        if haskey(incomplete_placeholders, t)
            t in found || push!(found, t)
        end
        for p in t.parameters
            _scan_placeholders!(found, p)
        end
    elseif t isa UnionAll
        _scan_placeholders!(found, t.var)
        _scan_placeholders!(found, t.body)
    elseif t isa Union
        _scan_placeholders!(found, t.a)
        _scan_placeholders!(found, t.b)
    elseif t isa TypeVar
        _scan_placeholders!(found, t.ub)
        _scan_placeholders!(found, t.lb)
    end
    return found
end

# Register `m` as a dependent of every `IncompleteRef` whose placeholder
# appears in `m.sig`. Called from the post-method-definition hook so the
# resolution pass can find every method to patch when the binding arrives.
# Idempotent: a method already on a ref's dependents list is not duplicated.
function incomplete_register_method!(m::Method)
    placeholders = incomplete_placeholders_in(m.sig)
    isempty(placeholders) && return nothing
    @lock incomplete_lock begin
        for p in placeholders
            ref = get(incomplete_placeholders, p, nothing)
            ref === nothing && continue
            m in ref.dependents || push!(ref.dependents, m)
        end
    end
    return nothing
end

# Substitute `placeholder => real_type` everywhere in a method signature
# tree (`Tuple`/`UnionAll`/`Union`/`TypeVar`). Returns the rewritten sig.
function _subst_placeholder(@nospecialize(t), placeholder::DataType, @nospecialize(real_type))
    t === placeholder && return real_type
    if t isa DataType
        any_changed = false
        new_params = Any[]
        for p in t.parameters
            np = _subst_placeholder(p, placeholder, real_type)
            np === p || (any_changed = true)
            push!(new_params, np)
        end
        any_changed || return t
        return t.name.wrapper{new_params...}
    elseif t isa UnionAll
        new_var = _subst_placeholder(t.var, placeholder, real_type)::TypeVar
        if new_var === t.var
            new_body = _subst_placeholder(t.body, placeholder, real_type)
            new_body === t.body && return t
            return UnionAll(new_var, new_body)
        else
            # The typevar's bounds changed, so a fresh `TypeVar` was minted.
            # Any reference to the *old* `TypeVar` inside the body must be
            # retargeted to the new one or the `UnionAll` becomes ill-formed
            # (body references a `TypeVar` that doesn't bind anywhere).
            rebound = _subst_typevar(t.body, t.var, new_var)
            new_body = _subst_placeholder(rebound, placeholder, real_type)
            return UnionAll(new_var, new_body)
        end
    elseif t isa Union
        new_a = _subst_placeholder(t.a, placeholder, real_type)
        new_b = _subst_placeholder(t.b, placeholder, real_type)
        (new_a === t.a && new_b === t.b) && return t
        return Union{new_a, new_b}
    elseif t isa TypeVar
        new_ub = _subst_placeholder(t.ub, placeholder, real_type)
        new_lb = _subst_placeholder(t.lb, placeholder, real_type)
        (new_ub === t.ub && new_lb === t.lb) && return t
        return TypeVar(t.name, new_lb, new_ub)
    end
    return t
end

# Walk a type tree and replace identity-references to `old` (a `TypeVar`)
# with `new`. Used by `_subst_placeholder` when a `UnionAll`'s typevar is
# itself rebuilt as a result of a placeholder substitution in its bounds.
function _subst_typevar(@nospecialize(t), old::TypeVar, new::TypeVar)
    t === old && return new
    if t isa DataType
        any_changed = false
        new_params = Any[]
        for p in t.parameters
            np = _subst_typevar(p, old, new)
            np === p || (any_changed = true)
            push!(new_params, np)
        end
        any_changed || return t
        return t.name.wrapper{new_params...}
    elseif t isa UnionAll
        new_var = _subst_typevar(t.var, old, new)::TypeVar
        new_body = _subst_typevar(t.body, old, new)
        (new_var === t.var && new_body === t.body) && return t
        return UnionAll(new_var, new_body)
    elseif t isa Union
        new_a = _subst_typevar(t.a, old, new)
        new_b = _subst_typevar(t.b, old, new)
        (new_a === t.a && new_b === t.b) && return t
        return Union{new_a, new_b}
    elseif t isa TypeVar
        new_ub = _subst_typevar(t.ub, old, new)
        new_lb = _subst_typevar(t.lb, old, new)
        (new_ub === t.ub && new_lb === t.lb) && return t
        return TypeVar(t.name, new_lb, new_ub)
    end
    return t
end

# Compute the patched signature for `m` with every known placeholder
# replaced by its currently-bound real type. Returns `nothing` if no
# substitution was performed.
function _patched_sig(m::Method)
    placeholders = incomplete_placeholders_in(m.sig)
    isempty(placeholders) && return nothing
    sig = m.sig
    for p in placeholders
        ref = get(incomplete_placeholders, p, nothing)
        ref === nothing && continue
        isdefined(ref.mod, ref.name) || continue
        real = getglobal(ref.mod, ref.name)
        real isa Type || continue
        sig = _subst_placeholder(sig, p, real)
    end
    return sig
end

# Rewrite `m.sig` to `new_sig` in place, re-keying the method-table entry.
# Wraps the `jl_method_resig` C primitive. The new signature must not
# itself contain any placeholders (assert under the registry lock).
function patch_method_sig!(m::Method, @nospecialize(new_sig))
    new_sig isa Type || throw(ArgumentError("patch_method_sig!: new_sig must be a Type"))
    isempty(incomplete_placeholders_in(new_sig)) ||
        throw(ArgumentError("patch_method_sig!: new_sig still contains placeholders"))
    ccall(:jl_method_resig, Cvoid, (Any, Any), m, new_sig)
    return m
end

# Register a zero-arg thunk to run when `(mod, name)` resolves. The thunk
# is invoked by `incomplete_resolve!` after dependent methods are patched.
# Used by struct / abstract / primitive / const-alias lowering to finalise
# the placeholder's body or rebind a const alias once `name` binds.
#
# If the thunk itself raises a recoverable `UndefVarError(:sym2)`, it is
# re-registered against `:sym2` so it runs when `sym2` later binds. Any
# other exception propagates to the caller of `incomplete_resolve!`.
function incomplete_register_finalizer!(mod::Module, name::Symbol, thunk,
                                        srcfile::Symbol = :none,
                                        srcline::Integer = 0,
                                        defined_name::Union{Symbol,Nothing} = nothing)
    ref = get_or_create_incomplete_ref(mod, name, srcfile, srcline)
    pf = PendingFinalizer(thunk, defined_name, srcfile, Int32(srcline))
    @lock incomplete_lock push!(ref.pending_finalizers, pf)
    return ref
end

# Process the `IncompleteRef` for `(mod, name)`: if `name` is now bound to a
# `Type`, patch every dependent method by substituting the placeholder for
# the real type and run any registered finalizer thunks. Methods whose
# patched sig still references some *other* unresolved placeholder are
# left on the dependents list to be retried when that other ref resolves.
# Finalizers that themselves raise a recoverable `UndefVarError(:sym2)`
# are re-registered against `:sym2`. The ref is dropped from the registry
# once its dependents and pending finalizers are both empty.
function incomplete_resolve!(mod::Module, name::Symbol)
    ref = find_incomplete_ref(mod, name)
    ref === nothing && return nothing
    isdefined(mod, name) || return nothing
    # Patch dependent methods if a placeholder was materialised.
    still_pending = Any[]
    if ref.placeholder !== nothing
        real = getglobal(mod, name)
        if real isa Type
            # Snapshot dependents under the lock; patching may call back into
            # the registry via `_patched_sig` so we must not hold it during
            # `ccall`.
            deps = @lock incomplete_lock copy(ref.dependents)
            for dep in deps
                dep isa Method || continue
                patched = _patched_sig(dep)
                patched === nothing && continue
                if isempty(incomplete_placeholders_in(patched))
                    patch_method_sig!(dep, patched)
                else
                    push!(still_pending, dep)
                end
            end
        else
            # Bound to a non-Type: cannot patch placeholder-using methods.
            # Keep dependents around in case the user replaces the binding
            # with a real Type later (rare; handled gracefully).
            still_pending = @lock incomplete_lock copy(ref.dependents)
        end
    end
    # Drain finalizer thunks (struct/abstract/primitive body finalization,
    # const-alias rebind). Run outside the registry lock — thunks may
    # allocate, evaluate user code, and re-register via
    # `incomplete_register_finalizer!`.
    thunks = @lock incomplete_lock begin
        ts = copy(ref.pending_finalizers)
        empty!(ref.pending_finalizers)
        ts
    end
    for pf in thunks
        try
            pf.thunk()
        catch e
            if e isa UndefVarError && incomplete_can_defer(e, mod) && e.var !== name
                incomplete_register_finalizer!(mod, e.var, pf.thunk,
                                               pf.srcfile, pf.srcline,
                                               pf.defined_name)
            else
                rethrow()
            end
        end
    end
    @lock incomplete_lock begin
        empty!(ref.dependents)
        append!(ref.dependents, still_pending)
        if isempty(ref.dependents) && isempty(ref.pending_finalizers)
            ph = ref.placeholder
            ph === nothing || delete!(incomplete_placeholders, ph)
            remove_incomplete_ref!(mod, name)
        end
    end
    return nothing
end

# Return `true` if `e` is an `UndefVarError` raised against `mod` for a
# name that is neither bound locally nor reachable via `using`/`import`
# (i.e. `isdefined(mod, sym)` is `false`). Requiring `e.scope === mod`
# excludes:
#  - errors that escaped from a nested `eval` in a different module,
#  - errors about local variables or static parameters (whose scope is
#    the symbol `:local` resp. `:static_parameter`),
#  - user-constructed `UndefVarError`s with no scope set.
incomplete_can_defer(e::UndefVarError, mod::Module) =
    isdefined(e, :scope) && e.scope === mod && !isdefined(mod, e.var)
incomplete_can_defer(@nospecialize(e), ::Module) = false

# 3-arg form additionally requires `ast` to be a definitional top-level form
# (`struct`, `function`, method `=`, `abstract`, `primitive`, `macro`, `const`,
# or a `macrocall` wrapping one). The form check excludes non-definitional
# evals (e.g. `Core.eval(M, :(f()))`) whose runtime `UndefVarError` would
# otherwise be silently deferred.
incomplete_can_defer(e::UndefVarError, mod::Module, @nospecialize(ast)) =
    incomplete_can_defer(e, mod) && incomplete_definitional_form(ast)

# Definitional top-level forms eligible for deferral: forms whose missing
# name appears in a *type position* of a definition (method signature, field
# type, supertype, or const type-alias RHS). Forms whose RHS is *executable
# code* (call/operator/comprehension/...) are excluded — a reference to a
# not-yet-defined name in executable position is a programming error and
# must surface eagerly.
function incomplete_definitional_form(@nospecialize(ast))
    ast isa Expr || return false
    h = ast.head
    if h === :struct || h === :abstract || h === :primitive ||
       h === :function || h === :macro
        return true
    elseif h === :(=) && length(ast.args) >= 1
        # Short-form method definition only: `f(args) = body` /
        # `f(args) where {T} = body`. Plain assignments `x = expr` are
        # executable and excluded.
        lhs = ast.args[1]
        return lhs isa Expr && (lhs.head === :call || lhs.head === :where)
    elseif h === :const && length(ast.args) >= 1
        # `const B = RHS`: defer only if RHS is a pure type expression
        # (bare name, qualified name, `T{…}`, `T where …`, `<:T`). A call,
        # arithmetic, or any value-producing RHS is runtime code; an
        # UndefVarError there is a real bug and must surface eagerly.
        inner = ast.args[1]
        inner isa Expr && inner.head === :(=) && length(inner.args) == 2 || return false
        return _is_type_position_expr(inner.args[2])
    elseif h === :macrocall
        # Unwrap to check the macro's argument (skip name + linenumbernode and
        # any other non-`Expr` macro arguments such as docstrings).
        for i in 2:length(ast.args)
            arg = ast.args[i]
            arg isa Expr || continue
            return incomplete_definitional_form(arg)
        end
    end
    return false
end

# Conservative check: does `e` look like a pure *type* expression (no
# value-producing computation)? Accepts: bare `Symbol`s, qualified names
# (`Mod.A`), parametric type applications (`A{…}`), `T where …`, and
# `<:T`/`>:T`. Rejects everything else, including `:call` — `A()` is a
# constructor *call*, not a type expression.
function _is_type_position_expr(@nospecialize(e))
    e isa Symbol && return true
    e isa Expr || return false
    h = e.head
    if h === :(.)
        return true
    elseif h === :curly
        return all(_is_type_position_expr, e.args)
    elseif h === :where
        return !isempty(e.args) && _is_type_position_expr(e.args[1])
    elseif (h === :(<:) || h === :(>:)) && length(e.args) == 1
        return _is_type_position_expr(e.args[1])
    end
    return false
end

# ---------------------------------------------------------------------------
# Rewrite-and-retry: lazy placeholder admission for method definitions.
#
# When a top-level method def's first eval raises `UndefVarError(:A)`, we
# rewrite occurrences of `:A` in the *signature's* type positions to
# `incomplete_typeref(mod, :A, file, line)` and re-eval. The wrapper
# materialises a placeholder `DataType`, so the method is admitted into the
# table in a dormant state (visible to `methods(f)`, not callable). The
# `incomplete_drain_ready` binding hook later swaps the placeholder for the
# real type via `jl_method_resig`.
#
# Cost: zero for method defs that don't hit `UndefVarError` (the common
# case). One AST walk + retry per missing symbol when they do. The body of
# the method is never rewritten — only the signature's type positions are
# touched, so the wrapper is invisible at call time.
#
# Only method-def forms are rewritten (struct field types are phase 9). For
# any other definitional form, the rewrite path bails out and we fall back
# to the original AST-defer queue.

# Return the AST sub-expression that holds the *signature* of `ast`, or
# `nothing` if `ast` is not a method-def form supported by the rewriter.
# `:macro` is included because lowering routes `macro m(x::A) body end` to
# `function (var"@m")(__source__, __module__, x::A) body end`, so the
# signature rewriter handles macro arg types the same way as method args.
function _incomplete_sig_of(ast::Expr)
    h = ast.head
    if h === :function || h === :(=) || h === :macro
        length(ast.args) >= 1 || return nothing
        return ast.args[1]
    end
    return nothing
end

# Return the name introduced by a top-level definitional form (`struct`,
# `abstract type`, `primitive type`, `const`, `macro`), or `nothing` for
# forms that don't bind a single user-visible name in the enclosing module
# (in particular function/method defs — those surface through their owning
# `Method` on the dependents list, so the diagnostic uses the method's
# name there).
function _incomplete_defined_name(ast::Expr)
    h = ast.head
    if h === :struct
        length(ast.args) >= 2 || return nothing
        n = ast.args[2]
        n isa Symbol && return n
        n isa Expr && n.head === :(<:) && length(n.args) >= 1 && (n = n.args[1])
        n isa Expr && n.head === :curly && length(n.args) >= 1 && (n = n.args[1])
        return n isa Symbol ? n : nothing
    elseif h === :abstract || h === :primitive
        length(ast.args) >= 1 || return nothing
        n = ast.args[1]
        n isa Symbol && return n
        n isa Expr && n.head === :(<:) && length(n.args) >= 1 && (n = n.args[1])
        n isa Expr && n.head === :curly && length(n.args) >= 1 && (n = n.args[1])
        return n isa Symbol ? n : nothing
    elseif h === :const
        length(ast.args) >= 1 || return nothing
        a = ast.args[1]
        if a isa Expr && a.head === :(=) && length(a.args) >= 1
            lhs = a.args[1]
            lhs isa Symbol && return lhs
            lhs isa Expr && lhs.head === :(::) && length(lhs.args) >= 1 && lhs.args[1] isa Symbol && return lhs.args[1]
        end
        return nothing
    elseif h === :macro
        length(ast.args) >= 1 || return nothing
        sig = ast.args[1]
        sig isa Expr && sig.head === :call && length(sig.args) >= 1 && sig.args[1] isa Symbol && return sig.args[1]
        return nothing
    end
    return nothing
end

# Construct the AST `incomplete_typeref(mod, sym, file, line)`. The function
# value is embedded directly so the call doesn't depend on `Base` being
# accessible by name in `mod` (e.g. `baremodule`s without `using Base`).
function _make_typeref_call(mod::Module, sym::Symbol, file::Symbol, line::Int32)
    return Expr(:call, incomplete_typeref, mod, QuoteNode(sym), QuoteNode(file), line)
end

# Mutate `sig` (a signature sub-AST) in place: wherever `sym` appears in a
# type position (RHS of `::`, RHS of `<:`/`>:`, parameter of `curly`, body of
# `where`, etc.) wrap it with a call to `incomplete_typeref`. Skips
# binding-position occurrences (LHS of `::`/`<:`, function name, typevar
# names). Returns `true` if any wrap happened.
function _incomplete_wrap_sig_typerefs!(sig, sym::Symbol, mod::Module,
                                        file::Symbol, line::Int32)
    sig isa Expr || return false
    changed = false
    h = sig.head
    if h === :(::)
        # 2-arg ::: args[1]=var name (binding), args[2]=type
        # 1-arg ::: args[1]=type
        tyidx = length(sig.args) == 2 ? 2 :
                length(sig.args) == 1 ? 1 : 0
        if tyidx != 0
            new_ty, c = _wrap_type_expr(sig.args[tyidx], sym, mod, file, line)
            if c
                sig.args[tyidx] = new_ty
                changed = true
            end
        end
    elseif (h === :(<:) || h === :(>:)) && length(sig.args) == 2
        # `T<:Bound`: args[1]=typevar name (binding), args[2]=bound type
        new_ty, c = _wrap_type_expr(sig.args[2], sym, mod, file, line)
        if c
            sig.args[2] = new_ty
            changed = true
        end
    end
    # Recurse into all sub-Exprs to find nested type positions
    # (`call`/`where`/`kw`/`tuple`/`parameters` wrappers).
    for a in sig.args
        a isa Expr && (changed |= _incomplete_wrap_sig_typerefs!(a, sym, mod, file, line))
    end
    return changed
end

# Wrap a *type expression* `ty`. If `ty` is the bare symbol `sym`, replace
# with the typeref call. If `ty` is a `curly`/`where`/`<:`/`>:` form,
# descend into its type-position children. Returns `(new_ty, changed)`.
function _wrap_type_expr(@nospecialize(ty), sym::Symbol, mod::Module,
                         file::Symbol, line::Int32)
    if ty === sym
        return (_make_typeref_call(mod, sym, file, line), true)
    elseif ty isa Expr
        h = ty.head
        if h === :curly
            # `Vector{A}` / `A{T}` — every arg is a type position
            changed = false
            new_args = Vector{Any}(undef, length(ty.args))
            for i in 1:length(ty.args)
                np, c = _wrap_type_expr(ty.args[i], sym, mod, file, line)
                new_args[i] = np
                changed |= c
            end
            changed && return (Expr(:curly, new_args...), true)
            return (ty, false)
        elseif h === :where
            # `T where {B}` — args[1] is body (type), args[2:end] are bounds
            changed = false
            new_args = copy(ty.args)
            nb, cb = _wrap_type_expr(new_args[1], sym, mod, file, line)
            if cb
                new_args[1] = nb
                changed = true
            end
            for i in 2:length(new_args)
                bnd = new_args[i]
                if bnd isa Expr && (bnd.head === :(<:) || bnd.head === :(>:)) &&
                   length(bnd.args) == 2
                    nb2, c2 = _wrap_type_expr(bnd.args[2], sym, mod, file, line)
                    if c2
                        bnd.args[2] = nb2
                        changed = true
                    end
                end
            end
            changed && return (Expr(:where, new_args...), true)
            return (ty, false)
        elseif (h === :(<:) || h === :(>:)) && length(ty.args) == 1
            # `<:A` as a type expression (rare)
            np, c = _wrap_type_expr(ty.args[1], sym, mod, file, line)
            c && return (Expr(h, np), true)
            return (ty, false)
        end
    end
    # Qualified names like `Mod.A` and any other shape: leave alone.
    return (ty, false)
end

# Extract the function-name sub-expression from a method-def signature.
# Returns a `Symbol`, an `Expr` (for `Mod.f` / `(Mod.f){T}`), or `nothing`.
function _incomplete_funcname_of(@nospecialize(sig))
    sig isa Symbol && return sig
    sig isa Expr || return nothing
    h = sig.head
    if h === :call
        return _incomplete_funcname_of(sig.args[1])
    elseif h === :where
        return _incomplete_funcname_of(sig.args[1])
    elseif h === :(::)
        # `f(...)::R = ...` — function inside the ::
        return _incomplete_funcname_of(sig.args[1])
    elseif h === :curly
        return _incomplete_funcname_of(sig.args[1])
    elseif h === :(.)
        return sig
    end
    return nothing
end

# After a successful rewrite-and-eval, register every method of the defined
# function that contains placeholders so the binding-event drain can find
# them. Best-effort: silently skips on any resolution failure.
_incomplete_lookup_global(mod::Module, sym::Symbol) =
    isdefined(mod, sym) ? getglobal(mod, sym) : nothing
function _incomplete_register_new_methods!(mod::Module, ast::Expr)
    sig = _incomplete_sig_of(ast)
    sig === nothing && return nothing
    fname = _incomplete_funcname_of(sig)
    fname === nothing && return nothing
    # Macro defs bind their method on `var"@<name>"`, not `<name>`.
    if ast.head === :macro && fname isa Symbol
        fname = Symbol("@", fname)
    end
    # `Core.eval(mod, ast)` above bumped the global world counter, but the
    # current task's `world_age` still points at the world before the eval.
    # Reading `mod.fname` and `methods(f)` at the stale age trips the strict
    # world-age binding-access warning. Route through `invokelatest` so the
    # reads execute in a world where the new method exists.
    f = try
        if fname isa Symbol
            invokelatest(_incomplete_lookup_global, mod, fname)
        else
            invokelatest(Core.eval, mod, fname)
        end
    catch
        nothing
    end
    f === nothing && return nothing
    ms = try invokelatest(methods, f) catch; nothing end
    ms === nothing && return nothing
    for m in ms
        m isa Method && incomplete_register_method!(m)
    end
    return nothing
end

# Structural copy of an AST that preserves shared leaf values (Modules,
# typed function objects embedded by `_make_typeref_call`, etc. — which
# `deepcopy` refuses to traverse). New `Expr` nodes at every internal level
# so subsequent in-place mutation of the copy doesn't leak back to `ast`.
_clone_ast(e::Expr) = Expr(e.head, Any[_clone_ast(a) for a in e.args]...)
_clone_ast(@nospecialize(x)) = x

# Try to admit `ast` as a dormant method by rewriting unbound type names
# to placeholder lookups and re-evaling. Returns `true` if eval succeeded
# (dormant or fully concrete), `false` if the rewrite couldn't help (caller
# should fall through to AST-defer).
function _incomplete_try_rewrite_and_eval!(mod::Module, sym::Symbol, ast::Expr)
    _incomplete_sig_of(ast) === nothing && return false
    file, line = incomplete_srcloc(ast)
    new_ast = _clone_ast(ast)
    wrapped = Symbol[sym]
    cur_sym = sym
    # Bounded retry: at most one rewrite per distinct missing symbol.
    while true
        sig = _incomplete_sig_of(new_ast)
        sig === nothing && return false
        _incomplete_wrap_sig_typerefs!(sig, cur_sym, mod, file, line) || return false
        try
            Core.eval(mod, new_ast)
            _incomplete_register_new_methods!(mod, new_ast)
            return true
        catch e
            if e isa UndefVarError && incomplete_can_defer(e, mod) &&
               !(e.var in wrapped)
                push!(wrapped, e.var)
                cur_sym = e.var
                continue
            end
            return false
        end
    end
end

# Register `ast` for re-evaluation once `e.var` becomes defined in `mod`.
# Before queuing, try the placeholder-admission rewrite path: if `ast` is a
# method def and we can wrap the missing name in its signature, the method
# is admitted dormant and we skip the queue entirely. Otherwise register a
# replay thunk on the missing symbol's `IncompleteRef`: when the symbol
# binds, `incomplete_resolve!` runs the thunk which re-evals `ast` in `mod`.
# Non-`UndefVarError` exceptions from the replay are wrapped in `LoadError`
# attributed to the original definition site; recoverable `UndefVarError`s
# propagate so `incomplete_resolve!` re-registers the thunk on the new
# missing name.
function incomplete_defer(mod::Module, e::UndefVarError, ast::Expr)
    _incomplete_try_rewrite_and_eval!(mod, e.var, ast) && return nothing
    file, line = incomplete_srcloc(ast)
    defined_name = _incomplete_defined_name(ast)
    thunk = let mod=mod, ast=ast, file=file, line=line
        function()
            try
                Core.eval(mod, ast)
            catch err
                err isa UndefVarError && rethrow()
                throw(LoadError(String(file), Int(line), err))
            end
        end
    end
    incomplete_register_finalizer!(mod, e.var, thunk, file, line, defined_name)
    return nothing
end

# Convenience helper for Julia callers that need to do top-level evaluation
# of a surface AST with the same defer-on-UndefVarError semantics the C
# toplevel evaluator provides. Returns `true` if `exc` was deferred and the
# caller should swallow it, `false` otherwise (caller should re-raise).
function incomplete_try_defer(mod::Module, @nospecialize(exc), @nospecialize(orig))
    exc isa UndefVarError || return false
    orig isa Expr || return false
    # Peel `:toplevel` / `:block` wrappers that may surround a single
    # definitional form. The parser wraps REPL input in `:toplevel`, and
    # surface transforms like `softscope` add a `:block`; both are no-ops
    # around a method/type/const definition and obscure the head check.
    inner = _incomplete_peel_wrapper(orig::Expr)
    inner isa Expr || return false
    incomplete_can_defer(exc, mod, inner) || return false
    incomplete_defer(mod, exc, inner)
    return true
end

# Strip `:toplevel` / `:block` wrappers that contain a single non-trivial
# child (ignoring `LineNumberNode`s and `:softscope` markers). Returns the
# inner expression, or `ast` unchanged if no single inner form is present.
function _incomplete_peel_wrapper(@nospecialize(ast))
    cur = ast
    while cur isa Expr && (cur.head === :toplevel || cur.head === :block)
        inner = nothing
        for arg in (cur::Expr).args
            arg isa LineNumberNode && continue
            if arg isa Expr && arg.head === :softscope
                continue
            end
            if inner === nothing
                inner = arg
            else
                # More than one non-trivial child: not a simple wrapper.
                return cur
            end
        end
        inner === nothing && return cur
        cur = inner
    end
    return cur
end

# Drain hooks invoked from the C runtime.
# Resolve every registered `IncompleteRef(mod, name)` whose `name` is now
# defined. `incomplete_resolve!` patches dependent methods via
# `jl_method_resig` and runs any registered finalizer thunks (which
# re-evaluate deferred struct/abstract/primitive/const-alias/macro defs).
#
# Routed through `invokelatest` so the body executes at the current world
# counter rather than the caller's (potentially stale) task world. The C
# binding-event hook fires immediately after a top-level definition that
# bumped the world; reads like `isdefined(mod, name)` and `getglobal(mod,
# name)` on the freshly bound symbol would otherwise trip the strict
# world-age binding-access warning (and will error in a future release).
function _incomplete_drain_ready_impl(mod::Module)
    ready_syms = @lock incomplete_lock begin
        st = get(incomplete_refs, mod, nothing)
        isnothing(st) ? Symbol[] : Symbol[s for s in keys(st) if isdefined(mod, s)]
    end
    for sym in ready_syms
        incomplete_resolve!(mod, sym)
    end
    return nothing
end

function incomplete_drain_ready(mod::Module)
    invokelatest(_incomplete_drain_ready_impl, mod)
    return nothing
end

# Called from the C runtime at module close. Throws `IncompleteTypeError`
# listing every `IncompleteRef(mod, name)` whose `name` never bound, with
# the source location of the first forward reference and a summary of how
# many methods / pending definitions are waiting on each name.
function incomplete_finalize(mod::Module)
    rst = get(incomplete_refs, mod, nothing)
    isnothing(rst) && return nothing
    entries = IncompleteTypeEntry[]
    for (name, ref) in rst
        isdefined(mod, name) && continue
        n_methods = count(d -> d isa Method, ref.dependents)
        push!(entries, IncompleteTypeEntry(name, ref.srcfile, ref.srcline,
                                           n_methods,
                                           length(ref.pending_finalizers)))
    end
    # Drop the module's registry slot; the placeholders themselves stay
    # alive only via the dependent methods (which are about to become
    # unreachable once the module is GC'd).
    for (_, ref) in rst
        ph = ref.placeholder
        ph === nothing || delete!(incomplete_placeholders, ph)
    end
    delete!(incomplete_refs, mod)
    isempty(entries) && return nothing
    throw(IncompleteTypeError(mod, entries))
end

function showerror(io::IO, e::IncompleteTypeError)
    n = length(e.entries)
    print(io, "IncompleteTypeError: module ", e.mod, " was closed with ",
          n, " unresolved forward reference", n == 1 ? "" : "s", ":")
    for ent in e.entries
        print(io, "\n  ", ent.name)
        if ent.srcfile !== :none || ent.srcline != 0
            print(io, " — first referenced at ", ent.srcfile, ":", ent.srcline)
        end
        waiting = String[]
        ent.n_methods > 0 && push!(waiting,
            string(ent.n_methods, " method", ent.n_methods == 1 ? "" : "s"))
        ent.n_pending > 0 && push!(waiting,
            string(ent.n_pending, " pending definition", ent.n_pending == 1 ? "" : "s"))
        isempty(waiting) || print(io, " (", join(waiting, ", "), " waiting)")
    end
    print(io, "\nSuggestion: define or import the missing name(s) at the module's top level.")
    return nothing
end

# First `(file, line)` pair found in `ast`, falling back to `(:none, 0)` if the
# AST carries no `LineNumberNode` (in practice unreachable for parsed top-level
# forms, but synthetic ASTs built by `Core.eval` callers may lack one).
function incomplete_srcloc(ast::Expr)
    for arg in ast.args
        if arg isa LineNumberNode
            return (isnothing(arg.file) ? :none : arg.file::Symbol, Int32(arg.line))
        end
        if arg isa Expr
            f, l = incomplete_srcloc(arg)
            f === :none && l == 0 || return (f, l)
        end
    end
    return (:none, Int32(0))
end
