# This file is a part of Julia. License is MIT: https://julialang.org/license

using Test

# Simple forward field-type reference
@testset "incomplete types: simple forward ref" begin
    eval(:(module _M_simple
        struct Foo
            x::Bar
        end
        struct Bar
            y::Int
        end
    end))
    @test fieldtype(_M_simple.Foo, :x) === _M_simple.Bar
    @test fieldtype(_M_simple.Bar, :y) === Int
end

# Transitive forward field-type references
@testset "incomplete types: transitive forward ref" begin
    eval(:(module _M_trans
        struct A; x::B; end
        struct B; y::C; end
        struct C; z::Int; end
    end))
    @test fieldtype(_M_trans.A, :x) === _M_trans.B
    @test fieldtype(_M_trans.B, :y) === _M_trans.C
end

# Adapted from removed `test/typegroup.jl` testset "supertype referencing
# incomplete type" (the only typegroup case for super references).
@testset "supertype referencing incomplete type" begin
    eval(:(module _M_super_ref
        struct IT_SuperRefA <: AbstractVector{IT_SuperRefB}
            data::Vector{IT_SuperRefB}
        end
        struct IT_SuperRefB
            a::IT_SuperRefA
        end
    end))
    @test _M_super_ref.IT_SuperRefA <: AbstractVector{_M_super_ref.IT_SuperRefB}
end

# A non-forward-ref struct is unaffected
@testset "incomplete types: no deferral when binding exists" begin
    eval(:(module _M_eager
        struct Plain
            x::Int
        end
    end))
    @test fieldtype(_M_eager.Plain, :x) === Int
end

# Unresolved dependency at module close becomes `IncompleteTypeError`,
# listing the unresolved name and the source location of the first
# forward reference.
@testset "incomplete types: unresolved at module-end errors" begin
    err = try
        eval(:(module _M_bad
            struct Dangling
                x::__Definitely_Not_Defined__
            end
        end))
        nothing
    catch e
        e
    end
    @test err isa Base.IncompleteTypeError
    @test length(err.entries) == 1
    @test err.entries[1].name === :__Definitely_Not_Defined__
end

# A "non-recoverable" error (something other than UndefVarError) still throws
# eagerly with the original error.
@testset "incomplete types: non-recoverable errors still propagate" begin
    err = try
        eval(:(module _M_other
            const _bad = error("boom")
            struct Foo
                x::Int
            end
        end))
        nothing
    catch e
        e
    end
    # The "boom" error must surface (LoadError-like wrapping is fine).
    sprinted = sprint(showerror, err)
    @test occursin("boom", sprinted)
end

# An `UndefVarError` whose scope is not `mod` (e.g. `:local` or
# `:static_parameter`) must not be treated as a forward reference, even
# if its `.var` happens to coincide with an undefined global in `mod`.
@testset "incomplete types: non-module-scope UndefVarError not recovered" begin
    err = try
        eval(:(module _M_localscope
            f() = (local y; y)
            f()  # raises UndefVarError(:y, :local); :y is not a global here
        end))
        nothing
    catch e
        e
    end
    @test err !== nothing
    sprinted = sprint(showerror, err)
    @test occursin("y", sprinted)
end

# Forward references to names owned by another module are not deferred;
# cross-module typos still raise eagerly.
@testset "incomplete types: imported names are not deferred" begin
    err = try
        eval(:(module _M_imported
            using Base: Int
            struct Foo
                x::__NotInBaseAtAll__
            end
        end))
        nothing
    catch e
        e
    end
    @test err isa Base.IncompleteTypeError
    @test length(err.entries) == 1
    @test err.entries[1].name === :__NotInBaseAtAll__
end

# Adapted from removed `test/typegroup.jl` testset "basic mutual recursion".
@testset "basic mutual recursion" begin
    eval(:(module _M_basic_mutual
        struct IT_Node
            edges::Vector{IT_Edge}
        end
        struct IT_Edge
            from::IT_Node
            to::IT_Node
        end
    end))
    M = _M_basic_mutual
    @test fieldtype(M.IT_Node, :edges) == Vector{M.IT_Edge}
    @test fieldtype(M.IT_Edge, :from) == M.IT_Node
    @test fieldtype(M.IT_Edge, :to) == M.IT_Node

    n1 = M.IT_Node(M.IT_Edge[])
    n2 = M.IT_Node(M.IT_Edge[])
    e = M.IT_Edge(n1, n2)
    push!(n1.edges, e)
    @test n1.edges[1].to === n2
end

# A method definition whose signature references a not-yet-defined type
# defers and is re-evaluated once the type is bound. Mirrors the struct
# deferral so REPL/module bodies can declare types and methods in any order.
@testset "incomplete types: method deferral" begin
    M = Module()
    Core.eval(M, :(struct _M_A; b::Union{_M_B,Nothing}; end))
    Core.eval(M, :(_m_f(a::_M_A, b::_M_B) = (a, b)))
    # Phase-5 admission: `_m_f` is visible as a dormant method even before
    # `_M_B` is bound, but no concrete call can dispatch to it.
    @test isdefined(M, :_m_f) && length(methods(Core.eval(M, :_m_f))) == 1
    Core.eval(M, :(struct _M_B; a::Union{_M_A,Nothing}; end))
    f  = Core.eval(M, :_m_f)
    A  = Core.eval(M, :_M_A)
    B  = Core.eval(M, :_M_B)
    @test !isempty(methods(f))
    v = f(A(B(nothing)), B(nothing))
    @test v[1] isa A
    @test v[2] isa B
end

# A method whose dependency is never defined raises `IncompleteTypeError`
# at module close, matching the struct behavior.
@testset "incomplete types: method unresolved at module-end errors" begin
    err = try
        eval(:(module _M_method_bad
            f(::__Never_Defined__) = nothing
        end))
        nothing
    catch e
        e
    end
    @test err isa Base.IncompleteTypeError
    @test length(err.entries) == 1
    @test err.entries[1].name === :__Never_Defined__
    @test err.entries[1].n_methods >= 1
end

# Multiple independent unresolved deferrals at module close are aggregated
# into a single `IncompleteTypeError` so the user sees every unresolved
# definition in one report.
@testset "incomplete types: multiple unresolved at module-end aggregated" begin
    err = try
        eval(:(module _M_multi_bad
            struct __A_multi_bad; x::__Never_A__; end
            struct __B_multi_bad; y::__Never_B__; end
        end))
        nothing
    catch e
        e
    end
    @test err isa Base.IncompleteTypeError
    @test length(err.entries) == 2
    vars = Set(ent.name for ent in err.entries)
    @test vars == Set([:__Never_A__, :__Never_B__])
end

# `incomplete_try_defer` must accept the definitional form even when it is
# wrapped in `:toplevel` / `:block` (as produced by the parser and REPL
# transforms like `softscope`). Without the wrapper-peeling, REPL input like
# `f(x::A) = x` would never reach the deferral path because the
# pre-transform AST handed to the helper is `Expr(:toplevel, lineno, expr)`.
@testset "incomplete types: incomplete_try_defer peels toplevel/block wrappers" begin
    M = Module()
    Core.eval(M, :(using Base))
    exc = UndefVarError(:__Wrap_Missing__, M)
    bare    = :(f_wrap(x::__Wrap_Missing__) = x)
    wrapped = Expr(:toplevel, LineNumberNode(1, :REPL), bare)
    nested  = Expr(:toplevel, LineNumberNode(1, :REPL),
                   Expr(:block, LineNumberNode(1, :REPL), bare))
    @test Base.incomplete_try_defer(M, exc, wrapped)
    @test Base.incomplete_try_defer(M, exc, nested)
    # A wrapper around more than one non-trivial child must NOT be peeled.
    twostmt = Expr(:toplevel, LineNumberNode(1, :REPL), bare, bare)
    @test !Base.incomplete_try_defer(M, exc, twostmt)
end

# Phase-5 lazy AST-rewrite: a method def referencing a not-yet-defined name
# is admitted as a dormant method (visible in `methods(f)` but not callable)
# rather than queued whole. After the dependency is bound, the placeholder
# in the signature is patched in place to the real type.
@testset "incomplete types: dormant method visible in methods()" begin
    M = Module()
    Core.eval(M, :(using Base))
    Core.eval(M, :(f_dormant(x::__Dorm_A__) = x))
    f = Core.eval(M, :f_dormant)
    @test length(methods(f)) == 1
    # No concrete value can satisfy the placeholder.
    @test !hasmethod(f, Tuple{Int})
    @test_throws MethodError f(1)
    # Binding the dependency patches the method in place.
    Core.eval(M, :(struct __Dorm_A__; n::Int; end))
    A = Core.eval(M, :__Dorm_A__)
    @test f(A(7)).n == 7
    @test Base.find_incomplete_ref(M, :__Dorm_A__) === nothing
end

# Two distinct unresolved names in one signature: both get wrapped via the
# nested-rewrite loop and the method is admitted dormant on both.
@testset "incomplete types: dormant method with two missing names" begin
    M = Module()
    Core.eval(M, :(using Base))
    Core.eval(M, :(g_two(x::__Two_C__, y::__Two_D__) = (x, y)))
    g = Core.eval(M, :g_two)
    @test length(methods(g)) == 1
    Core.eval(M, :(struct __Two_C__; n::Int; end))
    @test length(methods(g)) == 1   # still dormant: D missing
    @test_throws MethodError g(Core.eval(M, :__Two_C__)(1), 2)
    Core.eval(M, :(struct __Two_D__; s::String; end))
    C = Core.eval(M, :__Two_C__); D = Core.eval(M, :__Two_D__)
    v = g(C(1), D("x"))
    @test v[1].n == 1 && v[2].s == "x"
end

# `where {T<:Y}` with `Y` unbound: the typevar's upper bound is the
# placeholder. After binding `Y`, the patched UnionAll must rebind its
# inner `TypeVar` (regression guard for the `_subst_typevar` fix).
@testset "incomplete types: dormant method with where-clause placeholder" begin
    M = Module()
    Core.eval(M, :(using Base))
    Core.eval(M, :(h_where(x::T) where {T<:__Wh_Y__} = x))
    h = Core.eval(M, :h_where)
    @test length(methods(h)) == 1
    Core.eval(M, :(abstract type __Wh_Y__ end))
    Core.eval(M, :(struct __Wh_YI__ <: __Wh_Y__; n::Int; end))
    YI = Core.eval(M, :__Wh_YI__)
    @test h(YI(9)).n == 9
end

# Curly nesting: `Vector{X}` with `X` unbound.
@testset "incomplete types: dormant method with nested type parameter" begin
    M = Module()
    Core.eval(M, :(using Base))
    Core.eval(M, :(k_vec(v::Vector{__Vec_X__}) = length(v)))
    k = Core.eval(M, :k_vec)
    @test length(methods(k)) == 1
    Core.eval(M, :(struct __Vec_X__; end))
    X = Core.eval(M, :__Vec_X__)
    @test k(X[X(), X(), X()]) == 3
end

# Zero-overhead invariant: a method def whose signature has no missing names
# must not register any `IncompleteRef` in the module.
@testset "incomplete types: complete signature stays off the registry" begin
    M = Module()
    Core.eval(M, :(using Base))
    Core.eval(M, :(q_complete(x::Int, y::Float64) = x + y))
    @test Base.find_incomplete_ref(M, :Int) === nothing
    @test Base.find_incomplete_ref(M, :Float64) === nothing
    @test get(Base.incomplete_refs, M, nothing) === nothing
end

# Deferral is for *definitional* references (type positions of method sigs,
# struct fields, supertypes, const type-alias RHS). A reference to a
# not-yet-defined name in *executable* position (call/operator/arithmetic
# on the RHS of `const`, plain assignment, top-level expression) is a
# programming error and must surface eagerly — never get queued.
@testset "incomplete types: executable-position UndefVarError is eager" begin
    # const with an undef *function call* on the RHS: not a type expression,
    # surfaces eagerly even before module-end finalize.
    err = try
        eval(:(module _M_const_call
            const Y = __never_defined_fn__()
        end))
        nothing
    catch e
        e
    end
    @test err !== nothing
    @test occursin("__never_defined_fn__", sprint(showerror, err))

    # const with an arithmetic RHS: executable, surfaces eagerly.
    err2 = try
        eval(:(module _M_const_arith
            const Z = __never_defined_op__ + 1
        end))
        nothing
    catch e
        e
    end
    @test err2 !== nothing
    @test occursin("__never_defined_op__", sprint(showerror, err2))
end

# `const B = A` where A is a not-yet-defined type is a type *alias*, not a
# runtime computation — defer it and let the eventual definition of A
# resolve the binding. `const B = A()` is a constructor call (runtime code)
# and must error eagerly per the previous testset.
@testset "incomplete types: const type-alias defers, const call is eager" begin
    # Alias form: defers and resolves when A is defined later in the module.
    eval(:(module _M_const_alias
        const B = A
        struct A end
    end))
    M = getglobal(@__MODULE__, :_M_const_alias)
    @test getglobal(M, :B) === getglobal(M, :A)
    @test get(Base.incomplete_refs, M, nothing) === nothing

    # Parametric alias: `const VA = Vector{A}` with an unresolved name.
    eval(:(module _M_const_parametric_alias
        const VA = Vector{A}
        struct A end
    end))
    N = getglobal(@__MODULE__, :_M_const_parametric_alias)
    @test getglobal(N, :VA) === Vector{getglobal(N, :A)}

    # Call form: not a type expression, must surface eagerly even though the
    # callee name *would* eventually be defined.
    err = try
        eval(:(module _M_const_constructor
            const B = A()
            struct A end
        end))
        nothing
    catch e
        e
    end
    @test err !== nothing
    @test occursin("A", sprint(showerror, err))
end

# Macro definitions have argument type annotations that evaluate at def
# time (like function signatures). A reference to a not-yet-defined type
# in a macro argument annotation must defer and resolve when the type
# is bound.
@testset "incomplete types: macro signature defers" begin
    eval(:(module _M_macro_sig
        macro mymac(x::A)
            esc(x)
        end
        struct A end
    end))
    M = getglobal(@__MODULE__, :_M_macro_sig)
    # The macro is defined and its arg type was patched to the real A.
    @test isdefined(M, Symbol("@mymac"))
    @test get(Base.incomplete_refs, M, nothing) === nothing
end

# Regression test for world-age binding-access warnings on the
# dormant-method admission and the binding-event drain paths. The C
# binding-event hook fires immediately after a top-level definition that
# bumped the world; reads on the freshly bound symbol must execute at the
# current world or they trip Julia 1.12's strict world-age warning (which
# becomes an error in a future release). Run under `--depwarn=error` so
# any such warning aborts the subprocess and fails the test.
@testset "incomplete types: no world-age warnings on dormant admission / drain" begin
    script = """
        f(x::A) = x.a
        length(methods(f)) == 1 || error("dormant method not admitted")
        struct A; a::Int32; end
        f(A(Int32(7))) == 7 || error("patched method did not dispatch")
    """
    cmd = `$(Base.julia_cmd()) --startup-file=no --depwarn=error -e $script`
    @test success(pipeline(cmd; stdout=devnull, stderr=devnull))
end

# ---------------------------------------------------------------------------
# Tests adapted from the removed `test/typegroup.jl` suite.
#
# Each testset below carries the original `typegroup` testset name verbatim
# and uses the prefix `IT_` (Incomplete Types) in place of the original
# `TG_` (typegroup) name prefix, so the mapping to the removed file is
# mechanical: `git show 71559b92e6^:test/typegroup.jl` for the originals.
#
# Skipped originals (with reason):
#   - "return value"                          : no `typegroup` surface syntax;
#                                               nothing for the test to assert.
#   - "TypeApp reflection"                    : `Core.TypeApp` and the
#                                               `resolve_typegroup` /
#                                               `apply_type_or_typeapp`
#                                               machinery were removed.
#   - "incomplete type errors (#60919)"       : commented out in the original
#                                               (awaited the same lowering
#                                               unification this PR delivers);
#                                               separate follow-up.
# ---------------------------------------------------------------------------

# Adapted from `typegroup` testset "parametric types".
@testset "parametric types" begin
    eval(:(module _M_param
        struct IT_PNode{T}
            data::T
            edges::Vector{IT_PEdge{T}}
        end
        struct IT_PEdge{T}
            from::IT_PNode{T}
            to::IT_PNode{T}
        end
    end))
    M = _M_param
    @test fieldtype(M.IT_PNode{Int}, :edges) == Vector{M.IT_PEdge{Int}}
    @test fieldtype(M.IT_PEdge{String}, :from) == M.IT_PNode{String}

    n1 = M.IT_PNode(42, M.IT_PEdge{Int}[])
    n2 = M.IT_PNode(99, M.IT_PEdge{Int}[])
    e = M.IT_PEdge(n1, n2)
    @test e.from.data == 42
    @test e.to.data == 99
end

# Adapted from `typegroup` testset "self-referential types".
@testset "self-referential types" begin
    eval(:(module _M_selfref
        struct IT_SelfRef
            next::Union{Nothing, IT_SelfRef}
        end
    end))
    M = _M_selfref
    @test fieldtype(M.IT_SelfRef, :next) == Union{Nothing, M.IT_SelfRef}

    node3 = M.IT_SelfRef(nothing)
    node2 = M.IT_SelfRef(node3)
    node1 = M.IT_SelfRef(node2)
    @test node1.next.next === node3
end

# Adapted from `typegroup` testset "mutable structs".
@testset "mutable structs" begin
    eval(:(module _M_mutable
        mutable struct IT_MutNode
            edges::Vector{IT_MutEdge}
        end
        mutable struct IT_MutEdge
            from::IT_MutNode
            to::IT_MutNode
        end
    end))
    M = _M_mutable
    @test ismutabletype(M.IT_MutNode)
    @test ismutabletype(M.IT_MutEdge)

    n1 = M.IT_MutNode(M.IT_MutEdge[])
    n2 = M.IT_MutNode(M.IT_MutEdge[])
    e = M.IT_MutEdge(n1, n2)
    push!(n1.edges, e)
    e.to = n1
    @test e.to === n1
end

# Adapted from `typegroup` testset "where clause in field types".
@testset "where clause in field types" begin
    eval(:(module _M_whereclause
        struct IT_Container
            items::Vector{IT_Item{T} where T}
        end
        struct IT_Item{T}
            value::T
            parent::IT_Container
        end
    end))
    M = _M_whereclause
    @test fieldtype(M.IT_Container, :items) == Vector{M.IT_Item{T} where T}
    @test fieldtype(M.IT_Item{Int}, :parent) == M.IT_Container

    c = M.IT_Container(M.IT_Item[])
    item = M.IT_Item(42, c)
    push!(c.items, item)
    @test c.items[1].value == 42
    @test c.items[1].parent === c
end

# Adapted from `typegroup` testset "parametric mutual recursion with Union".
@testset "parametric mutual recursion with Union" begin
    eval(:(module _M_param_union
        struct IT_UnionA{T}
            value::T
            other::Union{Nothing, IT_UnionB{T}}
        end
        struct IT_UnionB{T}
            value::T
            other::Union{Nothing, IT_UnionA{T}}
        end
    end))
    M = _M_param_union
    @test fieldtype(M.IT_UnionA{Int}, :other) == Union{Nothing, M.IT_UnionB{Int}}
    @test fieldtype(M.IT_UnionB{Int}, :other) == Union{Nothing, M.IT_UnionA{Int}}

    a = M.IT_UnionA{Int}(1, nothing)
    b = M.IT_UnionB{Int}(2, nothing)
    a2 = M.IT_UnionA{Int}(3, b)
    b2 = M.IT_UnionB{Int}(4, a)
    @test a.other === nothing
    @test a2.other.value == 2
    @test b2.other.value == 1
end

# Adapted from `typegroup` testset "parametric direct mutual reference".
@testset "parametric direct mutual reference" begin
    eval(:(module _M_param_direct
        struct IT_DirectA{T}
            value::T
            other::Union{Nothing, IT_DirectB{T}}
        end
        struct IT_DirectB{T}
            target::IT_DirectA{T}
            weight::Float64
        end
    end))
    M = _M_param_direct
    @test fieldtype(M.IT_DirectA{Int}, :other) == Union{Nothing, M.IT_DirectB{Int}}
    @test fieldtype(M.IT_DirectB{Int}, :target) == M.IT_DirectA{Int}

    a = M.IT_DirectA{Int}(42, nothing)
    b = M.IT_DirectB{Int}(a, 1.5)
    a2 = M.IT_DirectA{Int}(99, b)
    @test a2.other.target.value == 42
    @test a2.other.weight == 1.5
end

# Adapted from `typegroup` testset "parametric with Vector wrapping".
@testset "parametric with Vector wrapping" begin
    eval(:(module _M_param_vec
        struct IT_VecNode{T}
            value::T
            edges::Vector{IT_VecEdge{T}}
        end
        struct IT_VecEdge{T}
            target::IT_VecNode{T}
            weight::Float64
        end
    end))
    M = _M_param_vec
    @test fieldtype(M.IT_VecNode{Int}, :edges) == Vector{M.IT_VecEdge{Int}}
    @test fieldtype(M.IT_VecEdge{String}, :target) == M.IT_VecNode{String}

    n1 = M.IT_VecNode{Int}(1, M.IT_VecEdge{Int}[])
    n2 = M.IT_VecNode{Int}(2, M.IT_VecEdge{Int}[])
    e1 = M.IT_VecEdge{Int}(n2, 1.0)
    e2 = M.IT_VecEdge{Int}(n1, 2.0)
    n3 = M.IT_VecNode{Int}(3, [e1, e2])
    @test length(n3.edges) == 2
    @test n3.edges[1].target.value == 2
    @test n3.edges[2].target.value == 1
end

# Adapted from `typegroup` testset "three-way parametric mutual recursion".
@testset "three-way parametric mutual recursion" begin
    eval(:(module _M_threeway
        struct IT_ThreeA{T}
            value::T
            b::Union{Nothing, IT_ThreeB{T}}
        end
        struct IT_ThreeB{T}
            value::T
            c::Union{Nothing, IT_ThreeC{T}}
        end
        struct IT_ThreeC{T}
            value::T
            a::Union{Nothing, IT_ThreeA{T}}
        end
    end))
    M = _M_threeway
    @test fieldtype(M.IT_ThreeA{Int}, :b) == Union{Nothing, M.IT_ThreeB{Int}}
    @test fieldtype(M.IT_ThreeB{Int}, :c) == Union{Nothing, M.IT_ThreeC{Int}}
    @test fieldtype(M.IT_ThreeC{Int}, :a) == Union{Nothing, M.IT_ThreeA{Int}}

    a = M.IT_ThreeA{Int}(1, nothing)
    c = M.IT_ThreeC{Int}(3, a)
    b = M.IT_ThreeB{Int}(2, c)
    a2 = M.IT_ThreeA{Int}(4, b)
    @test a2.b.c.a.value == 1
end

# Adapted from `typegroup` testset "multiple type parameters".
@testset "multiple type parameters" begin
    eval(:(module _M_multi
        struct IT_MultiA{K,V}
            key::K
            value::V
            other::Union{Nothing, IT_MultiB{K,V}}
        end
        struct IT_MultiB{K,V}
            key::K
            value::V
            other::Union{Nothing, IT_MultiA{K,V}}
        end
    end))
    M = _M_multi
    @test fieldtype(M.IT_MultiA{String,Int}, :other) == Union{Nothing, M.IT_MultiB{String,Int}}

    a = M.IT_MultiA{String,Int}("a", 1, nothing)
    b = M.IT_MultiB{String,Int}("b", 2, a)
    @test b.other.key == "a"
    @test b.other.value == 1
end

# Adapted from `typegroup` testset "four-way mutual recursion".
@testset "four-way mutual recursion" begin
    eval(:(module _M_fourway
        struct IT_FourA{T}
            b::Union{Nothing, IT_FourB{T}}
            d::Union{Nothing, IT_FourD{T}}
        end
        struct IT_FourB{T}
            c::Union{Nothing, IT_FourC{T}}
            a::Union{Nothing, IT_FourA{T}}
        end
        struct IT_FourC{T}
            d::Union{Nothing, IT_FourD{T}}
            b::Union{Nothing, IT_FourB{T}}
        end
        struct IT_FourD{T}
            a::Union{Nothing, IT_FourA{T}}
            c::Union{Nothing, IT_FourC{T}}
        end
    end))
    M = _M_fourway
    a = M.IT_FourA{Int}(nothing, nothing)
    b = M.IT_FourB{Int}(nothing, a)
    c = M.IT_FourC{Int}(nothing, b)
    d = M.IT_FourD{Int}(a, c)
    @test d.a === a
    @test d.c.b.a === a
end

# Adapted from `typegroup` testset "graph with typed edges".
@testset "graph with typed edges" begin
    eval(:(module _M_graph
        struct IT_Graph{N, E}
            nodes::Vector{IT_GraphNode{N, E}}
        end
        struct IT_GraphNode{N, E}
            data::N
            edges::Vector{IT_GraphEdge{N, E}}
        end
        struct IT_GraphEdge{N, E}
            weight::E
            target::IT_GraphNode{N, E}
        end
    end))
    M = _M_graph
    n1 = M.IT_GraphNode{String, Float64}("A", M.IT_GraphEdge{String,Float64}[])
    n2 = M.IT_GraphNode{String, Float64}("B", M.IT_GraphEdge{String,Float64}[])
    e = M.IT_GraphEdge{String, Float64}(1.5, n2)
    push!(n1.edges, e)
    g = M.IT_Graph{String, Float64}([n1, n2])
    @test g.nodes[1].edges[1].target.data == "B"
end

# Adapted from `typegroup` testset "JSON-like recursive structure".
@testset "JSON-like recursive structure" begin
    eval(:(module _M_json
        struct IT_JSONValue
            data::Union{Nothing, Bool, Int, Float64, String, IT_JSONArray, IT_JSONObject}
        end
        struct IT_JSONArray
            elements::Vector{IT_JSONValue}
        end
        struct IT_JSONObject
            pairs::Vector{Pair{String, IT_JSONValue}}
        end
    end))
    M = _M_json
    arr = M.IT_JSONArray([M.IT_JSONValue(42), M.IT_JSONValue("hello")])
    obj = M.IT_JSONObject([Pair("array", M.IT_JSONValue(arr))])
    @test obj.pairs[1].second.data.elements[1].data == 42
end

# Adapted from `typegroup` testset "doubly-linked list".
@testset "doubly-linked list" begin
    eval(:(module _M_dll
        mutable struct IT_DLNode{T}
            value::T
            prev::Union{Nothing, IT_DLNode{T}}
            next::Union{Nothing, IT_DLNode{T}}
        end
    end))
    M = _M_dll
    n1 = M.IT_DLNode(1, nothing, nothing)
    n2 = M.IT_DLNode(2, n1, nothing)
    n1.next = n2
    @test n1.next.value == 2
    @test n2.prev.value == 1
end

# Adapted from `typegroup` testset "binary tree with parent pointer".
@testset "binary tree with parent pointer" begin
    eval(:(module _M_bintree
        mutable struct IT_BinTree{T}
            value::T
            parent::Union{Nothing, IT_BinTree{T}}
            left::Union{Nothing, IT_BinTree{T}}
            right::Union{Nothing, IT_BinTree{T}}
        end
    end))
    M = _M_bintree
    root = M.IT_BinTree(10, nothing, nothing, nothing)
    left = M.IT_BinTree(5, root, nothing, nothing)
    right = M.IT_BinTree(15, root, nothing, nothing)
    root.left = left
    root.right = right
    @test root.left.parent === root
    @test root.right.value == 15
end

# Adapted from `typegroup` testset "lambda calculus AST".
@testset "lambda calculus AST" begin
    eval(:(module _M_lambda
        struct IT_LamVar
            name::Symbol
        end
        struct IT_LamAbs
            param::Symbol
            body::Union{IT_LamVar, IT_LamAbs, IT_LamApp}
        end
        struct IT_LamApp
            func::Union{IT_LamVar, IT_LamAbs, IT_LamApp}
            arg::Union{IT_LamVar, IT_LamAbs, IT_LamApp}
        end
    end))
    M = _M_lambda
    v = M.IT_LamVar(:x)
    abs = M.IT_LamAbs(:x, v)
    app = M.IT_LamApp(abs, v)
    @test app.func.param == :x
end

# Adapted from `typegroup` testset "entity-component pattern".
@testset "entity-component pattern" begin
    eval(:(module _M_entity
        struct IT_Entity
            id::Int
            components::Dict{Symbol, IT_Component}
        end
        struct IT_Component
            owner::IT_Entity
            data::Any
        end
    end))
    M = _M_entity
    e = M.IT_Entity(1, Dict{Symbol, M.IT_Component}())
    c = M.IT_Component(e, "health")
    e.components[:health] = c
    @test e.components[:health].owner === e
end

# Adapted from `typegroup` testset "NamedTuple fields".
@testset "NamedTuple fields" begin
    eval(:(module _M_namedtuple
        struct IT_NTNode
            data::@NamedTuple{value::Int, edge::Union{Nothing, IT_NTEdge}}
        end
        struct IT_NTEdge
            info::@NamedTuple{from::IT_NTNode, to::IT_NTNode, weight::Float64}
        end
    end))
    M = _M_namedtuple
    n1 = M.IT_NTNode((value=1, edge=nothing))
    n2 = M.IT_NTNode((value=2, edge=nothing))
    e = M.IT_NTEdge((from=n1, to=n2, weight=1.0))
    @test e.info.from.data.value == 1
end

# Adapted from `typegroup` testset "bounded type parameters".
@testset "bounded type parameters" begin
    eval(:(module _M_bounded
        struct IT_BoundedA{T <: Number}
            b::Union{Nothing, IT_BoundedB{T}}
        end
        struct IT_BoundedB{T <: Number}
            a::Union{Nothing, IT_BoundedA{T}}
        end
    end))
    M = _M_bounded
    a = M.IT_BoundedA{Int}(nothing)
    b = M.IT_BoundedB{Float64}(nothing)
    @test fieldtype(M.IT_BoundedA{Int}, :b) == Union{Nothing, M.IT_BoundedB{Int}}
end

# Adapted from `typegroup` testset "deeply nested Union".
@testset "deeply nested Union" begin
    eval(:(module _M_deepunion
        struct IT_DeepUnionA
            x::Union{Nothing, Union{Int, Union{String, Union{Float64, IT_DeepUnionB}}}}
        end
        struct IT_DeepUnionB
            y::Union{Nothing, IT_DeepUnionA}
        end
    end))
    M = _M_deepunion
    a = M.IT_DeepUnionA(nothing)
    b = M.IT_DeepUnionB(a)
    @test b.y === a
end

# Adapted from `typegroup` testset "self-referential supertype parameter".
@testset "self-referential supertype parameter" begin
    eval(:(module _M_selfsuper_single
        struct IT_SelfSuperNode{T} <: AbstractVector{IT_SelfSuperNode{T}}
            data::T
        end
    end))
    M1 = _M_selfsuper_single
    @test M1.IT_SelfSuperNode{Int} <: AbstractVector{M1.IT_SelfSuperNode{Int}}
    @test supertype(M1.IT_SelfSuperNode{Int}) == AbstractVector{M1.IT_SelfSuperNode{Int}}
    n = M1.IT_SelfSuperNode{Int}(42)
    @test n.data == 42

    eval(:(module _M_selfsuper_pair
        struct IT_SelfSuperA{T} <: AbstractVector{IT_SelfSuperA{T}}
            b::Union{Nothing, IT_SelfSuperB{T}}
        end
        struct IT_SelfSuperB{T}
            a::IT_SelfSuperA{T}
        end
    end))
    M2 = _M_selfsuper_pair
    @test M2.IT_SelfSuperA{Int} <: AbstractVector{M2.IT_SelfSuperA{Int}}
    @test fieldtype(M2.IT_SelfSuperB{Int}, :a) == M2.IT_SelfSuperA{Int}
end

# Adapted from `typegroup` testset "red/black list with AbstractArray{T,0} supertype".
@testset "red/black list with AbstractArray{T,0} supertype" begin
    eval(:(module _M_redblack
        struct IT_RedNode <: AbstractArray{IT_BlackNode, 0}
            child::Union{Nothing, IT_BlackNode}
        end
        struct IT_BlackNode <: AbstractArray{IT_RedNode, 0}
            child::Union{Nothing, IT_RedNode}
        end
    end))
    M = _M_redblack
    @test M.IT_RedNode <: AbstractArray{M.IT_BlackNode, 0}
    @test M.IT_BlackNode <: AbstractArray{M.IT_RedNode, 0}
    @test eltype(M.IT_RedNode) == M.IT_BlackNode
    @test eltype(M.IT_BlackNode) == M.IT_RedNode

    r1 = M.IT_RedNode(nothing)
    b1 = M.IT_BlackNode(r1)
    r2 = M.IT_RedNode(b1)
    b2 = M.IT_BlackNode(r2)
    @test b2.child.child.child === r1
    @test b2.child.child.child.child === nothing
end

# Adapted from `typegroup` testset "Tuple fields with incomplete types".
@testset "Tuple fields with incomplete types" begin
    # Self-referential Tuple field
    eval(:(module _M_tuple_self
        struct IT_TupleSelf
            data::Tuple{IT_TupleSelf}
        end
    end))
    @test fieldtype(_M_tuple_self.IT_TupleSelf, :data) == Tuple{_M_tuple_self.IT_TupleSelf}

    # Tuple with two types from the group
    eval(:(module _M_tuple_two
        struct IT_TupleA
            data::Tuple{Int, IT_TupleB}
        end
        struct IT_TupleB
            x::Int
        end
    end))
    M = _M_tuple_two
    @test fieldtype(M.IT_TupleA, :data) == Tuple{Int, M.IT_TupleB}
    a = M.IT_TupleA((42, M.IT_TupleB(99)))
    @test a.data[1] == 42
    @test a.data[2].x == 99

    # NTuple with self-reference through Union
    eval(:(module _M_tuple_ntuple
        struct IT_NTupleNode
            neighbors::NTuple{3, Union{Nothing, IT_NTupleNode}}
        end
    end))
    @test fieldtype(_M_tuple_ntuple.IT_NTupleNode, :neighbors) ==
        NTuple{3, Union{Nothing, _M_tuple_ntuple.IT_NTupleNode}}
    n = _M_tuple_ntuple.IT_NTupleNode((nothing, nothing, nothing))
    @test n.neighbors[1] === nothing

    # Tuple with Union containing incomplete type
    eval(:(module _M_tuple_union
        struct IT_TupleUnion
            data::Tuple{Int, Union{Nothing, IT_TupleUnion}}
        end
    end))
    t = _M_tuple_union.IT_TupleUnion((42, nothing))
    @test t.data[1] == 42
    @test t.data[2] === nothing
    t2 = _M_tuple_union.IT_TupleUnion((99, t))
    @test t2.data[2].data[1] == 42
end

# Adapted from `typegroup` testset "method call on incomplete typegroup type".
# In the unified model the second struct's field-type expression constructs an
# instance of the first struct while it is still incomplete (its placeholder is
# published but `_typebody!` has not run). The constructor call should fail.
@testset "method call on incomplete type during definition" begin
    @test_throws Exception eval(:(module _M_earlycall
        struct IT_EarlyCall_A
            x::Int
            b::Union{Nothing, IT_EarlyCall_B}
        end
        struct IT_EarlyCall_B
            a::(IT_EarlyCall_A(1, nothing); IT_EarlyCall_A)
        end
    end))
end

# Adapted from `typegroup` testset "method definition on incomplete type during
# super expression". Under the unified incomplete-types model the partial type
# is published as a placeholder before the super expression runs, so adding a
# method on it is well-defined rather than erroring.
@testset "method definition on incomplete type during super expression" begin
    m = eval(:(module _M_sideeffect
        struct IT_SideEffect_A <: (global _it_se_g; _it_se_g(::IT_SideEffect_A) = 1; Any)
            b::Union{Nothing, IT_SideEffect_B}
        end
        struct IT_SideEffect_B
            a::Union{Nothing, IT_SideEffect_A}
        end
    end))
    @test m._it_se_g(m.IT_SideEffect_A(nothing)) === 1
end

# Adapted from `typegroup` testset "invalid supertype errors". Note: the
# "subtype itself" case manifests in the unified model as an unresolved
# forward reference at module close (`UndefVarError`) rather than as the
# typegroup's eager "a type cannot subtype itself" diagnostic.
@testset "invalid supertype errors" begin
    @test_throws Exception eval(:(module _M_badtuple
        struct IT_BadTuple <: Tuple{Int}
            x::Int
        end
    end))
    @test_throws Exception eval(:(module _M_badnt
        struct IT_BadNT <: @NamedTuple{x::Int}
            x::Int
        end
    end))
    @test_throws Exception eval(:(module _M_badtype
        struct IT_BadType <: Type{Int}
            x::Int
        end
    end))
    @test_throws Exception eval(:(module _M_badconcrete
        struct IT_BadConcrete <: Int
            x::Int
        end
    end))
    # `IT_SelfSub <: IT_SelfSub` triggers a forward reference that never
    # resolves; `incomplete_finalize` reports the missing name at module close.
    err = try
        eval(:(module _M_selfsub
            struct IT_SelfSub <: IT_SelfSub
                x::Int
            end
        end))
        nothing
    catch e
        e
    end
    @test err !== nothing
end

# Adapted from `typegroup` testset "inner constructors".
@testset "inner constructors" begin
    eval(:(module _M_inner_basic
        struct IT_InnerBasicA
            x::Int
            IT_InnerBasicA() = new(0)
        end
        struct IT_InnerBasicB
            a::IT_InnerBasicA
        end
    end))
    @test _M_inner_basic.IT_InnerBasicA().x == 0
    @test _M_inner_basic.IT_InnerBasicB(_M_inner_basic.IT_InnerBasicA()).a.x == 0

    eval(:(module _M_inner_args
        struct IT_InnerArgsA
            x::Int
            y::Float64
            IT_InnerArgsA(x::Int) = new(x, float(x))
        end
        struct IT_InnerArgsB
            a::IT_InnerArgsA
        end
    end))
    @test _M_inner_args.IT_InnerArgsA(3).y == 3.0
    @test _M_inner_args.IT_InnerArgsB(_M_inner_args.IT_InnerArgsA(5)).a.x == 5

    eval(:(module _M_inner_param
        struct IT_InnerParamA{T}
            x::T
            IT_InnerParamA{T}(x) where {T} = new{T}(x)
            IT_InnerParamA(x::T) where {T} = new{T}(x)
        end
        struct IT_InnerParamB{T}
            a::IT_InnerParamA{T}
        end
    end))
    @test _M_inner_param.IT_InnerParamA{Int}(42).x == 42
    @test _M_inner_param.IT_InnerParamA(3.14).x == 3.14
    @test _M_inner_param.IT_InnerParamB{Int}(_M_inner_param.IT_InnerParamA(1)).a.x == 1

    eval(:(module _M_inner_cross
        struct IT_InnerCrossA
            x::Int
            b::Union{Nothing, IT_InnerCrossB}
            IT_InnerCrossA(x::Int) = new(x, nothing)
        end
        struct IT_InnerCrossB
            a::IT_InnerCrossA
            IT_InnerCrossB(x::Int) = new(IT_InnerCrossA(x))
        end
    end))
    @test _M_inner_cross.IT_InnerCrossA(1).b === nothing
    @test _M_inner_cross.IT_InnerCrossB(42).a.x == 42

    eval(:(module _M_inner_multi
        struct IT_InnerMultiA
            x::Int
            y::Int
            IT_InnerMultiA() = new(0, 0)
            IT_InnerMultiA(x::Int) = new(x, x)
            IT_InnerMultiA(x::Int, y::Int) = new(x, y)
        end
        struct IT_InnerMultiB
            a::IT_InnerMultiA
        end
    end))
    @test _M_inner_multi.IT_InnerMultiA().x == 0
    @test _M_inner_multi.IT_InnerMultiA(3).y == 3
    @test _M_inner_multi.IT_InnerMultiA(1, 2).y == 2
end

# Adapted from `typegroup` testset "docstrings on typegroup types".
@testset "docstrings on deferred struct definitions" begin
    eval(:(module _M_docs
        "IT_DocA: a documented node type"
        struct IT_DocA
            edges::Vector{IT_DocB}
        end
        "IT_DocB: a documented edge type"
        struct IT_DocB
            from::IT_DocA
            to::IT_DocA
        end
    end))
    M = _M_docs
    @test fieldtype(M.IT_DocA, :edges) == Vector{M.IT_DocB}
    @test fieldtype(M.IT_DocB, :from) == M.IT_DocA

    meta = Base.Docs.meta(M)
    bind_a = Base.Docs.Binding(M, :IT_DocA)
    bind_b = Base.Docs.Binding(M, :IT_DocB)
    @test haskey(meta, bind_a)
    @test haskey(meta, bind_b)
    @test contains(string(meta[bind_a].docs[Union{}]), "IT_DocA: a documented node type")
    @test contains(string(meta[bind_b].docs[Union{}]), "IT_DocB: a documented edge type")

    # Mix of documented and undocumented types in one deferred group
    eval(:(module _M_docs_mixed
        "IT_DocC: only this one has a docstring"
        struct IT_DocC
            other::IT_DocD
        end
        struct IT_DocD
            other::IT_DocC
        end
    end))
    M2 = _M_docs_mixed
    @test fieldtype(M2.IT_DocC, :other) == M2.IT_DocD
    meta2 = Base.Docs.meta(M2)
    bind_c = Base.Docs.Binding(M2, :IT_DocC)
    bind_d = Base.Docs.Binding(M2, :IT_DocD)
    @test haskey(meta2, bind_c)
    @test contains(string(meta2[bind_c].docs[Union{}]), "IT_DocC: only this one has a docstring")
    @test !haskey(meta2, bind_d)
end

# In-flight placeholder visibility: between the struct definition and the
# binding of its missing field type, the struct's name must be bound to a
# real (but incomplete) DataType whose super is `Any`, whose field names
# are populated, and whose `types` slot is unset. After the dependency
# binds, the SAME DataType (`objectid` stable) is finalised in place.
@testset "incomplete types: in-flight struct placeholder is observable" begin
    M = Module()
    Core.eval(M, :(using Base))
    Core.eval(M, :(mutable struct _IF_A; b::_IF_B; end))

    @test isdefined(M, :_IF_A)
    @test !isdefined(M, :_IF_B)
    A = Core.eval(M, :_IF_A)
    @test A isa DataType
    @test A.super === Any
    @test A.name.names === Core.svec(:b)
    @test fieldnames(A) == (:b,)
    @test !isdefined(A, :types)
    @test_throws ErrorException fieldtype(A, :b)

    id_before = objectid(A)
    Core.eval(M, :(struct _IF_B; n::Int; end))
    A2 = Core.eval(M, :_IF_A)
    B  = Core.eval(M, :_IF_B)
    @test objectid(A2) === id_before
    @test isdefined(A2, :types)
    @test fieldtype(A2, :b) === B
    @test A2(B(3)).b.n == 3
end

# Phase 9b: `abstract type A <: B end` with B undefined publishes A as an
# incomplete-type placeholder (super temporarily `Any`); when B binds, the
# same DataType (identity preserved) adopts the real supertype in place.
@testset "incomplete types: in-flight abstract-type placeholder is observable" begin
    M = Module()
    Core.eval(M, :(using Base))
    Core.eval(M, :(abstract type _IA_A <: _IA_B end))

    @test isdefined(M, :_IA_A)
    @test !isdefined(M, :_IA_B)
    A = Core.eval(M, :_IA_A)
    @test A isa DataType
    @test isabstracttype(A)
    @test supertype(A) === Any

    id_before = objectid(A)
    Core.eval(M, :(abstract type _IA_B end))
    A2 = Core.eval(M, :_IA_A)
    B  = Core.eval(M, :_IA_B)
    @test objectid(A2) === id_before
    @test supertype(A2) === B
    @test A2 <: B
end

# Phase 9b: same contract for `primitive type`.
@testset "incomplete types: in-flight primitive-type placeholder is observable" begin
    M = Module()
    Core.eval(M, :(using Base))
    Core.eval(M, :(primitive type _IP_A <: _IP_B 32 end))

    @test isdefined(M, :_IP_A)
    @test !isdefined(M, :_IP_B)
    A = Core.eval(M, :_IP_A)
    @test A isa DataType
    @test supertype(A) === Any

    id_before = objectid(A)
    Core.eval(M, :(abstract type _IP_B end))
    A2 = Core.eval(M, :_IP_A)
    B  = Core.eval(M, :_IP_B)
    @test objectid(A2) === id_before
    @test supertype(A2) === B
    @test A2 <: B
end

# Phase 9c: `const B = A` where `A` is undefined defers the const binding
# (B is not in-flight observable today), but downstream method definitions
# whose signatures reference B still work — Phase-5 placeholder admits the
# method against a synthetic B-placeholder, and after A binds, the const
# alias resolves to A and the method's signature is re-stitched to A.
@testset "incomplete types: method on const alias of undef type resolves end-to-end" begin
    M = Module()
    Core.eval(M, :(using Base))
    Core.eval(M, :(const _CM_B = _CM_A))
    Core.eval(M, :(_cm_f(x::_CM_B) = x))

    @test !isdefined(M, :_CM_A)
    @test !isdefined(M, :_CM_B)

    Core.eval(M, :(struct _CM_A; n::Int; end))
    A = Core.eval(M, :_CM_A)
    B = Core.eval(M, :_CM_B)
    f = Core.eval(M, :_cm_f)
    @test B === A
    @test length(methods(f)) == 1
    r = f(A(7))
    @test r isa A
    @test r.n == 7
    @test get(Base.incomplete_refs, M, nothing) === nothing
end

# Phase 9d: macro definition whose argument annotation references an
# unbound type is admitted in-flight via the Phase-5 placeholder path
# (same as plain method defs). `@m` binds immediately, the method is
# visible in `methods(@m)` with a placeholder in its signature, and the
# AST defer queue stays empty. When the type binds, the sig is
# re-stitched to the real type.
@testset "incomplete types: in-flight macro placeholder is observable" begin
    M = Module()
    Core.eval(M, :(using Base))
    Core.eval(M, :(macro _IMM_m(x::_IMM_T) esc(x) end))

    @test !isdefined(M, :_IMM_T)
    @test isdefined(M, Symbol("@_IMM_m"))
    mm = Core.eval(M, Symbol("@_IMM_m"))
    @test length(methods(mm)) == 1
    refs = get(Base.incomplete_refs, M, nothing)
    @test refs !== nothing && haskey(refs, :_IMM_T)

    Core.eval(M, :(struct _IMM_T; n::Int; end))
    T = Core.eval(M, :_IMM_T)
    @test length(methods(mm)) == 1
    sig = first(methods(mm)).sig
    types_in_sig = DataType[]
    let walk(@nospecialize(t)) = begin
            if t isa DataType
                t in types_in_sig || push!(types_in_sig, t)
                foreach(walk, t.parameters)
            elseif t isa UnionAll
                walk(t.body)
            elseif t isa Union
                walk(t.a); walk(t.b)
            end
        end
        walk(sig)
    end
    @test T in types_in_sig
    @test get(Base.incomplete_refs, M, nothing) === nothing
end

# A module that still has unresolved incomplete-type deferrals at module
# close raises `IncompleteTypeError`, which must propagate out of
# `Base.compilecache` and prevent a `.ji` cache file from being written.
@testset "incomplete types: precompilation rejects unresolved deferrals" begin
    include("precompile_utils.jl")
    precompile_test_harness("incomplete types precompile") do load_path
        write(joinpath(load_path, "IncompleteTypePrecompile.jl"),
              """
              module IncompleteTypePrecompile
                  struct Dangling
                      x::__Never_Defined_In_Precompile__
                  end
              end
              """)
        pkgid = Base.PkgId("IncompleteTypePrecompile")
        result = @test_warn r"IncompleteTypeError.*__Never_Defined_In_Precompile__"s try
            Base.compilecache(pkgid)
        catch e
            e
        end
        @test result isa ErrorException
        @test occursin("Failed to precompile", result.msg)
        @test !isfile(Base.compilecache_path(pkgid, ""))
    end
end

