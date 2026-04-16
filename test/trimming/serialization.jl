# Test that Serialization.serialize supports trimming

using Serialization

mutable struct TrimSerMut
    s::String
    m::Memory{Int}
    u::Union{Int, Nothing}
    t::Tuple{Int, String}
end

struct TrimSerImm
    a::Int
    b::TrimSerMut
end

function @main(args::Vector{String})::Cint
    dir = dirname(PROGRAM_FILE)

    # Serialize stdlib types (test runner will deserialize and verify)
    serialize(joinpath(dir, "_trim_stdlib.jls"),
             (42, "hello", :sym, (1, 2.0), Int[10, 20, 30]))

    # Serialize custom struct types (exercises struct/Memory/Union/Tuple paths)
    x = TrimSerImm(42, TrimSerMut("hello", [4, 2].ref.mem, 7, (3, "world")))
    serialize(joinpath(dir, "_trim_custom.jls"), x)

    println(Core.stdout, "serialization ok")
    return 0
end
