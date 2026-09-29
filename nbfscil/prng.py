"""xorshift32: the PRNG shared by the training-time rule and the bit-exact hardware model
(a 32-bit xorshift is a few XOR/shift gates in RTL)."""

MASK32 = 0xFFFFFFFF


def xorshift32(state):
    state ^= (state << 13) & MASK32
    state ^= state >> 17
    state ^= (state << 5) & MASK32
    return state & MASK32


def xorshift32_stream(seed, n):
    """Return (list of n successive 32-bit outputs, final state). seed must be non-zero."""
    s = (seed & MASK32) or 0x9E3779B9
    out = []
    for _ in range(n):
        s = xorshift32(s)
        out.append(s)
    return out, s
