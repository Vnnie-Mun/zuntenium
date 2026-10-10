"""
bit_utils — Bitstring ↔ integer conversions, parity, Hamming distance.
"""
from __future__ import annotations


def int_to_bitstring(n: int, width: int) -> str:
    """Convert integer n to a zero-padded binary string of given width."""
    return format(n, f"0{width}b")


def bitstring_to_int(s: str) -> int:
    """Convert a binary string to integer."""
    return int(s, 2)


def parity(n: int) -> int:
    """Return the parity of integer n (0 for even popcount, 1 for odd)."""
    p = 0
    while n:
        p ^= n & 1
        n >>= 1
    return p


def hamming_distance(a: str, b: str) -> int:
    """Return the Hamming distance between two equal-length bitstrings."""
    if len(a) != len(b):
        raise ValueError(f"Bitstrings must have equal length: {len(a)} != {len(b)}")
    return sum(x != y for x, y in zip(a, b))


def hamming_weight(s: str) -> int:
    """Return the Hamming weight (popcount) of a bitstring."""
    return s.count("1")


def flip_bit(s: str, position: int) -> str:
    """Flip the bit at the given position in a bitstring."""
    lst = list(s)
    lst[position] = "0" if lst[position] == "1" else "1"
    return "".join(lst)


def counts_to_probabilities(counts: dict[str, int], shots: int) -> dict[str, float]:
    """Normalize a counts dict to probabilities."""
    return {k: v / shots for k, v in counts.items()}
