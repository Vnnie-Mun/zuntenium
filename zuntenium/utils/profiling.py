"""
profiling — Timing decorators and memory tracking.
"""
from __future__ import annotations

import functools
import time
import tracemalloc
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def timed(func: F) -> F:
    """
    Decorator that prints wall-clock execution time.

    Usage::

        @timed
        def my_func():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[PROFILE] {func.__qualname__}: {elapsed:.3f} ms")
        return result
    return wrapper  # type: ignore[return-value]


def memory_tracked(func: F) -> F:
    """
    Decorator that measures peak memory usage during function execution.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tracemalloc.start()
        try:
            result = func(*args, **kwargs)
            _, peak = tracemalloc.get_traced_memory()
            print(f"[PROFILE] {func.__qualname__}: peak {peak / 1024:.1f} KB")
        finally:
            tracemalloc.stop()
        return result
    return wrapper  # type: ignore[return-value]


def benchmark(
    func: Callable[..., Any],
    *args: Any,
    repeat: int = 5,
    **kwargs: Any,
) -> dict[str, float]:
    """
    Run a function N times and return timing statistics.

    Returns:
        Dict with "mean_ms", "min_ms", "max_ms", "std_ms".
    """
    import numpy as np
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append((time.perf_counter() - start) * 1000)
    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "std_ms": float(arr.std()),
    }
