"""Timing utilities."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, TypeVar

T = TypeVar("T")


def timeit(func: Callable[..., T]) -> Callable[..., T]:
    """Simple decorator to measure execution time."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f}s")
        return result
    return wrapper


@contextmanager
def timer(name: str):
    start = time.time()
    yield
    end = time.time()
    print(f"{name} took {end - start:.2f}s")
