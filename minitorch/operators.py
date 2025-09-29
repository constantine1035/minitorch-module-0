"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

from typing import List, TypeVar

A = TypeVar("A")
B = TypeVar("B")

def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    return 1.0 / x


def log_back(a: float, b: float) -> float:
    """b * d(log(a))/da = b / a."""
    return b / a


def inv_back(a: float, b: float) -> float:
    """b * d(1/a)/da = b * (-1/a^2)."""
    return -b / (a * a)


def relu_back(a: float, b: float) -> float:
    """b если a>0 иначе 0."""
    return b if a > 0.0 else 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[A], B]) -> Callable[[Iterable[A]], List[B]]:
    def _apply(xs: Iterable[A]) -> List[B]:
        return [fn(x) for x in xs]
    return _apply


def zipWith(fn: Callable[[A, A], B]) -> Callable[[Iterable[A], Iterable[A]], List[B]]:
    def _apply(xs: Iterable[A], ys: Iterable[A]) -> List[B]:
        return [fn(x, y) for x, y in zip(xs, ys)]
    return _apply


def reduce(fn: Callable[[A, A], A], start: A) -> Callable[[Iterable[A]], A]:
    def _apply(xs: Iterable[A]) -> A:
        acc = start
        for x in xs:
            acc = fn(acc, x)
        return acc
    return _apply


def negList(xs: Iterable[float]) -> List[float]:
    return map(neg)(xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> List[float]:
    return zipWith(add)(xs, ys)


def sum(xs: Iterable[float]) -> float:
    return reduce(add, 0.0)(xs)


def prod(xs: Iterable[float]) -> float:
    return reduce(mul, 1.0)(xs)
