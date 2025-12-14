from __future__ import annotations
from jaxtyping import PyTree


def sanity1(x: int) -> bool:
    return False


def sanity2(x: int) -> bool:
    """
    Args:
        x: input
    """
    return False


def sanity3(x: int) -> bool:
    """
    Args:
        x (int): input
    """
    return False


def easy1(x: PyTree[int]) -> bool:
    return False


def easy2(x: PyTree[int]) -> bool:
    """
    Args:
        x: input
    """
    return False


def easy3(x: PyTree[int], y: str) -> bool:
    """
    Args:
        x (PyTree[int]): input
        y: name
    """
    return False


class MyClass:
    "Hi!"

    def __init__(self, i):
        self.i = i


def hard1(x: PyTree[MyClass], y: str) -> bool:
    return True


def hard2(x: PyTree[MyClass], y: str) -> bool:
    """
    Args:
        x: input
    """
    return True


def hard3(x: PyTree[MyClass], y: MyClass) -> bool:
    """
    Args:
        x (PyTree[MyClass]): input
        y (MyClass): extra
    """
    return True
