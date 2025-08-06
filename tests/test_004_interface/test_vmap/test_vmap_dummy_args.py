from pangolin.interface import *
from pangolin import ir
from collections.abc import Callable

from pangolin.interface.vmap import (
    convert_args,
    generated_nodes,
    vmap_dummy_args,
    AbstractOp,
    vmap_eval,
    vmap
)
import numpy as np

def test_vmap_dummy_args():
    in_axes = (0,)
    axis_size = None
    args = (makerv([1, 2, 3]),)
    dummy_args, axis_size = vmap_dummy_args(in_axes, axis_size, *args)
    assert len(dummy_args) == 1
    assert axis_size == 3
    assert dummy_args[0].shape == ()


def test_vmap_dummy_args1():
    a = makerv(np.ones((5, 3)))
    b = makerv(np.ones(5))
    (dummy_a, dummy_b), axis_size = vmap_dummy_args([0, 0], 5, a, b)
    assert dummy_a.shape == (3,)
    assert dummy_b.shape == ()
    assert axis_size == 5


def test_vmap_dummy_args2():
    a = makerv(np.ones((5, 3)))
    b = makerv(np.ones((3, 5)))
    (dummy_a, dummy_b), axis_size = vmap_dummy_args([0, 1], 5, a, b)
    assert dummy_a.shape == (3,)
    assert dummy_b.shape == (3,)
    assert axis_size == 5


def test_vmap_dummy_args3():
    a = makerv(np.ones((5, 3, 2)))
    b = makerv(np.ones((3, 1, 9)))
    (dummy_a, dummy_b), axis_size = vmap_dummy_args([1, None], None, a, b)
    assert dummy_a.shape == (5, 2)
    assert dummy_b.shape == (3, 1, 9)
    assert axis_size == 3
