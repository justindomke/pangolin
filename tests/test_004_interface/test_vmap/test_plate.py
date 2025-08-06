"""
Tests to make sure plate is doing the same thing as vmap
"""

from pangolin.interface.vmap import vmap, plate
from pangolin import *
from pangolin.interface import normal, multi_normal
import numpy as np


def test_iid_normals():
    x1 = vmap(lambda: normal(0, 1), None, axis_size=10)()
    x2 = plate(size=10)(lambda: normal(0, 1))
    assert x1.op == x2.op


def test_dependent_normals():
    z = normal(0, 1)
    x1 = vmap(lambda: normal(z, 1), None, axis_size=10)()
    x2 = plate(size=10)(lambda: normal(z, 1))
    assert x1.op == x2.op


def test_mult_normal_with_noise():
    z = multi_normal([0, 1, 2], np.eye(3))
    x1 = vmap(lambda z_i: normal(z_i, 1), axis_size=None)(z)
    x2 = vmap(lambda z_i: normal(z_i, 1), axis_size=3)(z)
    x3 = plate(z, size=None)(lambda z_i: normal(z_i, 1))
    x4 = plate(z, size=3)(lambda z_i: normal(z_i, 1))
    assert x1.op == x2.op == x3.op == x4.op


def test_mult_normal_with_noise_axis():
    z = multi_normal([0, 1, 2], np.eye(3))
    x1 = vmap(lambda z_i: normal(z_i, 1))(z)
    x2 = plate(z, size=3, in_axes=0)(lambda z_i: normal(z_i, 1))
    assert x1.op == x2.op
