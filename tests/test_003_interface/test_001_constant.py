from pangolin.interface import constant, InfixRV
from pangolin.ir import Constant
from numpy.testing import assert_array_equal
import numpy as np


def test_1():
    x = constant(1)
    assert isinstance(x, InfixRV)
    assert isinstance(x.op, Constant)
    assert x.shape == ()
    assert x.op.value == 1


def test_2():
    x = constant([1, 2])
    assert isinstance(x, InfixRV)
    assert isinstance(x.op, Constant)
    assert x.shape == (2,)
    assert_array_equal(x.op.value, np.array([1, 2]))


def test_3():
    x = constant((1, 2))
    assert x.shape == (2,)


def test_4():
    x = constant(((1, 2), (3, 4), (5, 6)))
    assert x.shape == (3, 2)


def test_numpy_array():
    x = np.random.randn(3, 4)
    x = constant(x)
    assert x.shape == (3, 4)


def test_jax_array():
    from jax import numpy as jnp

    x = jnp.ones((3, 4))
    y = constant(x)
    assert y.shape == (3, 4)
    assert_array_equal(x, y.op.value)
