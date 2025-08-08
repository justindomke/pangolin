from pangolin.simple_interface import makerv, InfixRV
from pangolin.ir import Constant
from numpy.testing import assert_array_equal
import numpy as np

def test_makerv1():
    x = makerv(1)
    assert isinstance(x, InfixRV)
    assert isinstance(x.op, Constant)
    assert x.shape == ()
    assert x.op.value == 1


def test_makerv2():
    x = makerv([1, 2])
    assert isinstance(x, InfixRV)
    assert isinstance(x.op, Constant)
    assert x.shape == (2,)
    assert_array_equal(x.op.value, np.array([1,2]))


def test_makerv3():
    x = makerv((1, 2))
    assert x.shape == (2,)


def test_makerv4():
    x = makerv(((1, 2), (3, 4), (5, 6)))
    assert x.shape == (3, 2)


def test_makerv5():
    import numpy as np

    x = np.random.randn(3, 4)
    x = makerv(x)
    assert x.shape == (3, 4)


def test_makerv6():
    from jax import numpy as np

    x = np.ones((3, 4))
    x = makerv(x)
    assert x.shape == (3, 4)
