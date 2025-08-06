from pangolin.interface import makerv

def test_makerv1():
    x = makerv(1)
    assert x.shape == ()


def test_makerv2():
    x = makerv([1, 2])
    assert x.shape == (2,)


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
