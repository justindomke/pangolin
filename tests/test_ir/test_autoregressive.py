from pangolin.ir import *

def test_normal_position_0():
    d = Autoregressive(normal_scale, 0, 5)
    assert d.get_shape((),(5,)) == (5,)

def test_normal_position_1():
    d = Autoregressive(normal_scale, 1, 5)
    assert d.get_shape((),(5,)) == (5,)

def test_diag_normal():
    diag_normal = VMapDist(normal_scale,(0,0))
    d = Autoregressive(diag_normal, position=0, axis_size=5)
    assert d.get_shape((10,),(5,10)) == (5,10)

def test_autoregressive1():
    d = Autoregressive(bernoulli, 0, 5)
    assert d.get_shape(()) == (5,)
