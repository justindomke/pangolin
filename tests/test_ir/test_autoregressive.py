from pangolin.ir import *
from pangolin import interface

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

def test_bernoulli():
    d = Autoregressive(bernoulli, 0, 5)
    assert d.get_shape(()) == (5,)

def test_autoregressive_wrapper1():
    x = normal_scale(0,1)
    z = interface.autoregressive(lambda x: normal_scale(x,x*x), 5)(x)
    interface.print_upstream(z)