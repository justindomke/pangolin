from pangolin.ir import *

def test_plain_normal():
    d = Composite(2, [normal_scale], [[0,1]])
    assert d.get_shape((),()) == ()
    assert d.random

def test_single_input():
    d = Composite(1, [normal_scale], [[0,0]])
    assert d.get_shape(()) == ()
    assert d.random

def test_diag_normal():
    # def f(x0,x1):
    #     x2 = np.arange(5)
    #     x3 = x1 * x2
    #     x4 = normal(x0, x3)
    d = Composite(2, [Constant(np.arange(5)), VMapDist(mul,(None,0)), VMapDist(normal_scale,(0,0))], [[],[1,2],[0,3]])
    assert d.get_shape((5,),()) == (5,)