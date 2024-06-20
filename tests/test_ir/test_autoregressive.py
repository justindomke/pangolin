from pangolin.ir import *
from pangolin import interface

def test_normal_mapped_scale():
    d = Autoregressive(normal_scale, num_constants=0, length=5)
    assert d.get_shape((),(5,)) == (5,)

def test_normal_constant_scale():
    d = Autoregressive(normal_scale, num_constants=1, length=5)
    assert d.get_shape((),()) == (5,)

def test_diag_normal_mapped_scale():
    diag_normal = VMapDist(normal_scale,(0,0))
    d = Autoregressive(diag_normal, num_constants=0, length=5)
    assert d.get_shape((10,),(5,10)) == (5,10)

def test_diag_normal_constant_scale():
    diag_normal = VMapDist(normal_scale,(0,0))
    d = Autoregressive(diag_normal, num_constants=1, length=5)
    assert d.get_shape((10,),(10,)) == (5,10)

def test_bernoulli():
    d = Autoregressive(bernoulli, num_constants=0, length=5)
    assert d.get_shape(()) == (5,)

def test_autoregressive_wrapper1():
    x = normal_scale(0,1)
    z = interface.autoregressive(lambda x: normal_scale(x,x*x), 5)(x)
    interface.print_upstream(z)

def test_autoregressive_wrapper2():
    x = normal_scale(0,1)
    scale = makerv(2)
    z = interface.autoregressive(lambda x: normal_scale(x,scale), 5)(x)
    interface.print_upstream(z)
    assert z.shape == (5,)

def test_autoregressive_exponential1():
    x = exponential(1)
    z = interface.autoregressive(lambda last: exponential(last), 100)(x)
    assert z.shape == (100,)

def test_autoregressive_closure1():
    x = exponential(1)
    a = exponential(2)
    z = interface.autoregressive(lambda last: exponential(a*last), 100)(x)
    assert z.shape == (100,)
    assert isinstance(z.cond_dist, Autoregressive)
    assert z.cond_dist.num_constants == 1
    print(f"{z.parents=}")
    assert z.parents == (x,a)
    base_cond_dist = z.cond_dist.base_cond_dist
    assert isinstance(base_cond_dist, Composite)
    assert base_cond_dist.cond_dists == [mul, exponential]
    assert base_cond_dist.par_nums == [[1,0],[2]] # [[a,last],[tmp]]


