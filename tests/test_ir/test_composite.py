from pangolin.ir import *
from pangolin import interface

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

def test_make_composite1():
    def f(a):
        b = a*a
        c = exponential(b)
        return c
    op, consts = interface.make_composite(f,())
    assert consts == []
    assert op.num_inputs == 1
    assert op.cond_dists == [mul, exponential]
    assert op.par_nums == [[0,0],[1]]

def test_make_composite2():
    def f(a,b):
        c = a*b
        d = a+c
        e = normal_scale(c,d)
        return e
    op, consts = interface.make_composite(f,(),())
    assert consts == []
    assert op.num_inputs == 2
    assert op.cond_dists == [mul, add, normal_scale]
    assert op.par_nums == [[0,1],[0,2],[2,3]]

def test_make_composite_closure1():
    x = normal_scale(0,1)
    def f(a):
        return normal_scale(x,a)
    op, consts = interface.make_composite(f,())
    assert consts == [x]
    assert op.num_inputs == 2
    assert op.cond_dists == [normal_scale]
    assert op.par_nums == [[1,0]] # [[x,a]]

def test_make_composite_closure2():
    x = normal_scale(0,1)
    y = normal_scale(0,1)
    def f():
        z = x*y
        return normal_scale(y,z)
    op, consts = interface.make_composite(f)
    assert consts == [x,y]
    assert op.num_inputs == 2
    assert op.cond_dists == [mul,normal_scale]
    assert op.par_nums == [[0,1],[1,2]] # [[x,y],[y,z]]

def test_composite():
    def f(a):
        b = a*a
        c = exponential(b)
        return c
    x = interface.normal(0,1)
    z = interface.composite(f)(x)
    assert z.shape == ()
    assert z.parents == (x,)
    op = z.cond_dist
    assert isinstance(op,Composite)
    assert op.num_inputs == 1
    assert op.cond_dists == [mul, exponential]
    assert op.par_nums == [[0, 0], [1]]

def test_composite_inside_vmap():
    def f(a):
        return normal_scale(a,a*a)

    x = normal_scale(0,1)
    z = interface.vmap(interface.composite(f),None,3)(x)
    interface.print_upstream(z)

def test_vmap_inside_composite():
    def f(a):
        b = a*a
        return interface.vmap(normal_scale,None,3)(a,b)

    x = normal_scale(0,1)
    z = interface.composite(f)(x)
    interface.print_upstream(z)
