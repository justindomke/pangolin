from cleanpangolin.interface import *
from cleanpangolin.ir import Composite
from cleanpangolin.interface.composite import make_composite, composite
from cleanpangolin.interface.loops import Loop, VMapRV

def test_make_composite_plain_normal():
    op, consts = make_composite(normal, (), ())
    assert op == Composite(2, (ir.Normal(),), ((0, 1),))
    assert consts == []


def test_make_composite1():
    def f(a):
        b = a * a
        c = exponential(b)
        return c

    op, consts = make_composite(f, ())

    num_inputs = 1
    ops = (ir.Mul(), ir.Exponential())
    par_nums = ((0, 0), (1,))
    assert op == Composite(num_inputs, ops, par_nums)
    assert consts == []


def test_make_composite2():
    def f(a, b):
        c = a * b
        d = a + c
        e = normal(c, d)
        return e

    op, consts = make_composite(f, (), ())
    num_inputs = 2
    ops = (ir.Mul(), ir.Add(), ir.Normal())
    par_nums = ((0, 1), (0, 2), (2, 3))
    assert op == Composite(num_inputs, ops, par_nums)
    assert consts == []


def test_make_composite_closure1():
    x = normal(0, 1)

    def f(a):
        return normal(x, a)

    op, consts = make_composite(f, ())
    num_inputs = 2
    ops = (ir.Normal(),)
    par_nums = ((1, 0),)  # [[x,a]]
    assert op == Composite(num_inputs, ops, par_nums)
    assert consts == [x]


def test_make_composite_closure2():
    x = normal(0, 1)
    y = normal(0, 1)

    def f():
        z = x * y
        return normal(y, z)

    op, consts = make_composite(f)
    num_inputs = 2
    ops = (ir.Mul(), ir.Normal())
    par_nums = ((0, 1), (1, 2))  # [[x,y],[y,z]]
    assert op == Composite(num_inputs, ops, par_nums)
    assert consts == [x, y]


def test_composite():
    def f(a):
        b = a * a
        c = exponential(b)
        return c

    x = normal(0, 1)
    z = composite(f)(x)
    assert z.shape == ()
    assert z.parents == (x,)
    num_inputs = 1
    ops = (ir.Mul(), ir.Exponential())
    par_nums = ((0, 0), (1,))
    assert z.op == Composite(num_inputs, ops, par_nums)


def test_composite_inside_vmap():
    def f(a):
        return normal(a, a * a)

    x = normal(0, 1)
    z = vmap(composite(f), None, 3)(x)

    composite_op = ir.Composite(1, (ir.Mul(), ir.Normal()), ((0, 0), (0, 1)))
    vmap_op = ir.VMap(composite_op, (None,), 3)
    assert z.op == vmap_op


def test_vmap_inside_composite():
    def f(a):
        b = a*a
        return vmap(normal,None,3)(a,b)

    x = normal(0,1)
    z = composite(f)(x)

    vmap_op = ir.VMap(ir.Normal(),(None,None),3)
    composite_op = ir.Composite(1,(ir.Mul(),vmap_op),((0,0),(0,1)))
    assert z.op == composite_op

def test_composite_with_loops():
    def f(x,y):
        with Loop() as i:
            with Loop() as j:
                (c := VMapRV())[i,j] = x[i] * y[j]
                (d := VMapRV())[i,j] = exp(c[i,j])
        return d

    x = vmap(normal,None,5)(0,1)
    y = vmap(normal,None,5)(0,1)

    z = f(x,y)
    print(f"{z=}")

    z = composite(f)(x,y)
    print(f"{z=}")

