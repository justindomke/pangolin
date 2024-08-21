from pangolin.interface import *
from pangolin.ir import Composite
from pangolin.interface.composite import make_composite, composite_flat, composite
from pangolin.interface.loops import Loop, slot
from pangolin import ir


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


def test_make_composite_closure3():
    #x = normal(0, 1)
    y = normal(0, 1)

    def f(x):
        z = x * y
        return normal(y, z)

    op, consts = make_composite(f, ())
    num_inputs = 2
    ops = (ir.Mul(), ir.Normal())
    par_nums = ((0, 1), (1, 2))  # [[x,y],[y,z]]
    assert op == Composite(num_inputs, ops, par_nums)
    assert consts == [y]


def test_make_composite_closure4():
    x = normal(0, 1)
    #y = normal(0, 1)

    def f(y):
        z = x * y
        return normal(y, z)

    op, consts = make_composite(f, ())
    num_inputs = 2
    ops = (ir.Mul(), ir.Normal())
    par_nums = ((1, 0), (0, 2))  # [[x,y],[y,z]]
    assert op == Composite(num_inputs, ops, par_nums)
    assert consts == [x]


def test_make_composite_normal_const_rv():
    scale = makerv(3.3)

    def fun(x):
        return normal(x, scale)

    op, consts = make_composite(fun, ())
    num_inputs = 2
    ops = (ir.Normal(),)
    par_nums = ((0, 1),)  # [[x,scale]]
    assert op == Composite(num_inputs, ops, par_nums)
    assert consts == [scale]


def test_composite_flat_mul_exponential():
    def f(a):
        b = a * a
        c = exponential(b)
        return c

    x = normal(0, 1)
    z = composite_flat(f)(x)
    assert z.shape == ()
    assert z.parents == (x,)
    num_inputs = 1
    ops = (ir.Mul(), ir.Exponential())
    par_nums = ((0, 0), (1,))
    assert z.op == Composite(num_inputs, ops, par_nums)


def test_composite_flat_inside_vmap():
    def f(a):
        return normal(a, a * a)

    x = normal(0, 1)
    z = vmap(composite_flat(f), None, 3)(x)

    composite_op = ir.Composite(1, (ir.Mul(), ir.Normal()), ((0, 0), (0, 1)))
    vmap_op = ir.VMap(composite_op, (None,), 3)
    assert z.op == vmap_op


def test_vmap_inside_composite_flat():
    def f(a):
        b = a * a
        return vmap(normal, None, 3)(a, b)

    x = normal(0, 1)
    z = composite_flat(f)(x)

    vmap_op = ir.VMap(ir.Normal(), (None, None), 3)
    composite_op = ir.Composite(1, (ir.Mul(), vmap_op), ((0, 0), (0, 1)))
    assert z.op == composite_op


def test_composite_flat_with_loops():
    def f(x, y):
        with Loop() as i:
            with Loop() as j:
                (c := slot())[i, j] = x[i] * y[j]
                (d := slot())[i, j] = exp(c[i, j])
        return d

    x = vmap(normal, None, 5)(0, 1)
    y = vmap(normal, None, 5)(0, 1)

    z = f(x, y)
    print(f"{z=}")

    z = composite_flat(f)(x, y)

    c_vmap = ir.VMap(ir.VMap(ir.Mul(), (None, 0)), (0, None))
    d_vmap = ir.VMap(ir.VMap(ir.Exp(), (0,)), (0,))

    assert z.op == ir.Composite(2, (c_vmap, d_vmap), ((0, 1), (2,)))
    assert z.parents == (x, y)


def test_composite_flat_decorator():
    @composite_flat
    def f(x, y):
        with Loop() as i:
            with Loop() as j:
                (c := slot())[i, j] = x[i] * y[j]
                (d := slot())[i, j] = exp(c[i, j])
        return d

    x = vmap(normal, None, 5)(0, 1)
    y = vmap(normal, None, 5)(0, 1)

    z = f(x, y)
    print(f"{z=}")

    z = f(x, y)

    c_vmap = ir.VMap(ir.VMap(ir.Mul(), (None, 0)), (0, None))
    d_vmap = ir.VMap(ir.VMap(ir.Exp(), (0,)), (0,))

    assert z.op == ir.Composite(2, (c_vmap, d_vmap), ((0, 1), (2,)))
    assert z.parents == (x, y)


# def test_double_loop_recursive_slots():
#     x = makerv([0,1,2,3])
#     y = makerv([0,1,2])
#
#     a = slot()
#     with Loop() as i:
#         b = slot()
#         with Loop() as j:
#             b[j] = x[i] * y[j]
#         a[i] = sum(b,0)
#
#     print(f"{a=}")


# def test_loop_inside_function_inside_loop():
#     def f(x):
#         y = slot()
#         with Loop() as i:
#             y[i] = x[i] * x[i]
#         return y
#
#     def g(X):
#         Y = slot()
#         with Loop() as i:
#             Y[i] = f(X[i,:])
#         return Y
#
#     X = makerv([[0,1,2],[3,4,5]])
#     g(X)


def test_composite_exponential():
    @composite
    def fun(x):
        return exponential(x)

    z = fun(2.2)
    assert z.op == ir.Composite(1, (ir.Exponential(),), ((0,),))
    assert z.parents[0].op == ir.Constant(2.2)


def test_composite_normal():
    @composite
    def fun(x):
        return normal(x, 3.3)

    z = fun(2.2)

    assert z.op == ir.Composite(1, (ir.Constant(3.3), ir.Normal()), ((), (0, 1)))
    assert z.parents[0].op == ir.Constant(2.2)


def test_composite_normal_const_rv():
    scale = makerv(3.3)

    @composite
    def fun(x):
        return normal(x, scale)

    z = fun(2.2)

    assert z.op == ir.Composite(2, [ir.Normal()], [[0,1]])
    assert z.parents[0].op == ir.Constant(2.2)
    assert z.parents[1].op == ir.Constant(3.3)

def test_composite_normal_const_rv_reversed():
    loc = makerv(3.3)

    @composite
    def fun(x):
        return normal(loc, x)

    z = fun(2.2)

    assert z.op == ir.Composite(2, [ir.Normal()], [[1,0]])
    assert z.parents[0].op == ir.Constant(2.2)
    assert z.parents[1].op == ir.Constant(3.3)


def test_composite_mul_exponential():
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
        b = a * a
        return vmap(normal, None, 3)(a, b)

    x = normal(0, 1)
    z = composite_flat(f)(x)

    vmap_op = ir.VMap(ir.Normal(), (None, None), 3)
    composite_op = ir.Composite(1, (ir.Mul(), vmap_op), ((0, 0), (0, 1)))
    assert z.op == composite_op


def test_composite_with_loops():
    def f(x, y):
        with Loop() as i:
            with Loop() as j:
                (c := slot())[i, j] = x[i] * y[j]
                (d := slot())[i, j] = exp(c[i, j])
        return d

    x = vmap(normal, None, 5)(0, 1)
    y = vmap(normal, None, 5)(0, 1)

    z = f(x, y)
    print(f"{z=}")

    z = composite(f)(x, y)

    c_vmap = ir.VMap(ir.VMap(ir.Mul(), (None, 0)), (0, None))
    d_vmap = ir.VMap(ir.VMap(ir.Exp(), (0,)), (0,))

    assert z.op == ir.Composite(2, (c_vmap, d_vmap), ((0, 1), (2,)))
    assert z.parents == (x, y)


def test_composite_decorator():
    @composite
    def f(x, y):
        with Loop() as i:
            with Loop() as j:
                (c := slot())[i, j] = x[i] * y[j]
                (d := slot())[i, j] = exp(c[i, j])
        return d

    x = vmap(normal, None, 5)(0, 1)
    y = vmap(normal, None, 5)(0, 1)

    z = f(x, y)
    print(f"{z=}")

    z = f(x, y)

    c_vmap = ir.VMap(ir.VMap(ir.Mul(), (None, 0)), (0, None))
    d_vmap = ir.VMap(ir.VMap(ir.Exp(), (0,)), (0,))

    assert z.op == ir.Composite(2, (c_vmap, d_vmap), ((0, 1), (2,)))
    assert z.parents == (x, y)


def test_composite_complex_inputs():
    @composite
    def f(input):
        (x, (y, z)) = input
        a = x + z
        b = y / z
        return a * b

    z = f((0, (1, 2)))

    assert z.op == ir.Composite(3, (ir.Add(), ir.Div(), ir.Mul()), ((0, 2), (1, 2), (3, 4)))

def test_composite_norm():
    x = makerv(0.5)
    noise = makerv(1e-3)

    @composite
    def f(last):
        return normal(last, noise)  # +1

    y = f(x)

    assert y.op == Composite(2, [ir.Normal()], [[0,1]] )
    assert y.parents == (x, noise)