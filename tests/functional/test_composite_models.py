from pangolin import ir
from pangolin.ir import RV


from pangolin.interface import *
from pangolin.ir import Composite
from pangolin.functional.compositing import composite
import pytest


def test_make_composite_plain_normal():
    loc = constant(0)
    scale = constant(1)
    result = composite(normal, simplify=False)(loc, scale)
    assert result.op == Composite(2, (ir.Normal(),), ((0, 1),))
    assert result.parents == (loc, scale)


def test_make_composite1():
    def f(a):
        b = a * a
        c = exponential(b)
        return c

    x = constant(1)
    result = composite(f)(x)

    num_inputs = 1
    ops = (ir.Mul(), ir.Exponential())
    par_nums = ((0, 0), (1,))
    assert result.op == Composite(num_inputs, ops, par_nums)
    assert result.parents == (x,)


def test_make_composite2():
    def f(a, b):
        c = a * b
        d = a + c
        e = normal(c, d)
        return e

    a = constant(1)
    b = constant(2)
    result = composite(f)(a, b)
    num_inputs = 2
    ops = (ir.Mul(), ir.Add(), ir.Normal())
    par_nums = ((0, 1), (0, 2), (2, 3))
    assert result.op == Composite(num_inputs, ops, par_nums)
    assert result.parents == (a, b)


def test_make_composite_closure1():
    x = normal(0, 1)

    def f(a):
        return normal(x, a)

    a = constant(2)
    result = composite(f)(a)
    num_inputs = 2
    ops = (ir.Normal(),)
    par_nums = ((1, 0),)  # [[x,a]]
    assert result.op == Composite(num_inputs, ops, par_nums)
    assert result.parents == (a, x)


def test_make_composite_closure2():
    x = normal(0, 1)
    y = normal(0, 1)

    def f():
        z = x * y
        return normal(y, z)

    result = composite(f)()
    num_inputs = 2
    ops = (ir.Mul(), ir.Normal())
    par_nums = ((0, 1), (1, 2))  # [[x,y],[y,z]]
    assert result.op == Composite(num_inputs, ops, par_nums)
    assert result.parents == (x, y)


def test_make_composite_closure3():
    # x = normal(0, 1)
    y = normal(0, 1)

    def f(x):
        z = x * y
        return normal(y, z)

    x = constant(2)
    result = composite(f)(x)
    num_inputs = 2
    ops = (ir.Mul(), ir.Normal())
    par_nums = ((0, 1), (1, 2))  # [[x,y],[y,z]]
    assert result.op == Composite(num_inputs, ops, par_nums)
    assert result.parents == (x, y)


def test_make_composite_closure4():
    x = normal(0, 1)
    # y = normal(0, 1)

    def f(y):
        z = x * y
        return normal(y, z)

    y = constant(2)
    result = composite(f)(y)
    num_inputs = 2
    ops = (ir.Mul(), ir.Normal())
    par_nums = ((1, 0), (0, 2))  # [[x,y],[y,z]]
    assert result.op == Composite(num_inputs, ops, par_nums)
    assert result.parents == (y, x)


def test_make_composite_normal_const_rv():
    scale = constant(3.3)

    def fun(x):
        return normal(x, scale)

    x = constant(2)
    result = composite(fun, simplify=False)(x)
    num_inputs = 2
    ops = (ir.Normal(),)
    par_nums = ((0, 1),)  # [[x,scale]]
    assert result.op == Composite(num_inputs, ops, par_nums)
    assert result.parents == (x, scale)


def test_composite_flat_explicit_inputs_precede_closure_captures():
    captured = constant(10)

    def fun(explicit):
        return captured - explicit

    explicit = constant(3)
    result = composite(fun)(explicit)

    assert result.op == Composite(2, (ir.Sub(),), ((1, 0),))
    assert result.parents == (explicit, captured)


def test_composite_flat_mul_exponential():
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


def test_composite_flat_inside_vmap1():
    def f(a):
        return normal(a, a * a)

    x = normal(0, 1)
    z = vmap(composite(f), None, 3)(x)

    composite_op = ir.Composite(1, (ir.Mul(), ir.Normal()), ((0, 0), (0, 1)))
    vmap_op = ir.VMap(composite_op, (None,), 3)
    assert z.op == vmap_op


def test_vmap_inside_composite_flat2():
    def f(a):
        b = a * a
        return vmap(normal, None, 3)(a, b)

    x = normal(0, 1)
    z = composite(f)(x)

    vmap_op = ir.VMap(ir.Normal(), (None, None), 3)
    composite_op = ir.Composite(1, (ir.Mul(), vmap_op), ((0, 0), (0, 1)))
    assert z.op == composite_op


def test_composite_exponential():
    @composite
    def fun(x):
        return exponential(x)

    z = fun(2.2)
    assert z.op == ir.Composite(
        0,
        (ir.Constant(2.2), ir.Exponential()),
        ((), (0,)),
    )
    assert z.parents == ()


def test_composite_normal():
    @composite
    def fun(x):
        return normal(x, 3.3)

    z = fun(2.2)

    assert z.op == ir.Composite(
        0,
        (ir.Constant(2.2), ir.Constant(3.3), ir.Normal()),
        ((), (), (0, 1)),
    )
    assert z.parents == ()


def test_composite_normal_const_rv():
    scale = constant(3.3)

    @composite
    def fun(x):
        return normal(x, scale)

    z = fun(2.2)

    assert z.op == ir.Composite(
        1,
        (ir.Constant(2.2), ir.Normal()),
        ((), (1, 0)),
    )
    assert z.parents == (scale,)


def test_composite_normal_const_rv_reversed():
    loc = constant(3.3)

    @composite
    def fun(x):
        return normal(loc, x)

    z = fun(2.2)

    assert z.op == ir.Composite(
        1,
        (ir.Constant(2.2), ir.Normal()),
        ((), (0, 1)),
    )
    assert z.parents == (loc,)


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


def test_composite_flat_inside_vmap():
    def f(a):
        return normal(a, a * a)

    x = normal(0, 1)
    z = vmap(composite(f), None, 3)(x)

    composite_op = ir.Composite(1, (ir.Mul(), ir.Normal()), ((0, 0), (0, 1)))
    vmap_op = ir.VMap(composite_op, (None,), 3)
    assert z.op == vmap_op


def test_composite_inside_vmap():
    def f(a):
        return normal(a, a * a)

    x = normal(0, 1)
    z = vmap(composite(f), None, 3)(x)

    composite_op = ir.Composite(1, (ir.Mul(), ir.Normal()), ((0, 0), (0, 1)))
    vmap_op = ir.VMap(composite_op, (None,), 3)
    assert z.op == vmap_op


def test_vmap_inside_composite_flat():
    def f(a):
        b = a * a
        return vmap(normal, None, 3)(a, b)

    x = normal(0, 1)
    z = composite(f)(x)

    vmap_op = ir.VMap(ir.Normal(), (None, None), 3)
    composite_op = ir.Composite(1, (ir.Mul(), vmap_op), ((0, 0), (0, 1)))
    assert z.op == composite_op


def test_vmap_inside_composite():
    def f(a):
        b = a * a
        return vmap(normal, None, 3)(a, b)

    x = normal(0, 1)
    z = composite(f)(x)

    vmap_op = ir.VMap(ir.Normal(), (None, None), 3)
    composite_op = ir.Composite(1, (ir.Mul(), vmap_op), ((0, 0), (0, 1)))
    assert z.op == composite_op


def test_composite_complex_inputs():
    @composite
    def f(input):
        x, (y, z) = input
        a = add(x, z)
        b = div(y, z)
        return mul(a, b)

    z = f((0, (1, 2)))

    assert z.op == ir.Composite(
        0,
        (
            ir.Constant(0),
            ir.Constant(2),
            ir.Add(),
            ir.Constant(1),
            ir.Constant(2),
            ir.Div(),
            ir.Mul(),
        ),
        ((), (), (0, 1), (), (), (3, 4), (2, 5)),
    )
    assert z.parents == ()


def test_composite_requires_one_rv_output():
    def fun(x):
        return x + 1, x + 2

    with pytest.raises(TypeError, match="must return exactly one newly created RV"):
        composite(fun)(constant(1))


def test_composite_rejects_non_rv_output():
    with pytest.raises(TypeError, match="PyTree whose leaves are RV"):
        composite(lambda x: 1)(constant(1))


def test_composite_rejects_existing_output():
    with pytest.raises(ValueError, match="every output must be created during the function call"):
        composite(lambda x: x)(constant(1))


def test_composite_norm():
    x = constant(0.5)
    noise = constant(1e-3)

    def f(last):
        return normal(last, noise)  # +1

    y = composite(f, simplify=False)(x)

    assert y.op == Composite(2, (ir.Normal(),), [[0, 1]])
    assert y.parents == (x, noise)
