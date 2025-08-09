from pangolin.simple_interface import *
from pangolin import ir
from collections.abc import Callable

from pangolin.simple_interface.vmapping import (
    convert_args,
    generated_nodes,
    vmap_dummy_args,
    AbstractOp,
    vmap_eval_flat,
    vmap
)
import numpy as np


def test_square():
    def f(x):
        return x*x

    x = constant([1,2,3])
    y = vmap(f)(x)

    assert y.op == ir.VMap(ir.Mul(),(0,0),3)
    assert y.parents == (x, x)

def test_add_and_square():
    def f(x,y):
        z = x+y
        return z*z

    x = constant([1,2,3])
    y = constant([4,5,6])
    u = vmap(f)(x,y)

    assert u.op == ir.VMap(ir.Mul(),(0,0),3)
    z = u.parents[0]
    assert u.parents == (z,z)
    assert z.op == ir.VMap(ir.Add(),(0,0),3)
    assert z.parents == (x,y)

# def test_indexing():
#     def f(x):
#         return x[0] + x[1]

#     x = makerv([[1,2],[3,4],[5,6]])
#     z = vmap(f)(x)



# def test_vmap_partial_mapping():
#     def f(x,y):
#         z = x[0]*y
#         u = x[1]*y
#         return z+u
#
#     x = makerv([[1,2],[3,4],[5,6]])
#     y = makerv(7)
#     t = vmap(f,(0,None))(x,y)
#

def test_single_arg():
    y = vmap(exponential, 0, 3)(np.zeros(3))
    assert y.shape == (3,)
    assert y.op == ir.VMap(ir.Exponential(), (0,), 3)

    y = vmap(exponential, 0, None)(np.zeros(3))
    assert y.shape == (3,)
    assert y.op == ir.VMap(ir.Exponential(), (0,), 3)

def test_two_args():
    y = vmap(normal, 0, 3)(np.zeros(3), np.ones(3))
    assert y.shape == (3,)
    assert y.op == ir.VMap(ir.Normal(), (0,0), 3)

    y = vmap(normal, None, 3)(np.array(0), np.array(1))
    assert y.shape == (3,)
    assert y.op == ir.VMap(ir.Normal(), (None, None), 3)

    y = vmap(normal, (0, 0), 3)(np.zeros(3), np.ones(3))
    assert y.shape == (3,)
    assert y.op == ir.VMap(ir.Normal(), (0, 0), 3)

    y = vmap(normal, (0, None), 3)(np.zeros(3), np.array(1))
    assert y.shape == (3,)
    assert y.op == ir.VMap(ir.Normal(), (0, None), 3)

    y = vmap(normal, (None, 0), 3)(np.array(0), np.ones(3))
    assert y.shape == (3,)
    assert y.op == ir.VMap(ir.Normal(), (None, 0), 3)

def test_two_args_lists():
    y = vmap(normal, [0, 0], 3)(np.zeros(3), np.ones(3))
    assert y.shape == (3,)

    y = vmap(normal, [0, None], 3)(np.zeros(3), np.array(1))
    assert y.shape == (3,)

    y = vmap(normal, [None, 0], 3)(np.array(0), np.ones(3))
    assert y.shape == (3,)

def test_use_arg_twice():
    def f(x):
        return normal(x, x)

    y = vmap(f)(np.array([3.3, 4.4, 5.5]))
    assert y.shape == (3,)
    assert y.op == ir.VMap(ir.Normal(), (0, 0), 3)
    assert y.parents[0] == y.parents[1]

    y = vmap(f, in_axes=0, axis_size=3)(np.array([3.3, 4.4, 5.5]))
    assert y.shape == (3,)
    assert y.op == ir.VMap(ir.Normal(), (0, 0), 3)
    assert y.parents[0] == y.parents[1]


def test_use_arg_twice_complicated():
    def f(x):
        loc = x * 1.1
        scale = x**2.2
        return normal(loc, scale)

    y = vmap(f)(np.array([3.3, 4.4, 5.5]))
    assert y.shape == (3,)
    assert y.parents[0].parents[0] == y.parents[1].parents[0]

    y = vmap(f, in_axes=0, axis_size=None)(np.array([3.3, 4.4, 5.5]))
    assert y.shape == (3,)
    assert y.parents[0].parents[0] == y.parents[1].parents[0]

def test_outside_arg():
    x = constant([1,2,3])
    y = constant(4)
    def f(xi):
        return normal(xi,y)
    z = vmap(f)(x)
    assert z.shape == (3,)
    assert z.parents == (x,y)
    assert z.op == ir.VMap(ir.Normal(), (0, None), 3)

def test_deterministic_scalars():
    # notice z is a SCALAR, because + is deterministic
    x = constant(1)
    y = constant(2)
    def f(x,y):
        z = x+y
        return z*z
    u = vmap(f,None,3)(x,y)
    assert u.shape == (3,)
    assert u.op == ir.VMap(ir.Mul(), (None, None), 3)
    z = u.parents[0]
    assert u.parents == (z,z)
    assert z.shape == ()
    assert z.op == ir.Add()
    assert z.parents == (x,y)

def test_random_scalars():
    # unlike the previous example, notice z is now a VECTOR, because normal is random
    x = constant(1)
    y = constant(2)
    def f(x,y):
        z = normal(x,y)
        return z*z
    u = vmap(f,None,3)(x,y)
    assert u.shape == (3,)
    assert u.op == ir.VMap(ir.Mul(), (0, 0), 3)
    z = u.parents[0]
    assert u.parents == (z,z)
    assert z.shape == (3,)
    assert z.op == ir.VMap(ir.Normal(), (None, None), 3)
    assert z.parents == (x,y)


def test_vmap_dict():
    def f(stuff):
        x = stuff["x"]
        y, z = stuff["yz"]
        a = normal(x, 1)
        b = normal(a, y)
        c = normal(b, z)
        return ({"a": a}, b, c)

    stuff = {"x": 1.1, "yz": (2.2 * np.ones(5), 3.3)}
    in_axes = {'x': None, 'yz': (0, None)}
    out = vmap(f, in_axes=in_axes, axis_size=5)(stuff)
    assert out[0]["a"].shape == (5,)
    assert out[1].shape == (5,)
    assert out[2].shape == (5,)

def test_vmap_dict_prefix():
    def f(stuff):
        x = stuff["x"]
        y, z = stuff["yz"]
        a = normal(x, 1)
        b = normal(a, y)
        c = normal(b, z)
        return ({"a": a}, b, c)

    stuff = {"x": 1.1, "yz": (2.2 * np.ones(5), np.ones(5))}
    in_axes = {'x': None, 'yz': 0}
    out = vmap(f, in_axes=in_axes, axis_size=5)(stuff)
    assert out[0]["a"].shape == (5,)
    assert out[1].shape == (5,)
    assert out[2].shape == (5,)


def test_vmap4():
    def f():
        return normal(np.array(1), np.array(2))

    y = vmap(f, in_axes=None, axis_size=3)()
    assert y.shape == (3,)


def test_vmap5():
    def f(loc, scale):
        x = normal(loc, scale)
        y = normal(x, scale)
        return (x, y)

    loc = np.array([1, 2, 3])
    scale = np.array([4, 5, 6])
    x, y = vmap(f, in_axes=(0, 0), axis_size=3)(loc, scale)
    assert x.shape == (3,)
    assert y.shape == (3,)


def test_vmap6():
    def f(loc, scale):
        x = normal(loc, scale)
        y = normal(x, scale)
        return (x, y)

    loc = np.array([1, 2, 3])
    scale = np.array([4, 5, 6])
    x, y = vmap(f, in_axes=(0, 0), axis_size=None)(loc, scale)
    assert x.shape == (3,)
    assert y.shape == (3,)


def test_vmap7():
    def f(loc, scale):
        x = normal(loc, scale)
        y = normal(x, scale)
        return (x, y)

    loc = np.array([1, 2, 3])
    scale = np.array([4, 5, 6])
    x, y = vmap(f, in_axes=0, axis_size=None)(loc, scale)
    assert x.shape == (3,)
    assert y.shape == (3,)


def test_vmap8():
    def f(stuff):
        x = stuff["x"]
        y, z = stuff["yz"]
        a = normal(x, 1)
        b = normal(a, y)
        c = normal(b, z)
        return ({"a": a}, b, c)

    stuff = {"x": 1.1, "yz": (2.2, 3.3)}
    out = vmap(f, in_axes=None, axis_size=5)(stuff)
    assert out[0]["a"].shape == (5,)
    assert out[1].shape == (5,)
    assert out[2].shape == (5,)


def test_vmap10():
    def f(x):
        loc = x * 1.1
        scale = x**2.2
        return {"y": normal(loc, scale)}

    stuff = vmap(f, 0, None)(np.array([3.3, 4.4, 5.5]))
    assert stuff["y"].shape == (3,)


def test_vmap11():
    def f(x):
        loc = x * 1.1
        scale = x**2.2
        y = normal(loc, scale)
        x = normal(0, 1)
        z = normal(1, 2)
        return {"y": y, "xz": (x, z)}

    stuff = vmap(f, 0, None)(np.array([3.3, 4.4, 5.5]))
    # fancy pattern matching
    match stuff:
        case {"y": y, "xz": (x, z)}:
            assert y.shape == (3,)
            assert x.shape == (3,)
            assert z.shape == (3,)
        case _:
            assert False, "should be impossible"


def test_vmap12():
    loc = 0.5

    def f(scale):
        return normal(loc, scale)

    x = vmap(f, 0, None)(np.array([2.2, 3.3, 4.4]))
    assert x.shape == (3,)


def test_vmap13():
    loc = 0.5
    scale = 1.3

    def f():
        return normal(loc, scale)

    x = vmap(f, None, 3)()
    assert x.shape == (3,)


def test_vmap14():
    x = normal(1.1, 2.2)
    y = vmap(lambda: normal(x, 1), None, 3)()
    assert y.shape == (3,)


def test_vmap15():
    x = normal(0, 1)
    y, z = vmap(
        lambda: (yi := normal(x, 2), zi := vmap(lambda: normal(yi, 3), None, 5)()),
        None,
        3,
    )()
