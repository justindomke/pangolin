from pangolin.interface import (
    RV,
    normal,
    normal_scale,
    makerv,
    exp,
    Constant,
    vmap,
    VMapDist,
    viz_upstream,
    print_upstream,
    add,
)
import numpy as np  # type: ignore
from pangolin.loops import Loop, SlicedRV, slice_existing_rv, make_sliced_rv, VMapRV
from pangolin import *
from pangolin.automap import automap


def test1():
    loc = makerv(0)
    scale = makerv(1)
    x = [normal(loc, scale) for i in range(5)]
    y = automap(x)
    assert y.cond_dist == VMapDist(normal_scale, (None, None), 5)
    assert y.parents == (loc, scale)


def test2():
    loc = makerv(0)
    scale = makerv([1.0, 2, 3, 4, 5])
    x = [normal(loc, scale[i]) for i in range(5)]
    y = automap(x)
    assert y.cond_dist == VMapDist(normal_scale, (None, 0), 5)
    assert y.parents == (loc, scale)


def test3():
    loc = makerv(0)
    scale = makerv(1)
    x = [[normal(loc, scale) for i in range(5)] for i in range(3)]
    y = automap(x)
    assert y.cond_dist == VMapDist(
        VMapDist(normal_scale, (None, None), 5), (None, None), 3
    )
    assert y.parents == (loc, scale)


def test4():
    x = [normal(0, 1) for i in range(5)]
    y = automap(x)
    assert y.cond_dist == VMapDist(normal_scale, (None, None), 5)
    assert y.parents[0].cond_dist == Constant(0)
    assert y.parents[1].cond_dist == Constant(1)


def test5():
    x = [exp(normal(0, 1)) for i in range(5)]
    y = automap(x)
    print_upstream(y)
    assert y.cond_dist == VMapDist(exp, (0,), 5)
    [z] = y.parents
    assert z.cond_dist == VMapDist(normal_scale, [None, None], 5)
    assert z.parents[0].cond_dist == Constant(0)
    assert z.parents[1].cond_dist == Constant(1)

    # assert y.cond_dist == VMapDist(normal_scale, (None, None), 5)
    # assert y.parents[0].cond_dist == Constant(0)
    # assert y.parents[1].cond_dist == Constant(1)
