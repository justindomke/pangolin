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
from pangolin.automap import automap, automap_flat, AutomapError
from pangolin.arrays import Array
from pangolin.calculate import Calculate
from pangolin import inference_stan, inference_numpyro_modelbased

def test_flat_normal_rv_parents():
    loc = makerv(0)
    scale = makerv(1)
    x = automap_flat(normal(loc,scale) for i in range(5))
    print_upstream(x)
    assert x.cond_dist == VMapDist(normal_scale,(None,None),5)
    assert x.parents == (loc,scale)

def test_flat_normal_numerical_parent():
    """
    This also works
    """
    loc = 0
    scale = makerv(1)
    x = automap_flat(normal(loc, scale) for i in range(5))
    print_upstream(x)
    assert x.cond_dist == VMapDist(normal_scale, (None, None), 5)
    assert x.parents[0].cond_dist == Constant(0)
    assert x.parents[1] == scale

def test_flat_normal_array_parent():
    """
    This also works
    """
    loc = 0
    scales = makerv([1.1,2.2,3.3,4.4,5.5])
    x = automap_flat(normal(loc, scale) for scale in scales)
    print_upstream(x)
    assert x.cond_dist == VMapDist(normal_scale, (None, 0), 5)
    assert x.parents[0].cond_dist == Constant(0)
    assert x.parents[1] == scales

def test_flat_normal_list_parent_error():
    """
    This shouldn't work—requires recursion to turn scales into a RV
    """
    loc = 0
    scales = [1.1,2.2,3.3,4.4,5.5]
    try:
        x = automap_flat(normal(loc, scale) for scale in scales)
        assert False
    except AutomapError:
        pass # raised exception as expected


def test_flat_vmap_dists():
    args = makerv([[1,2,3],[4,5,6]])
    big = vmap(vmap(exponential))(args)
    print(big[0])
    print(big[1])
    z = automap_flat([big[0],big[1]])
    print_upstream(z)


