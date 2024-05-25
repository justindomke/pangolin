import numpy as np

from .ir import RV, AbstractCondDist, VMapDist, makerv, CondDist, Index, Constant
from typing import Sequence, List, Self
from . import util
from types import GeneratorType

import pangolin

from pangolin import vmap

from . import inference_numpyro_modelbased

def is_sequence(x):
    return isinstance(x, Sequence) or (isinstance(x, np.ndarray))

def deep_equal(x,y):
    assert isinstance(x,RV)
    assert isinstance(y,RV)
    if x.cond_dist.random:
        return x is y
    return x.cond_dist == y.cond_dist and len(x.parents) == len(y.parents) and all(deep_equal(p_x,p_y) for p_x,p_y in zip(x.parents,y.parents))

def get_unsliced(d):
    assert isinstance(d, Index)
    count = 0
    where_unsliced = None
    for n, s in enumerate(d.slices):
        if not isinstance(s, slice):
            count += 1
            where_unsliced = n
    assert count == 1
    return where_unsliced

class AutomapError(Exception):
    def __init__(self, message=None):
        if message is None:
            super().__init__()
        else:
            super().__init__(message)


def vec_args_flat(p):
    assert is_sequence(p)
    assert all(isinstance(pi, RV) for pi in p)

    p0 = p[0]
    if all(pi is p0 for pi in p):
        return (p0, None)
    if all(deep_equal(pi,p0) for pi in p):
        return (p0, None)

    #if not all(isinstance(pi.cond_dist, Index) for pi in p):
    for pi in p:
        if not isinstance(pi.cond_dist, Index):
            raise AutomapError(f"vec_args_flat not unmapped and not Index instead {pi.cond_dist}")

    k = get_unsliced(p0.cond_dist)
    v = p0.parents[0]

    for n, pn in enumerate(p):
        if get_unsliced(pn.cond_dist) != k:
            raise AutomapError("vec_args_flat got incoherent unsliced dims")

        if pn.parents[0] != v:
            raise AutomapError("vec_args_flat got incoherent Index parents")

        if pn.parents[1].cond_dist != Constant(n):
            raise AutomapError("vec_args_flat got Constant(m) with m != n")

    return v, k


def automap_flat(x):
    """
    Automap a single (non-recursive) sequence that doesn't require any recursion
    """
    if isinstance(x, GeneratorType):
        x = tuple(x)

    if not is_sequence(x):
        raise AutomapError("automap_flat given non-sequence")

    if not all(isinstance(xn,RV) for xn in x):
        raise AutomapError("automap_flat given non-RVs in sequence")

    for xi in x:
        if len(xi.parents) != len(x[0].parents):
            raise AutomapError("automap_flat given arguments with differing numbers of parents")
        if xi.cond_dist != x[0].cond_dist:
            raise AutomapError("automap_flat given RVs with non-matching cond dists")

    # special case to bash indices together
    # this is a "dangerous optimization" because it uses values that might be different in different branches
    # and the problems this creates might not be discernable for a long time
    if all(isinstance(xn.cond_dist,Index) for xn in x):
        if len(x[0].parents)==2:
            if all(xn.parents[0]==x[0].parents[0] for xn in x):
                if all(xn.parents[1].cond_dist == Constant(n) for n,xn in enumerate(x)):
                    return x[0].parents[0]

    # special case to construct larger Constants
    # if all(isinstance(xi.cond_dist, Constant) for xi in x):
    #     vals = [xi.cond_dist.value for xi in x]
    #     if not all(val.shape == val[0].shape for val in vals):
    #         raise AutomapError("automap_flat given Constants of differing shape")
    #     return makerv(np.array(vals))


    v = []
    k = []
    for m in range(len(x[0].parents)):
        p = [xn.parents[m] for xn in x]
        my_v, my_k = vec_args_flat(p)
        v.append(my_v)
        k.append(my_k)

    new_rv = VMapDist(x[0].cond_dist, k, len(x))(*v)
    return new_rv


def automap(x):
    if isinstance(x, GeneratorType):
        x = tuple(x)

    assert is_sequence(x)

    if all(is_sequence(xi) for xi in x):
        new_x = automap(xi for xi in x)
        # now, recursion should never be required—just do a flat map

        # try:
        #     new_x = [automap_recursive(xi, indent=indent + 1, check=check, toplevel=toplevel, opt=opt, verbose=verbose,
        #                            trace=trace, nomerge=nomerge) for i, xi in enumerate(x)]
        #     return automap_recursive(new_x, indent=indent + 1, check=check, toplevel=toplevel, opt=opt, verbose=verbose,
        #                          trace=trace, nomerge=nomerge)
        # except UnmergableParentsError:
        #     new_x = [automap_recursive(xi, indent=indent + 1, check=check, toplevel=toplevel, opt=0, verbose=verbose,
        #                                trace=trace, nomerge=nomerge) for i, xi in enumerate(x)]
        #     return automap_recursive(new_x, indent=indent + 1, check=check, toplevel=toplevel, opt=0, verbose=verbose,
        #                              trace=trace, nomerge=nomerge)