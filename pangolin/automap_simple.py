import numpy as np

from .ir import RV, AbstractCondDist, VMapDist, makerv, CondDist, Index, Constant
from typing import Sequence, List, Self
from . import util
from types import GeneratorType

import pangolin

from pangolin import vmap

from . import inference_numpyro_modelbased


"""Ultra robust version of rolling that never does anything dangerous
"""

def where_single_slice(d):
    if not isinstance(d, Index):
        return -1
    count = 0
    where_unsliced = None
    for n, s in enumerate(d.slices):
        if not isinstance(s, slice):
            count += 1
            where_unsliced = n
    if count > 1:
        return -1
    return where_unsliced


def num_unsliced(d: Index):
    assert isinstance(d, Index)
    count = 0
    for n, s in enumerate(d.slices):
        if not isinstance(s, slice):
            count += 1
    return count


def get_unsliced(d: Index):
    assert isinstance(d, Index)
    count = 0
    where_unsliced = None
    for n, s in enumerate(d.slices):
        if not isinstance(s, slice):
            count += 1
            where_unsliced = n
    assert count == 1
    return where_unsliced


def num_non_none(l):
    assert isinstance(l, Sequence)
    count = 0
    for li in l:
        if li is not None:
            count += 1
    return count


def where_non_none(l):
    assert num_non_none(l) == 1
    for n, li in enumerate(l):
        if li is not None:
            return n
    assert False, "should be impossible"


def full_range_rv(x):
    if len(x.shape) != 1:
        return False
    [N] = x.shape
    return x.cond_dist == Constant(range(N))


def is_sequence(x):
    return isinstance(x, Sequence) or (isinstance(x, np.ndarray))


class UnmergableParentsError(Exception):
    pass

def check_parents_compatible(x,*,indent):
    N = len(x)
    M = len(x[0].parents)
    dist = x[0].cond_dist
    # print(f"{dist=}")
    # print(f"{tuple(xi.cond_dist for xi in x)=}")
    # print(f"{[p.cond_dist for p in x[0].parents]=}")
    # print(f"{[[p.cond_dist for p in xi.parents] for xi in x]=}")
    # print(f"{[p.shape for p in x[0].parents]=}")
    # print(f"{[[p.shape for p in xi.parents] for xi in x]=}")
    # for xi in x:
    #     assert xi.cond_dist == dist
    #     assert len(xi.parents) == M
    for xi in x:
        #assert xi.cond_dist == dist
        if xi.cond_dist != dist:
            #print(f"{' '*indent}UNMERGABLE!")
            #print(f"UNMERGABLE {trace}")
            raise UnmergableParentsError()
        assert len(xi.parents) == M

    return dist, N, M


def which_slice_kth_arg(d: Index, k):
    assert k > 0, "0th arg is for variable being indexed"
    nones_seen = 0
    for where in range(len(d.slices)):
        if d.slices[where] is None:
            if nones_seen == k - 1:
                return where
            nones_seen += 1
    assert False, "should be impossible"


def is_pointless_rv(x):
    d = x.cond_dist
    while isinstance(d, VMapDist):
        d = d.base_cond_dist
    if not isinstance(d, Index):
        return False
    val = np.arange(np.product(x.shape)).reshape(x.shape)
    val = makerv(val)
    if val.shape != x.parents[0].shape:
        return False
    if not all(isinstance(p.cond_dist, Constant) for p in x.parents[1:]):
        return False

    y = RV(x.cond_dist, val, *x.parents[1:])

    [samp] = inference_numpyro_modelbased.ancestor_sample_flat([y])

    # print(f"{samp=}")
    # print(f"{val.cond_dist.value=}")

    return samp.shape == x.shape and np.all(samp == val.cond_dist.value)

    # def ancestor_sample_flat(vars, *, niter=None):

def equal_along_axis(x,axis):
    # holy hell this is awkward
    ref = x[(slice(None),)*axis + (0,)][(slice(None),)*axis + (np.newaxis,)]
    return np.all(x==ref)

def remove_redundant_dimension(arr, axis):
    assert equal_along_axis(arr, axis)
    return arr[(slice(None),)*axis + (0,)]



def automap(x, *, indent=0, check=True, toplevel=True, verbose=False):
    """
    Transform a sequence of RVs into a single VMapDist RV and transform those
    individual RVs into Indexes into that VMapDist RV.

    Only allowed to modify random RVs when toplevel=true

    Also, recurse onto parents where possible/necessary.
    """

    if isinstance(x, GeneratorType):
        x = tuple(x)

    if verbose:
        print(f'{" " * (indent * 4)}automap called on {len(x)} inputs:',end='')
        def display(xi):
            s = ""
            if isinstance(xi, RV):
                print(
                    f"<{s}{xi.cond_dist} {xi.shape} {[p.shape for p in xi.parents]} {[p.cond_dist if isinstance(p, RV) else type(p) for p in xi.parents]}> ",end='')
            elif isinstance(xi, np.ndarray):
                print(f"<{s}{type(xi)} {xi.shape}> ", end='')
            else:
                print(f"<{s}{type(xi)}> ", end='')
        for xi in x:
            display(xi)
        print('')

    if check:
        assert is_sequence(x)

    #print(nomerge, trace)

    if all(is_sequence(xi) for xi in x):
        new_x = [automap(xi, indent=indent + 1, check=check, toplevel=toplevel, verbose=verbose) for i, xi in enumerate(x)]
        return automap(new_x, indent=indent + 1, check=check, toplevel=toplevel, verbose=verbose)

    if all(isinstance(xi.cond_dist, Constant) for xi in x):
        # stack together constant parents (no need to reassign)
        vals = [xi.cond_dist.value for xi in x]
        rez = makerv(np.array(vals))
        return rez

    dist, N, M = check_parents_compatible(x, indent=indent)

    if not toplevel and dist.random:
        raise ValueError("automap can only process random RVs at top level")

    # get arguments
    v = []
    k = []
    for m in range(M): # M is number of parents for each element of the sequence
        p = [xn.parents[m] for xn in x]
        my_v, my_k = vec_args(p, indent=indent, check=check, toplevel=toplevel, verbose=verbose)
        v.append(my_v)
        k.append(my_k)

    # equivalent list comprehensions—work but doesn't seem to be faster
    # v,k = zip(*[vec_args([xn.parents[m] for xn in x],check_validity) for m in range(M)])

    # create new vmapped RV
    new_rv = VMapDist(dist, k, N)(*v)

    if is_pointless_rv(new_rv):
        return new_rv.parents[0]

    if verbose:
        print(f'{" " * (indent * 4)}automap returning: {new_rv.cond_dist} {new_rv.shape}')

    return new_rv


def deep_equal(x,y):
    assert isinstance(x,RV)
    assert isinstance(y,RV)
    if x.cond_dist.random:
        return x is y
    return x.cond_dist == y.cond_dist and not isinstance(x.cond_dist,Constant) and len(x.parents) == len(y.parents) and all(deep_equal(p_x,p_y) for p_x,p_y in zip(x.parents,y.parents))


def vec_args(p,  *, indent=0, check, toplevel, verbose):
    p0 = p[0]
    d0 = p0.cond_dist

    if all(pi is p0 for pi in p):
        # if all parents are the same, return the first one and don't map
        return (p0, None)
    elif all(deep_equal(pi,p0) for pi in p):
        # also safe to map in this case!
        return (p0, None)
    try:
        # if all parents are indices into the same RV with a single shared
        # unsliced dimension and constant indices running from 0 to N
        # then map over that dimension of the RV and discard the constant indices

        if check:
            assert isinstance(p0.cond_dist, Index)
        k = get_unsliced(p0.cond_dist)
        v = p0.parents[0]
        # if check_validity:

        for n, pn in enumerate(p):
            assert isinstance(pn.cond_dist, Index)
            assert get_unsliced(pn.cond_dist) == k
            assert pn.parents[0] == v
            assert pn.parents[1].cond_dist == Constant(n)
        return v, k
    except AssertionError:  # if none of that worked (nodes not all same and not Index), recurse
        p_vec = automap(p, indent=indent + 1, check=check, toplevel=False, verbose=verbose)
        return p_vec, 0
