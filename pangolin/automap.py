import numpy as np

from .ir import RV, AbstractCondDist, VMapDist, makerv, CondDist, Index, Constant
from typing import Sequence, List, Self
from . import util

import pangolin

from . import inference_numpyro_modelbased

def where_single_slice(d):
    if not isinstance(d,Index):
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


def check_parents_compatible(x):
    N = len(x)
    M = len(x[0].parents)
    dist = x[0].cond_dist
    for xi in x:
        assert xi.cond_dist == dist
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


# when can a stack of vmap and index be collapsed?
# If:
# - The first parent is never mapped
# - Each other parent is mapped exactly once
# - Each other parent is "full" (corresponds to the full range)
# - The slices and mapped axes somehow work out so that everything eventually permutes into the right place
# How to check this?
# - Perhaps the easiest way is just to "run" it?
# - Like... actually run it?
# - If you start with unique values and end with the same values then you're solid!

def is_pointless_rv(x):
    d = x.cond_dist
    while isinstance(d,VMapDist):
        d = d.base_cond_dist
    if not isinstance(d,Index):
        return False
    val = np.arange(np.product(x.shape)).reshape(x.shape)
    val = makerv(val)
    if val.shape != x.parents[0].shape:
        return False
    if not all(isinstance(p.cond_dist,Constant) for p in x.parents[1:]):
        return False

    y = RV(x.cond_dist,val,*x.parents[1:])

    [samp] = inference_numpyro_modelbased.ancestor_sample_flat([y])

    print(f"{samp=}")
    print(f"{val.cond_dist.value=}")

    return samp.shape == x.shape and np.all(samp == val.cond_dist.value)

    #def ancestor_sample_flat(vars, *, niter=None):

def automap(x,check_validity):
    """
    Transform a sequence of RVs into a single VMapDist RV and transform those
    individual RVs into Indexes into that VMapDist RV.

    Also, recurse onto parents where possible/necessary.
    """

    if check_validity:
        assert is_sequence(x)

    # recurse if necessary
    if all(is_sequence(xi) for xi in x):
        return automap([automap(xi,check_validity) for xi in x],check_validity)


    if all(isinstance(xi.cond_dist, Constant) for xi in x):
        # stack together constant parents (no need to reassign)
        vals = [xi.cond_dist.value for xi in x]
        tmp = makerv(np.array(vals))
        return tmp


    dist, N, M = check_parents_compatible(x)

    # # get arguments
    # v = [None] * M
    # k = [None] * M
    # for m in range(M):
    #     p = [xn.parents[m] for xn in x]
    #     v[m], k[m] = vec_args(p,check_validity)

    # get arguments
    v = []
    k = []
    for m in range(M):
        p = [xn.parents[m] for xn in x]
        my_v, my_k = vec_args(p,check_validity)
        v.append(my_v)
        k.append(my_k)


    # equivalent list comprehensions—work but doesn't seem to be faster
    # v,k = zip(*[vec_args([xn.parents[m] for xn in x],check_validity) for m in range(M)])

    # create new vmapped RV
    new_rv = VMapDist(dist, k, N)(*v)

    if is_pointless_rv(new_rv):
        # print("about to return:")
        # pangolin.print_upstream(new_rv.parents[0])
        return new_rv.parents[0]

    # assign old RVs to be slices of new vmapped RV
    for n, xn in enumerate(x):
        xn.reassign(new_rv[n])

    # print("about to return:")
    # pangolin.print_upstream(new_rv)

    return new_rv


def vec_args(p,check_validity):
    p0 = p[0]
    d0 = p0.cond_dist
    if all(pi == p0 for pi in p):
        # if all parents are the same, return the first one and don't map
        return (p0, None)
    # seems wrong?
    # if all(pi.cond_dist == p0.cond_dist and pi.parents == p0.parents for pi in p):
    #     raise Exception("new optimization!")
    #     return (p0, None)
    # if isinstance(d0, Constant) and all(pi.cond_dist == d0 for pi in p):
    #     # if all parents are different but are equal constants, return the
    #     # first one and don't map
    #     return (p0, None)
    try:
        # if all parents are indices into the same RV with a single shared
        # unsliced dimension and constant indices running from 0 to N
        # then map over that dimension of the RV and discard the constant indices

        if check_validity:
            assert isinstance(p0.cond_dist, Index)
        k = get_unsliced(p0.cond_dist)
        v = p0.parents[0]
        if check_validity:
            for n, pn in enumerate(p):
                assert isinstance(pn.cond_dist, Index)
                assert get_unsliced(pn.cond_dist) == k
                assert pn.parents[0] == v
                assert pn.parents[1].cond_dist == Constant(n)
        return v, k
    except AssertionError: # if none of that worked (nodes not all same and not Index), recurse
        p_vec = automap(p,check_validity)
        return vec_args(p_vec,check_validity)



def roll(x):
    """
    Slimmed-down version, no error checking
    """

    # recurse if necessary
    if all(is_sequence(xi) for xi in x):
        return roll([roll(xi) for xi in x])

    if all(isinstance(xi.cond_dist, Constant) for xi in x):
        # stack together constant parents (no need to reassign)
        return makerv(np.array([xi.cond_dist.value for xi in x]))

    N = len(x)
    M = len(x[0].parents)
    dist = x[0].cond_dist

    # get arguments
    v = []
    k = []
    for m in range(M):
        p = [xn.parents[m] for xn in x]
        my_v, my_k = roll_args(p)
        v.append(my_v)
        k.append(my_k)


    # create new vmapped RV
    new_rv = VMapDist(dist, k, N)(*v)

    if is_pointless_rv(new_rv):
        # print("about to return:")
        # pangolin.print_upstream(new_rv.parents[0])
        return new_rv.parents[0]

    # assign old RVs to be slices of new vmapped RV
    for n, xn in enumerate(x):
        xn.reassign(new_rv[n])

    # print("about to return:")
    # pangolin.print_upstream(new_rv)

    return new_rv


def roll_args(p):
    p0 = p[0]
    d0 = p0.cond_dist
    if all(pi == p0 for pi in p):
        # if all parents are the same, return the first one and don't map
        return (p0, None)
    try:
        k = get_unsliced(p0.cond_dist)
        v = p0.parents[0]
        return v, k
    except AssertionError: # if none of that worked (nodes not all same and not Index), recurse
        p_vec = roll(p)
        return roll_args(p_vec)
