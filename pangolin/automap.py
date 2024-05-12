import numpy as np

from .ir import RV, AbstractCondDist, VMapDist, makerv, CondDist, Index, Constant
from typing import Sequence, List, Self
from . import util


from . import inference_numpyro_modelbased


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

def automap(x):
    """
    Transform a sequence of RVs into a single VMapDist RV and transform those
    individual RVs into Indexes into that VMapDist RV.

    Also, recurse onto parents where possible/necessary.
    """

    assert is_sequence(x)

    # recurse if necessary
    if all(is_sequence(xi) for xi in x):
        return automap([automap(xi) for xi in x])

    if all(isinstance(xi.cond_dist, Constant) for xi in x):
        # stack together constant parents (no need to reassign)
        vals = [xi.cond_dist.value for xi in x]
        #return makerv(np.stack(vals))
        return makerv(np.array(vals))

    dist, N, M = check_parents_compatible(x)

    # get arguments
    v = [None] * M
    k = [None] * M
    for m in range(M):
        p = [xn.parents[m] for xn in x]
        v[m], k[m] = vec_args(p)


    # if you're vmapping over an Index
    # and you're vmapping over one argument only
    # and that argument has a cond_dist of Constant(range(N))
    # if you're mapping over an Index
    # then skip the vmap, and just turn that argument into a slice instead!

    # if isinstance(dist,Index):
    #     if num_non_none(k)==1: # no mapped arguments

    # I think the reason test21 fails is that this is TOO EAGER and accepts cases it SHOULDN'T ACCEPT

    # if isinstance(dist, Index):
    #     if num_non_none(k) == 1:  # one mapped argument
    #         in_axis = where_non_none(k)
    #         if in_axis > 0:  # don't do this when mapping over argument itself
    #             if k[in_axis] == 0:  # only map over dim 0 (redundant?)
    #                 if v[in_axis].cond_dist == Constant(range(N)):
    #                     where_slice = which_slice_kth_arg(dist, in_axis)
    #                     if where_slice == 0:
    #                         # only do this when mapping over first dim
    #                         # (otherwise the vmap acts like a transpose)
    #
    #                         print("WE GOT A CONSTANT")
    #                         print(dist)
    #                         print(k)
    #                         print(v)
    #                         print(f"{where_slice=}")
    #
    #                         assert dist.slices[where_slice] is None
    #                         new_slices = dist.slices[:where_slice] + (slice(None),) + dist.slices[where_slice+1:]
    #                         new_v = v[:in_axis] + v[in_axis+1:]
    #
    #                         if all(s == slice(None) for s in new_slices) and len(new_v) == 1:
    #                             # if you're slicing all dims, don't
    #                             return new_v[0]
    #
    #                         return Index(*new_slices)(*new_v)

    # create new vmapped RV
    new_rv = VMapDist(dist, k, N)(*v)

    if is_pointless_rv(new_rv):
        return new_rv.parents[0]

    # assign old RVs to be slices of new vmapped RV
    for n, xn in enumerate(x):
        xn.reassign(new_rv[n])

    return new_rv


def vec_args(p):
    p0 = p[0]
    d0 = p0.cond_dist
    if all(pi == p0 for pi in p):
        # if all parents are the same, return the first one and don't map
        return (p0, None)
    if all(pi.cond_dist == p0.cond_dist and pi.parents == p0.parents for pi in p):
        #raise Exception("new optimization!")
        return (p0, None)
    if isinstance(d0, Constant) and all(pi.cond_dist == d0 for pi in p):
        # if all parents are different but are equal constants, return the
        # first one and don't map
        return (p0, None)
    try:
        # if all parents are indices into the same RV with a single shared
        # unsliced dimension and constant indices running from 0 to N
        # then map over that dimension of the RV and discard the constant indices

        assert isinstance(p0.cond_dist, Index)
        k = get_unsliced(p0.cond_dist)
        v = p0.parents[0]
        for n, pn in enumerate(p):
            assert isinstance(pn.cond_dist, Index)
            assert get_unsliced(pn.cond_dist) == k
            assert pn.parents[0] == v
            assert pn.parents[1].cond_dist == Constant(n)
        return v, k
    except AssertionError: # if none of that worked (nodes not all same and not Index), recurse
        p_vec = automap(p)
        return vec_args(p_vec)