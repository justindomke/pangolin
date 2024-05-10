import numpy as np

from .ir import RV, AbstractCondDist, VMapDist, makerv, CondDist, Index, Constant
from typing import Sequence, List, Self
from . import util


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


def num_non_None(l):
    assert isinstance(l, Sequence)
    count = 0
    for li in l:
        if li is not None:
            count += 1
    return count


def where_non_None(l):
    assert num_non_None(l) == 1
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


# def is_pointless_rv(x):
#     if isinstance(x.cond_dist, Index):
#         assert len(x.cond_dist.slices) == len(x.shape)
#
#         where_nonsliced = 1  # start at 1 to skip array being indexed
#         for i in range(len(x.cond_dist.slices)):
#             s = x.cond_dist.slices[i]
#             if s is None:
#                 N = x.shape[i]
#                 p = x.parents[where_nonsliced]
#                 where_nonsliced += 1
#                 if p.cond_dist != Constant(range(N)):
#                     print("a", p.cond_dist)
#                     return False
#             elif s != slice(None):
#                 print("b", s)
#                 return False
#
#         return True
#     # elif isinstance(x,VMapDist)
#     return False


# def is_pointless_rv(x):
#     d = x.cond_dist
#     if isinstance(d, Index):
#         if num_unsliced(d) == 0:
#             return all(s == slice(None) for s in d.slices)
#         elif num_unsliced(d) == 1:
#             k = get_unsliced(x.cond_dist)
#             N = x.shape[k]
#             idx = x.parents[1]  # always
#             print(f"{N=} {idx=}")
#             return idx.cond_dist == Constant(np.arange(N))
#         else:
#             # double non-sliced indexing NEVER pointless
#             # because triggers numpy advanced indexing
#             return False
#
#     elif isinstance(d, VMapDist):
#         if num_non_None(d.in_axes) != 1:
#             return False
#         elif not isinstance(d.base_cond_dist, Index):
#             return False
#         elif d.in_axes[0] is not None:
#             print("array being indexed INTO is mapped")
#             return False
#         else:
#             k = where_non_None(d.in_axes)
#             return x.parents[k].cond_dist == Constant(np.arange(x.shape[k - 1]))
#
#     # elif isinstance(x,VMapDist)
#     return False


def is_pointless_rv(x):
    # take an n-dimensional x
    # it's pointless if:
    # - it's an Index inside of a stack of VMapDists
    # - the Index has no slices that aren't full slices
    # - first argument is never vmapped (that's the array x would be equivalent to)
    # - all other arguments are either ALWAYS slices or get mapped exactly once over axis 0
    # - all other arguments are ranges
    # - also need something about correct ORDER of mapping TODO TODO TODO

    # TODO
    # SUSPECT INSISTING in_axes just decrease is not necessary!?

    # make sure you have a stack of VMapDists on top of an Index and first arg never mapped
    d = x.cond_dist
    times_mapped = [0] * len(x.parents)
    while isinstance(d, VMapDist):
        if num_non_None(d.in_axes) == 0:
            pass
        elif num_non_None(d.in_axes) > 1:
            return False
        else:
            k = where_non_None(d.in_axes)
            if d.in_axes[k] != 0:
                return False
            if k == 0:
                return False
            times_mapped[k] += 1
            if times_mapped[k] > 1:
                return False
        d = d.base_cond_dist
    if not isinstance(d, Index) or any(
            isinstance(s, slice) and s != slice(None) for s in d.slices
    ):
        print("b")
        return False

    # print(f"{times_mapped=}")
    #
    # if times_mapped[0] != 0:
    #     print("c")
    #     return False
    #
    # for m in times_mapped[1:]:
    #     if m > 1:
    #         print("d")
    #         return False

    for p in x.parents[1:]:
        if p.ndim != 1:
            print("e")
            return False
        if p.cond_dist != Constant(range(p.shape[0])):
            print("f")
            return False

    return True


# s = d.in_axes[i]
# if isinstance(s, slice) and s != slice(None):
#     print("a")
#     return False
# assert isinstance(s, int)


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
        # no need to re-assign parents since they're constants.
        vals = [xi.cond_dist.value for xi in x]
        return makerv(np.stack(vals))

    dist, N, M = check_parents_compatible(x)

    # get arguments
    v = [None] * M
    k = [None] * M
    for m in range(M):
        p = [xn.parents[m] for xn in x]
        v[m], k[m] = vec_args(p)

    if isinstance(dist, Index):
        # print("IS INDEX")
        if all(s is None for s in dist.slices):
            # print("ALL UNSLICED")
            # print(f"{v=}")
            # print([full_range_rv(vi) for vi in v[1:]])
            assert len(v) == len(dist.slices) + 1
            if all(full_range_rv(vi) for vi in v[1:]):
                raise Exception("Thought this was obsolete!")
                print("ALL FULL")
                return v

    # if isinstance(dist, Index):
    #     if num_unsliced(dist) == 1:
    #         assert len(v) == 2
    #         if v[1].ndim == 1:
    #             if v[1].cond_dist == Constant(range(v[1].shape[0])):
    #                 print("no need for new cond dist!")
    #                 return v[0]

    # if isinstance(dist, Index):
    #     if all((s == slice(None) or s is None) for s in dist.slices):
    #         where_v = 0
    #         for n in range(len(dist.slices)):
    #             if dist.slices[n] is None:
    #                 if v[where_v].cond_dist == Constant(range(N)):
    #                     new_slices = (
    #                         dist.slices[:n] + (slice(None),) + dist.slices[n + 1 :]
    #                     )
    #                     dist = Index(new_slices)
    #                     print("DOING IT")
    #                     v = v[:where_v] + v[where_v + 1 :]
    #                     k = k[:where_v] + k[where_v + 1 :]
    #                 where_v += 1

    # if num_unsliced(dist) == 1:
    #     assert len(v) == 2
    #     if v[1].ndim == 1:
    #         if v[1].cond_dist == Constant(range(v[1].shape[0])):
    #             #print("no need for new cond dist!")
    #             #return v[0]

    # if you're vmapping over an Index
    # and you're vmapping over one argument only
    # and that argument has a cond_dist of Constant(range(N))
    # if you're mapping over an Index
    # then skip the vmap, and just turn that argument into a slice instead!

    if isinstance(dist, Index):
        if num_non_None(k) == 1:  # one mapped argument
            in_axis = where_non_None(k)
            if in_axis > 0:  # don't do this when mapping over argument itself
                if k[in_axis] == 0:  # only map over dim 0 (unnecessary?)
                    if v[in_axis].cond_dist == Constant(range(N)):
                        print("WE GOT SOMETHING TO OPTIMIZE!")
                        where_slice = in_axis - 1
                        assert dist.slices[where_slice] is None
                        new_slices = dist.slices[:where_slice] + (slice(None),) + dist.slices[where_slice+1:]
                        new_v = v[:in_axis] + v[in_axis+1:]
                        print(f"{dist=}")
                        print(f"{k=}")
                        #print(f"{v=}")
                        print(f"{[vi.shape for vi in v]=}")

                        print(f"{new_slices=}")
                        #print(f"{new_args=}")
                        print(f"{[vi.shape for vi in new_v]=}")
                        return Index(*new_slices)(*new_v)

    # create new vmapped RV
    new_rv = VMapDist(dist, k, N)(*v)
    # assign old RVs to be slices of new vmapped RV
    for n, xn in enumerate(x):
        xn.reassign(new_rv[n])

    # check: are you creating a pointless index RV?
    # if isinstance(new_rv.cond_dist, Index):
    #    print(f"INDEX: {new_rv=}")
    # if isinstance(new_rv.cond_dist, VMapDist):
    #     if isinstance(new_rv.cond_dist.base_cond_dist, Index):
    #         print(f"VMAP OVER INDEX: {new_rv=}")
    # if is_pointless_rv(new_rv):
    #     print("WE GOT OURSELVES A POINTLESS!")
    #     return v[0]

    return new_rv

    # return VMapDist(dist, k, N)(*v)


def vec_args(p):
    p0 = p[0]
    d0 = p0.cond_dist
    if all(pi == p0 for pi in p):
        # if all parents are the same, return the first one and don't map
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
    except AssertionError:
        # if none of that worked, then recurse

        # nodes not all same and not all Index
        p_vec = automap(p)

        # used to have this here, now seems unnecessary
        # for p_old, p_new in zip(p, p_vec):
        #    p_old.reassign(p_new)

        return vec_args(p_vec)


def destroy(x):
    if isinstance(x, Sequence):
        for xi in x:
            destroy(xi)
    else:
        x.clear()
