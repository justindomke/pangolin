from .ir import RV, AbstractCondDist, VMapDist, makerv, CondDist, Index, Constant
from typing import Sequence, List, Self
from . import util


def automap(x):
    assert isinstance(x, Sequence)
    if all(isinstance(xi, Sequence) for xi in x):
        return automap([automap(xi) for xi in x])

    N = len(x)
    M = len(x[0].parents)
    dist = x[0].cond_dist
    for xi in x:
        assert xi.cond_dist == dist
        assert len(xi.parents) == M

    v = []
    k = []
    for m in range(M):
        p = [xn.parents[m] for xn in x]
        my_v, my_k = vec_args(p)
        v.append(my_v)
        k.append(my_k)
    return VMapDist(dist, k, N)(*v)


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


def vec_args(p):
    p0 = p[0]
    d0 = p0.cond_dist
    if all(pi == p0 for pi in p):
        return (p0, None)
    if isinstance(d0, Constant) and all(pi.cond_dist == d0 for pi in p):
        return (p0, None)
    try:
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
        # DANGER: WHAT IF THE SAME THING APPEARS IMPLICITLY IN MULTIPLE PLACES?
        # THEN THIS WILL BE DIFFERENT IN EACH PLACE

        # nodes not all same and not all Index
        p_vec = automap(p)
        p_new = [p_vec[i] for i in range(p_vec.shape[0])]
        return vec_args(p_new)
