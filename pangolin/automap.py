import numpy as np

from .ir import RV, AbstractCondDist, VMapDist, makerv, CondDist, Index, Constant
from typing import Sequence, List, Self
from . import util
from types import GeneratorType

import pangolin

from pangolin import vmap

from . import inference_numpyro_modelbased


class EqualityHasher:
    # class to compute equalities recursively

    def __init__(self):
        # self.dict = util.TwoDict()
        self.dict = {}

    def eq(self, a, b):
        if (a, b) not in self.dict:
            self.dict[a, b] = self.equal(a, b)
        return self.dict[a, b]

    def equal(self, a, b):
        if a.cond_dist != b.cond_dist:
            return False
        d = a.cond_dist
        if d.random:
            return id(a) == id(b)
        # return len(a.parents) == len(b.parents) and all(self.equal(p1,p2) for p1,p2 in zip(a.parents,b.parents))
        return len(a.parents) == len(b.parents) and all(id(p1) == id(p2) for p1, p2 in zip(a.parents, b.parents))


# h = EqualityHasher()

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

def check_parents_compatible(x):
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
            print("UNMERGABLE!")
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


def map_if_needed(x):
    """
    If some of x are VMapDist and others are just dist (with a matching base_dist) (and deterministic)
    then transform the non-mapped things into maps
    """
    length = len(x)
    if all(xi.cond_dist == x[0].cond_dist for xi in x):
        return x

    if all(isinstance(xi.cond_dist, Constant) for xi in x):
        return x

    is_mapped = [isinstance(x[i].cond_dist, VMapDist) for i in range(length)]
    dists = [x[i].cond_dist.base_cond_dist if is_mapped[i] else x[i].cond_dist for i in range(length)]

    #print(f"{dists=}")
    assert all(d == dists[0] for d in dists)
    axis_size = None
    for i in range(length):
        if is_mapped[i]:
            axis_size = x[i].shape[0]
    assert axis_size is not None, "didn't find any mapped vars"
    new_x = [x[i] if is_mapped[i] else vmap(x[i].cond_dist, None, axis_size)(*x[i].parents) for i in range(length)]
    return new_x


def expand_if_needed(x):
    """
    given some set of vmapped x
    if some axis fails to be mapped on one but IS mapped on others
    AND the corresponding parent is a constant
    then blow up the constant to a larger size and map it
    """
    assert is_sequence(x)
    assert all(isinstance(xi.cond_dist, VMapDist) for xi in x)
    assert all(xi.cond_dist.base_cond_dist == x[0].cond_dist.base_cond_dist for xi in x)
    assert all(len(xi.parents) == len(x[0].parents) for xi in x)
    num_parents = len(x[0].parents)
    base_dist = x[0].cond_dist.base_cond_dist

    mapped_axes = [None] * num_parents
    mapped_shape = [None] * num_parents
    mapped_size = None
    for xi in x:
        for n in range(num_parents):
            if xi.cond_dist.in_axes[n] is not None:
                # TODO: check before overwrite
                mapped_axes[n] = xi.cond_dist.in_axes[n]
                mapped_shape[n] = xi.parents[n].shape[mapped_axes[n]]
                if xi.cond_dist.axis_size is not None:
                    mapped_size = xi.cond_dist.axis_size
    mapped_axes = tuple(mapped_axes)
    # print(f"{mapped_axes=}")
    # print(f"{mapped_shape=}")

    new_x = []
    for xi in x:
        if xi.cond_dist.in_axes != mapped_axes:
            new_parents = []
            for n in range(num_parents):
                if xi.cond_dist.in_axes[n] is None and mapped_axes[n] is not None:
                    assert isinstance(xi.parents[n].cond_dist, Constant)
                    # new_value = xi.parents[0]
                    new_value = np.repeat(xi.parents[n].cond_dist.value, mapped_shape[n], axis=mapped_axes[n])
                    new_parents.append(makerv(new_value))
                else:
                    new_parents.append(xi.parents[n])
            new_xi = VMapDist(base_dist, mapped_axes, mapped_size)(*new_parents)
            new_x.append(new_xi)
        else:
            new_x.append(xi)
    # print('-------------------------')
    # print(f"{[xi.cond_dist for xi in x]=}")
    # print(f"{[xi.cond_dist for xi in new_x]=}")

    return new_x

def automap(x, nomerge=False, *, indent=0, check=True):
    """
    Transform a sequence of RVs into a single VMapDist RV and transform those
    individual RVs into Indexes into that VMapDist RV.

    Also, recurse onto parents where possible/necessary.
    """

    if isinstance(x, GeneratorType):
        x = tuple(x)

    # print(f'{" " * (indent * 4)}automap({nomerge}) called on {len(x)} inputs:')
    # def display(xi):
    #     s = " " * (indent * 4)
    #     if isinstance(xi, RV):
    #         print(
    #             f"{s}-{xi.cond_dist} {xi.shape} {[p.shape for p in xi.parents]} {[p.cond_dist if isinstance(p, RV) else type(p) for p in xi.parents]}")
    #     elif isinstance(xi, np.ndarray):
    #         print(f"{s}-{type(xi)} {xi.shape}")
    #     else:
    #         print(f"{s}-{type(xi)}")
    # for xi in x:
    #     display(xi)

    if check:
        assert is_sequence(x)

    # recurse if necessary
    # if all(is_sequence(xi) for xi in x):
    #     try:
    #         new_x = [automap(xi, nomerge, indent=indent + 1, check=check) for xi in x]
    #         rez = automap(new_x, nomerge, indent=indent + 1, check=check)  # TODO: right?
    #         #print(f'\n{" " * (indent * 4)}automap returning: {rez.cond_dist} {rez.shape}')
    #         return rez
    #     except UnmergableParentsError:
    #         try:
    #             new_x = [automap(xi, True, indent=indent + 1, check=check) for xi in x]
    #             rez = automap(new_x, True, indent=indent + 1, check=check)  # TODO: right?
    #             #print(f'\n{" " * (indent * 4)}automap returning: {rez.cond_dist} {rez.shape}')
    #             return rez
    #         except UnmergableParentsError:
    #             raise UnmergableParentsError()


    if all(is_sequence(xi) for xi in x):
        try:
            new_x = [automap(xi, False, indent=indent + 1, check=check) for xi in x]
            try:
                return automap(new_x, False, indent=indent + 1, check=check)
            except UnmergableParentsError:
                return automap(new_x, True, indent=indent + 1, check=check)
        except UnmergableParentsError:
            new_x = [automap(xi, True, indent=indent + 1, check=check) for xi in x]
            try:
                return automap(new_x, False, indent=indent + 1, check=check)
            except UnmergableParentsError:
                return automap(new_x, True, indent=indent + 1, check=check)



    #x = map_if_needed(x)

    # try:
    #     x = expand_if_needed(x)
    # except AssertionError:
    #     pass

        # return automap([automap(xi,check_validity) for xi in x],check_validity)

    if all(isinstance(xi.cond_dist, Constant) for xi in x):
        # stack together constant parents (no need to reassign)
        vals = [xi.cond_dist.value for xi in x]
        rez = makerv(np.array(vals))
        #print(f'{" " * (indent * 4)}automap returning: {rez.cond_dist} {rez.shape}')
        return rez

    dist, N, M = check_parents_compatible(x)

    # get arguments
    v = []
    k = []
    for m in range(M):
        p = [xn.parents[m] for xn in x]
        my_v, my_k = vec_args(p, nomerge, indent=indent, check=check)
        v.append(my_v)
        k.append(my_k)

    # equivalent list comprehensions—work but doesn't seem to be faster
    # v,k = zip(*[vec_args([xn.parents[m] for xn in x],check_validity) for m in range(M)])

    # create new vmapped RV
    new_rv = VMapDist(dist, k, N)(*v)

    if is_pointless_rv(new_rv):
        return new_rv.parents[0]

    # DON'T re-assign old RVs
    # This is dangerous because could lead to dual usage. But it's the simplest way to make exceptions work

    #print(f'\n{" " * (indent * 4)}automap returning: {new_rv.cond_dist} {new_rv.shape}')

    return new_rv


def vec_args(p, nomerge, *, indent=0, check):
    p0 = p[0]
    d0 = p0.cond_dist
    if nomerge:
        if all(pi is p0 for pi in p):
            # if all parents are the same, return the first one and don't map
            return (p0, None)
    else:
        if all(pi == p0 for pi in p):
            # if all parents are the same, return the first one and don't map
            return (p0, None)
    # print(f"{p=}")

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
            assert pn.parents[0] == v  # THIS IS THE SLOW ONE!
            # assert h.eq(pn.parents[0],v)
            assert pn.parents[1].cond_dist == Constant(n)
        return v, k
    except AssertionError:  # if none of that worked (nodes not all same and not Index), recurse
        #try:
        #   p_vec = automap(p, False, indent=indent + 1, check=check)
        #except UnmergableParentsError:
        #    p_vec = automap(p, True, indent=indent + 1, check=check)
        #p_vec = automap(p, True, indent=indent + 1, check=check)
        #return vec_args(p_vec, True, indent=indent + 1, check=check)

        try:
            p_vec = automap(p, False, indent=indent + 1, check=check)
            try:
                return vec_args(p_vec, False, indent=indent + 1, check=check)
            except UnmergableParentsError:
                return vec_args(p_vec, True, indent=indent + 1, check=check)
        except UnmergableParentsError:
            p_vec = automap(p, True, indent=indent + 1, check=check)
            try:
                return vec_args(p_vec, False, indent=indent + 1, check=check)
            except UnmergableParentsError:
                return vec_args(p_vec, True, indent=indent + 1, check=check)

# def roll(x):
#     """
#     Slimmed-down version, no error checking
#     """
#
#     # recurse if necessary
#     if all(is_sequence(xi) for xi in x):
#         return roll([roll(xi) for xi in x])
#
#     if all(isinstance(xi.cond_dist, Constant) for xi in x):
#         # stack together constant parents (no need to reassign)
#         return makerv(np.array([xi.cond_dist.value for xi in x]))
#
#     N = len(x)
#     M = len(x[0].parents)
#     dist = x[0].cond_dist
#
#     # get arguments
#     v = []
#     k = []
#     for m in range(M):
#         p = [xn.parents[m] for xn in x]
#         my_v, my_k = roll_args(p)
#         v.append(my_v)
#         k.append(my_k)
#
#     # create new vmapped RV
#     new_rv = VMapDist(dist, k, N)(*v)
#
#     if is_pointless_rv(new_rv):
#         # print("about to return:")
#         # pangolin.print_upstream(new_rv.parents[0])
#         return new_rv.parents[0]
#
#     # assign old RVs to be slices of new vmapped RV
#     for n, xn in enumerate(x):
#         # DANGER DANGER DANGER
#         # could this ever change the semantics of equality?
#         # well, sure.
#         xn.reassign(new_rv[n])
#
#     # print("about to return:")
#     # pangolin.print_upstream(new_rv)
#
#     return new_rv
#
#
# def roll_args(p):
#     p0 = p[0]
#     d0 = p0.cond_dist
#     if all(pi == p0 for pi in p):
#         # if all parents are the same, return the first one and don't map
#         return (p0, None)
#     try:
#         k = get_unsliced(p0.cond_dist)
#         v = p0.parents[0]
#         return v, k
#     except AssertionError:  # if none of that worked (nodes not all same and not Index), recurse
#         p_vec = roll(p)
#         return roll_args(p_vec)
