from pangolin import *
import pangolin.automap


class Array1D(RV):
    """
    An Array is a bunch of "slots" into which you can assign other RVs.
    Once all the slots are full it "becomes" a VMapDist
    """

    def __init__(self, length):
        # do not call super!
        assert isinstance(length, int)
        self.rvs = np.empty(length, dtype=RV)
        # self.num_empty = length
        self._shape = None  # overwritten when later initialized

    def __setitem__(self, idx, value):
        assert isinstance(value, RV)
        assert self.rvs[idx] is None, "can only assign into slot of Array once"

        if hasattr(value, "cond_dist"):
            # guess shape from first assignment
            expected_shape = (len(self.rvs),) + value.shape
            if self._shape is None:
                self._shape = expected_shape
            else:
                assert self._shape == expected_shape

            # make copy of RV
            self.rvs[idx] = RV(value.cond_dist, *value.parents)
            # make sure no one else uses rv
            value.clear()
        else:
            self.rvs[idx] = value
        # self.num_empty -= 1
        # if self.num_empty == 0:
        #    self.activate()
        if all(hasattr(r, "cond_dist") for r in self.rvs):
            self.activate()

    def activate(self):
        # NOW you become a real RV
        dist = self.rvs[0].cond_dist
        M = len(self.rvs[0].parents)
        for x in self.rvs:
            isinstance(x, RV)
            assert x.cond_dist == dist
            assert len(x.parents) == M
        v = []
        k = []
        for m in range(M):
            p = []
            for n, x in enumerate(self.rvs):
                p.append(x.parents[m])
            # print(f"{p=}")
            my_v, my_k = vec_args(p)
            v.append(my_v)
            k.append(my_k)
            # print(f"{my_v=} {my_k=}")
        new_dist = VMapDist(dist, k, len(self.rvs))
        super().__init__(new_dist, *v)


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
    par0 = p0.parents
    if all(pn.cond_dist == d0 for pn in p):  # and pn.parents == par0
        print("NEW CASE!")
        print([pn.parents for pn in p])

    if all(pi == p0 for pi in p):
        return (p0, None)
    if isinstance(d0, Constant) and all(pi.cond_dist == d0 for pi in p):
        return (p0, None)
    if all(isinstance(pi.cond_dist, Constant) for pi in p):
        vals = [pi.cond_dist.value for pi in p]
        return makerv(np.stack(vals)), 0

    assert isinstance(p0.cond_dist, Index)
    k = get_unsliced(p0.cond_dist)
    v = p0.parents[0]
    # print(f"{p=}")
    print(d0)
    print([pn.parents[0] for pn in p])
    print([pn.parents[1].cond_dist for pn in p])
    for n, pn in enumerate(p):
        assert isinstance(pn.cond_dist, Index)
        assert get_unsliced(pn.cond_dist) == k
        assert pn.parents[0] == v
        assert pn.parents[1].cond_dist == Constant(n)
    return v, k


class Array(RV):
    """
    Make Array1D do all the actual work
    """

    def __init__(self, shape):
        # do not call super!
        self.init_shape = shape
        self.arr = Array1D(shape[0])
        if len(shape) > 1:
            for i in range(shape[0]):
                self.arr[i] = Array(shape[1:])
        # self.num_empty = shape[0]
        self._shape = None

    def __setitem__(self, idx, value):
        # guess shape from first assignment
        expected_shape = self.init_shape + value.shape
        if self._shape is None:
            self._shape = expected_shape
        else:
            assert self._shape == expected_shape

        if len(self.init_shape) == 1:
            self.arr[idx] = value
        else:
            # self.arr[idx[0]][idx[1:]] = value
            child = self.arr.rvs[idx[0]]
            child[idx[1:]] = value
            if all([hasattr(x, "cond_dist") for x in self.arr.rvs]):
                self.arr.activate()

        if hasattr(self.arr, "cond_dist"):
            super().__init__(self.arr.cond_dist, *self.arr.parents)


class ArrayND(RV):
    """
    Make Array1D do all the actual work
    """

    def __init__(self, shape):
        # do not call super!
        self.init_shape = shape
        self.rvs = np.empty(shape, dtype=RV)
        self.num_empty = np.product(shape)
        self._shape = None

    def __setitem__(self, idx, value):
        assert isinstance(value, RV)
        assert self.rvs[idx] is None, "can only assign into slot of Array once"

        if hasattr(value, "cond_dist"):
            # guess shape from first assignment
            expected_shape = self.init_shape + value.shape
            if self._shape is None:
                self._shape = expected_shape
            else:
                assert self._shape == expected_shape

            # make copy of RV
            self.rvs[idx] = RV(value.cond_dist, *value.parents)
            # make sure no one else uses rv
            value.clear()
            self.num_empty -= 1
            if self.num_empty == 0:
                self.activate()

    def activate(self):
        print("ACTIVATING")
        print(f"{self.rvs=}")
        rv = pangolin.automap.automap(self.rvs)
        super().__init__(rv.cond_dist, *rv.parents)
