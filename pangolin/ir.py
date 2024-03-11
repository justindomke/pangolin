"""
The core of Pangolin: The intermediate representation that is used to represent probabilistic models. This representation is very simple: `CondDist` objects represent conditional distributions. `RV` objects represent random variables. A random variable just has two members: A `CondDist` and a list of parent `RV`s.
"""

from abc import ABC, abstractmethod

import numpy
from . import dag, util
import jax.tree_util
from collections import defaultdict

np = numpy  # by default use regular numpy but users can, e.g., set ir.np = jax.numpy


class CondDist(ABC):
    """
    A `CondDist` represents a conditional distribution. Note that it *only*
    represents it—all functionality for sampling or density evaluation, etc. is left
    to inference engines.
    * Frozen after creation.
    * `__eq__` should be defined so that *mathematically equivalent*
    `CondDist`s are equal, regardless of if they occupy the same place in memory.
    Unnecessary for pre-defined `CondDist` objects like `normal_scale` or `add`,
    but needed for `CondDist`s that are constructed with parameters, like `VMapDist`.
    * Some `CondDist`s, e.g. multivariate normal distributions, can have different
    shapes depending on the shapes of the parents. So a concrete `CondDist` must
    provide a `get_shape(*parents_shapes)` method to resolve this.
    """

    _frozen = False

    def __init__(self, name, random):
        assert isinstance(name, str)
        assert isinstance(random, bool)
        self.name = name
        self.random = random
        self._frozen = True  # freeze after init

    @abstractmethod
    def get_shape(self, *parents_shapes):
        pass

    def __setattr__(self, key, value):
        if self._frozen:
            raise TypeError("CondDists are immutable after init.")
        else:
            self.__dict__[key] = value

    def __call__(self, *parents):
        """when you call a conditional distribution you get a RV"""
        parents = tuple(makerv(p) for p in parents)

        return make_sliced_rv(self, *parents, all_loops=Loop.loops)

    def __repr__(self):
        return self.name


class Constant(CondDist):
    """
    Represents a "constant" distribution. Has no parents. Data is always stored as a
    numpy array. You can switch it to use jax's version of numpy by setting `ir.np =
    jax.numpy`.
    """

    def __init__(self, value):
        """
        `Constant` distributions are initialized with the constant value they should
        represent. This can be anything that can be converted to a numpy array by
        `numpy.array`.
        """
        self.value = np.array(value)
        """The actual stored data, stored as an immutable numpy array"""
        if np == numpy:
            self.value.flags.writeable = False
        super().__init__(name="constant", random=False)

    def get_shape(self):
        """"""
        return self.value.shape

    def __eq__(self, other):
        if isinstance(other, Constant):
            return self.value.shape == other.value.shape and np.all(
                self.value == other.value
            )
        return False

    def __hash__(self):
        return hash(self.value.tobytes())

    def __repr__(self):
        array_str = repr(self.value)  # get base string
        array_str = array_str[6:-1]  # cut off "array(" and ")"
        array_str = array_str.replace("\n", "")  # remove newlines
        array_str = array_str.replace(" ", "")  # remove specs
        return "Constant(" + array_str + ")"

    def __str__(self):
        # return str(self.value).replace("\n", "").replace("  ", " ")
        return np.array_str(self.value, precision=3).replace("\n", "").replace("  ", " ")


class AllScalarCondDist(CondDist):
    """
    Convenience class to create "all scalar" distributions, where all parents and
    outputs are scalar.
    """

    def __init__(self, num_parents, name, random):
        self.num_parents = num_parents
        super().__init__(name=name, random=random)

    def get_shape(self, *parents_shapes):
        assert len(parents_shapes) == self.num_parents
        for shape in parents_shapes:
            assert shape == (), "all parents must have shape ()"
        return ()


normal_scale = AllScalarCondDist(2, "normal_scale", True)
normal_prec = AllScalarCondDist(2, "normal_prec", True)
bernoulli = AllScalarCondDist(1, "bernoulli", True)
bernoulli_logit = AllScalarCondDist(1, "bernoulli_logit", True)
binomial = AllScalarCondDist(2, "binomial", True)
uniform = AllScalarCondDist(2, "uniform", True)
beta = AllScalarCondDist(2, "beta", True)
exponential = AllScalarCondDist(1, "exponential", True)
beta_binomial = AllScalarCondDist(3, "beta_binomial", True)
add = AllScalarCondDist(2, "add", False)
sub = AllScalarCondDist(2, "sub", False)
mul = AllScalarCondDist(2, "mul", False)
div = AllScalarCondDist(2, "div", False)
pow = AllScalarCondDist(2, "pow", False)
abs = AllScalarCondDist(1, "abs", False)
exp = AllScalarCondDist(1, "exp", False)


class VecMatCondDist(CondDist):
    """
    Convenience class to create "vec mat" distributions that take as input a vector of
    length N, a matrix of size NxN and is a vector of length N
    """

    def __init__(self, name):
        super().__init__(name=name, random=True)

    def get_shape(self, vec_shape, mat_shape):
        assert len(vec_shape) == 1, "first parameter must be a vector"
        assert len(mat_shape) == 2, "second parameter must be a matrix"
        N = vec_shape[0]
        assert mat_shape == (
            N,
            N,
        ), "second parameter must be matrix with size matching first parameter"
        return (N,)


multi_normal_cov = VecMatCondDist("multi_normal_cov")


class MatMul(CondDist):
    """
    A class that does matrix multiplication, following the rules of `numpy.matmul`.
    Currently only 1d and 2d arrays are supported.
    """

    def __init__(self):
        super().__init__(name="matmul", random=False)

    def get_shape(self, a_shape, b_shape):
        # could someday generalize to handle more dimensions
        assert len(a_shape) >= 1, "args to @ must have at least 1 dim"
        assert len(b_shape) >= 1, "args to @ must have at least 1 dim"
        assert len(a_shape) <= 2, "args to @ must have at most 2 dims"
        assert len(b_shape) <= 2, "args to @ must have at most 2 dims"

        # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
        # The behavior depends on the arguments in the following way.
        # * If both arguments are 2-D they are multiplied like conventional matrices.
        # * If either argument is N-D, N > 2, it is treated as a stack of matrices
        #   residing in the last two indexes and broadcast accordingly.
        # * If the first argument is 1-D, it is promoted to a matrix by prepending a
        #   1 to its dimensions. After matrix multiplication the prepended 1 is removed.
        # * If the second argument is 1-D, it is promoted to a matrix by appending a
        #   1 to its dimensions. After matrix multiplication the appended 1 is removed.

        if len(a_shape) == 1 and len(b_shape) == 1:
            # inner product
            assert a_shape == b_shape
            return ()
        elif len(a_shape) == 1 and len(b_shape) == 2:
            # vector-matrix product
            assert a_shape[0] == b_shape[0]
            return (b_shape[1],)
        elif len(a_shape) == 2 and len(b_shape) == 1:
            # matrix-vector product
            assert a_shape[1] == b_shape[0]
            return (a_shape[0],)
        elif len(a_shape) == 2 and len(b_shape) == 2:
            # matrix-matrix product
            assert a_shape[1] == b_shape[0]
            return (a_shape[0], b_shape[1])
        else:
            raise Exception("bug: should be impossible")


matmul = MatMul()


class Categorical(CondDist):
    def __init__(self):
        super().__init__(name="categorical", random=True)

    def get_shape(self, weights_shape):
        # TODO: check shape
        return ()


categorical = Categorical()


class Dirichlet(CondDist):
    def __init__(self):
        super().__init__(name="dirichlet", random=True)

    def get_shape(self, weights_shape):
        # TODO: check shape
        return weights_shape


dirichlet = Dirichlet()


class Multinomial(CondDist):
    def __init__(self):
        super().__init__(name="multinomial", random=True)

    def get_shape(self, n_shape, p_shape):
        assert n_shape == ()
        assert len(p_shape) == 1
        return p_shape


multinomial = Multinomial()


class Sum(CondDist):
    def __init__(self, axis):
        # TODO: check axis is scalar
        # someday could support tuples of axes like numpy
        self.axis = axis
        super().__init__(name="sum", random=False)

    def get_shape(self, x_shape):
        if self.axis is None:
            return ()
        else:
            return x_shape[: self.axis] + x_shape[self.axis + 1 :]

    def __repr__(self):
        return f"Sum(axis={self.axis})"

    def __str__(self):
        return f"sum(axis={self.axis})"

    def __eq__(self, other):
        if isinstance(other, Sum):
            return self.axis == other.axis
        return False

    def __hash__(self):
        return hash(self.axis)


def slice_length(size, slice):
    return len(np.ones(size)[slice])


class Index(CondDist):
    """
    Index into a RV
    Note: slices must be FIXED when array is created
    """

    def __init__(self, *slices):
        self.slices = slices
        super().__init__(name="index", random=False)

    @property
    def advanced_at_start(self):
        # numpy has stupid rules: if advanced indices are separated by a slice
        # then all advanced indices go to start of output
        # otherwise go to location of first advanced index
        num_advanced = self.slices.count(None)
        if num_advanced <= 1:
            return False
        first_advanced = self.slices.index(None)
        slice_probe = self.slices[first_advanced : first_advanced + num_advanced]
        if all(s is None for s in slice_probe):
            return False  # in place
        else:
            return True

    def get_shape(self, var_shape, *indices_shapes):
        if len(self.slices) != len(var_shape):
            raise Exception("number of slots doesn't match number of dims of var")

        for idx_shape1 in indices_shapes:
            for idx_shape2 in indices_shapes:
                assert (
                    idx_shape1 == idx_shape2
                ), "all indices must have same shape (no broadcasting yet)"

        output_shape = ()
        idx_added = False
        for n, my_slice in enumerate(self.slices):
            if my_slice:
                output_shape += (slice_length(var_shape[n], my_slice),)
            else:
                idx_shape = indices_shapes[0]  # do here in case all sliced!
                if not idx_added:
                    if self.advanced_at_start:
                        output_shape = idx_shape + output_shape
                    else:
                        output_shape += idx_shape
                    idx_added = True
        return output_shape

    def __repr__(self):
        return "Index(slices=" + repr(self.slices) + ")"

    def __str__(self):
        def slice_str(s):
            match s:
                case None:
                    return "∅"
                case slice(start=None, stop=None, step=None):
                    return ":"
                case slice(start=a, stop=b, step=c):
                    if a is None:
                        a = ""
                    if b is None:
                        b = ""
                    if c is None:
                        c = ""
                    return f"{a}:{b}:{c}"
                case _:
                    raise Exception("not a slice")

        new_slices = tuple(slice_str(s) for s in self.slices)
        return "index" + util.comma_separated(new_slices)

    def __eq__(self, other):
        if isinstance(other, Index):
            return self.slices == other.slices
        return False

    def __hash__(self):
        return hash(str(self.slices))


def index(var, indices):
    if not isinstance(indices, tuple):
        indices = (indices,)  # TODO: this makes me nervous...

    # add extra full slices
    indices = indices + (slice(None, None, None),) * (var.ndim - len(indices))

    slices = []
    parents = []
    for index in indices:
        if isinstance(index, slice):
            slices.append(index)
        else:
            parents.append(index)
            slices.append(None)
    return Index(*slices)(var, *parents)


################################################################################
# Low-level LogProb dist: Wrap a random cond_dist to get a non-random one
################################################################################


class LogProb(CondDist):
    def __init__(self, base_cond_dist):
        assert base_cond_dist.random, "LogProb can only be called on random dists"
        self.base_cond_dist = base_cond_dist
        super().__init__(name="LogProb", random=False)

    def get_shape(self, val_shape, *base_parent_shapes):
        assert val_shape == self.base_cond_dist.get_shape(
            *base_parent_shapes
        ), "value shape unexpected"
        return ()

    def __eq__(self, other):
        if isinstance(other, LogProb):
            return self.base_cond_dist == other.base_cond_dist
        return False

    def __hash__(self):
        return self.base_cond_dist.__hash__()

    def __repr__(self):
        return "LogProb(" + repr(self.base_cond_dist) + ")"

    def __str__(self):
        return "LogProb(" + str(self.base_cond_dist) + ")"


################################################################################
# Low-level vmap operation: Turns one cond_dist into another
################################################################################


class Blank:
    def __repr__(self):
        return "∅"

    def __str__(self):
        return "∅"


blank = Blank()


def split_shape(shape, i):
    if i is None:
        new_shape = shape
        new_axis_size = None
    else:
        lo, mid, hi = (shape[:i], shape[i], shape[i + 1 :])
        new_shape = lo + hi
        new_axis_size = shape[i]
    return new_shape, new_axis_size


def get_sliced_shapes(shapes, in_axes, axis_size):
    axis_size = axis_size
    remaining_shapes = []
    for i, shape in zip(in_axes, shapes):
        new_shape, new_axis_size = split_shape(shape, i)
        remaining_shapes.append(new_shape)
        if axis_size is None:
            axis_size = new_axis_size
        elif new_axis_size is not None:
            assert axis_size == new_axis_size, "incoherent axis size"
    return remaining_shapes, axis_size


class VMapDist(CondDist):
    def __init__(self, base_cond_dist, in_axes, axis_size=None):
        assert isinstance(base_cond_dist, CondDist)
        # assert isinstance(in_axes, tuple), "in_axes must be tuple"
        if isinstance(in_axes, list):
            in_axes = tuple(in_axes)
        if axis_size is None:
            assert any(
                axis is not None for axis in in_axes
            ), "if axis_size=None, at least one axis must be mapped"
        else:
            assert isinstance(axis_size, int), "axis_size must be None or int"

        self.base_cond_dist = base_cond_dist
        self.in_axes = in_axes
        self.axis_size = axis_size
        super().__init__(name="VMapDist", random=base_cond_dist.random)

    def get_shape(self, *parents_shapes):
        remaining_shapes, axis_size = get_sliced_shapes(
            parents_shapes, self.in_axes, self.axis_size
        )
        dummy_shape = self.base_cond_dist.get_shape(*remaining_shapes)
        return (axis_size,) + dummy_shape

    def __repr__(self):
        return (
            f"VMapDist({repr(self.base_cond_dist)}, {repr(self.in_axes)}, "
            f"{repr(self.axis_size)})"
        )

    def __str__(self):
        # return "vmap(" + str(self.axis_size) + ', ' + str(self.in_axes) + ', '  + str(self.base_cond_dist) + ')'
        new_in_axes = jax.tree_util.tree_map(
            lambda x: blank if x is None else x,
            self.in_axes,
            is_leaf=util.is_leaf_with_none,
        )
        return (
            "vmap("
            + str(self.axis_size)
            + ", "
            + str(list(new_in_axes))
            + ", "
            + str(self.base_cond_dist)
            + ")"
        )

    def __eq__(self, other):
        if isinstance(other, VMapDist):
            return (
                self.base_cond_dist == other.base_cond_dist
                and self.in_axes == other.in_axes
                and self.axis_size == other.axis_size
            )
        return False

    def __hash__(self):
        return hash((self.base_cond_dist, self.in_axes, self.axis_size))


################################################################################
# For convenience, gather all cond_dists
################################################################################

all_cond_dists = [
    normal_scale,
    normal_prec,
    bernoulli,
    binomial,
    uniform,
    beta,
    exponential,
    beta_binomial,
    multi_normal_cov,
    categorical,
    dirichlet,
    multinomial,
    add,
    sub,
    mul,
    div,
    pow,
    abs,
    exp,
    matmul,
]

all_cond_dist_classes = [Sum, Index, VMapDist]


################################################################################
# for convenience, make scalar functions "auto vectorizing"
# so you can do x + 2 when x is a vector
################################################################################


def implicit_vectorized_scalar_cond_dist(cond_dist: AllScalarCondDist):
    def getdist(*parents):
        assert len(parents) == cond_dist.num_parents
        parents = tuple(makerv(p) for p in parents)
        vec_shape = None
        in_axes = []
        # make sure vectorizable
        for p in parents:
            if p.shape == ():
                # scalars are always OK
                in_axes.append(None)
            else:
                if vec_shape:
                    assert (
                        p.shape == vec_shape
                    ), "can only vectorize scalars + arrays of same shape"
                else:
                    vec_shape = p.shape
                in_axes.append(0)

        if vec_shape is None:
            return cond_dist(*parents)
        else:
            in_axes = tuple(in_axes)
            d = cond_dist
            vectorized_dims = len(vec_shape)
            for i in range(vectorized_dims):
                d = VMapDist(d, in_axes)
            return d(*parents)

    return getdist


vec_add = implicit_vectorized_scalar_cond_dist(add)
vec_sub = implicit_vectorized_scalar_cond_dist(sub)
vec_mul = implicit_vectorized_scalar_cond_dist(mul)
vec_div = implicit_vectorized_scalar_cond_dist(div)
vec_pow = implicit_vectorized_scalar_cond_dist(pow)


################################################################################
# RVs are very simple: Just remember parents and cond_dist and shape
################################################################################


def makerv(a):
    if isinstance(a, RV):
        return a
    else:
        cond_dist = Constant(a)
        # return RV(cond_dist) # previous—avoid calling cond dist
        return cond_dist()  # new, explicit call


class RV(dag.Node):
    _frozen = False
    __array_priority__ = 1000  # so x @ y works when x numpy.ndarray and y RV

    def __init__(self, cond_dist, *parents):
        parents_shapes = tuple(p.shape for p in parents)
        self._shape = cond_dist.get_shape(*parents_shapes)
        self.cond_dist = cond_dist
        super().__init__(*parents)
        self._frozen = True

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def __getitem__(self, idx):
        # TODO: special cases for Loops
        if isinstance(idx, Loop):
            return slice_existing_rv(self, [idx], Loop.loops)
        elif isinstance(idx, tuple) and any(isinstance(p, Loop) for p in idx):
            for p in idx:
                assert isinstance(p, Loop) or p == slice(
                    None
                ), "can only mix Loop with full slice"
            return slice_existing_rv(self, idx, Loop.loops)

        if self.ndim == 0:
            raise Exception("can't index scalar RV")
        elif isinstance(idx, tuple) and len(idx) > self.ndim:
            raise Exception("RV indexed with more dimensions than exist")
        return index(self, idx)

    def __repr__(self):
        ret = "RV(" + repr(self.cond_dist)
        if self.parents:
            ret += ", parents=[" + util.comma_separated(self.parents, repr, False) + "]"
        ret += ")"
        return ret

    def __str__(self):
        ret = str(self.cond_dist)
        if self.parents:
            ret += util.comma_separated(self.parents, str)
        return ret

    def __setattr__(self, key, value):
        if self._frozen:
            raise Exception("RVs are immutable after init.")
        else:
            self.__dict__[key] = value

    def __add__(self, b):
        return vec_add(self, b)

    __radd__ = __add__

    def __sub__(self, b):
        return vec_sub(self, b)

    def __rsub__(self, b):
        return vec_sub(b, self)

    def __mul__(self, b):
        return vec_mul(self, b)

    __rmul__ = __mul__

    def __truediv__(self, b):
        return vec_div(self, b)

    def __rtruediv__(self, b):
        return vec_div(b, self)

    def __pow__(self, b):
        return vec_pow(self, b)

    def __rpow__(self, a):
        return vec_pow(a, self)

    def __matmul__(self, a):
        return matmul(self, a)

    def __rmatmul__(self, a):
        return matmul(a, self)


class AbstractCondDist(CondDist):
    def __init__(self, shape):
        self.shape = shape
        super().__init__(name="abstract", random=False)

    def get_shape(self):
        return self.shape


# import here to avoid circular import problems
from .loops import Loop, make_sliced_rv, slice_existing_rv
