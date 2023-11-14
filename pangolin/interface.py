from . import dag, util

# import dag
import numpy as np
import jax

# import util


################################################################################
# Conditional distributions
################################################################################


class CondDist:
    def __init__(self, name):
        self.name = name  # only for printing, no functionality
        self._frozen = True  # freeze after init

    def __call__(self, *parents):
        """when you call a conditional distribution you get a RV"""
        parents = (makerv(p) for p in parents)
        return RV(self, *parents)

    def get_shape(self, *parents_shapes):
        raise NotImplementedError("Must construct for specific cases")

    def __repr__(self):
        return self.name

    @property
    def is_random(self):
        raise NotImplementedError()

    _frozen = False

    def __setattr__(self, key, value):
        """
        Redefine __setattr__ so that CondDists are immutable after init
        """
        if self._frozen:
            raise Exception(
                'CondDists are "frozen" — attributes cannot be changed '
                "after assigned"
            )
        else:
            self.__dict__[key] = value


class Constant(CondDist):
    def __init__(self, value):
        self.value = np.array(value)
        # super().__init__(self.value.shape)
        super().__init__("Constant")

    def get_shape(self):
        return self.value.shape

    @property
    def is_random(self):
        return False

    def __repr__(self):
        array_str = repr(self.value)  # get base string
        array_str = array_str[6:-1]  # cut off "array(" and ")"
        array_str = array_str.replace("\n", "")  # remove newlines
        array_str = array_str.replace(" ", "")  # remove specs
        return "Constant(" + array_str + ")"

    def __str__(self):
        # return str(self.value).replace("\n", "").replace("  ", " ")
        return (
            np.array2string(self.value, precision=3)
            .replace("\n", "")
            .replace("  ", " ")
        )


class AllScalarCondDist(CondDist):
    def __init__(self, num_parents, name, random):
        self.num_parents = num_parents
        self.random = random
        super().__init__(name)

    def get_shape(self, *parents_shapes):
        for shape in parents_shapes:
            assert shape == (), "all parents must have shape ()"
        return ()

    @property
    def is_random(self):
        return self.random


class VecMatCondDist(CondDist):
    """AA Represents a distribution that takes a vector of length N, a matrix of size NxN and is a vector of length N"""

    def __init__(self, name):
        super().__init__(name)

    def get_shape(self, vec_shape, mat_shape):
        assert len(vec_shape) == 1
        assert len(mat_shape) == 2
        N = vec_shape[0]
        assert mat_shape == (N, N)
        return (N,)

    @property
    def is_random(self):
        return True


# create stuff here (no functionality in this package)
normal_scale = AllScalarCondDist(2, "normal_scale", True)
normal_prec = AllScalarCondDist(2, "normal_prec", True)
bernoulli = AllScalarCondDist(1, "bernoulli", True)
binomial = AllScalarCondDist(2, "binomial", True)
uniform = AllScalarCondDist(2, "uniform", True)
beta = AllScalarCondDist(2, "beta", True)
exponential = AllScalarCondDist(2, "exponential", True)
beta_binomial = AllScalarCondDist(3, "beta_binomial", True)
multi_normal_cov = VecMatCondDist("multi_normal_cov")
add = AllScalarCondDist(2, "add", False)
sub = AllScalarCondDist(2, "sub", False)
mul = AllScalarCondDist(2, "mul", False)
div = AllScalarCondDist(2, "div", False)
pow = AllScalarCondDist(2, "pow", False)
abs = AllScalarCondDist(1, "abs", False)
exp = AllScalarCondDist(1, "exp", False)


def normal(loc, scale=None, prec=None):
    "1-d normals with multiple possible parameterizations"
    match (scale, prec):
        case (scale, None):
            return normal_scale(loc, scale)
        case (None, prec):
            return normal_prec(loc, prec)
        case _:
            raise Exception("must provide scale or prec but not both")


class MatMul(CondDist):
    def __init__(self):
        super().__init__("matmul")

    def get_shape(self, a_shape, b_shape):
        # TODO: generalize ?
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

    @property
    def is_random(self):
        return False


matmul = MatMul()


class Categorical(CondDist):
    def __init__(self):
        super().__init__("categorical")

    def get_shape(self, weights_shape):
        # TODO: check shape
        return ()

    @property
    def is_random(self):
        return True


categorical = Categorical()


class Dirichlet(CondDist):
    def __init__(self):
        super().__init__("dirichlet")

    def get_shape(self, weights_shape):
        # TODO: check shape
        return weights_shape

    @property
    def is_random(self):
        return True


dirichlet = Dirichlet()


class Multinomial(CondDist):
    def __init__(self):
        super().__init__("multinomial")

    def get_shape(self, n_shape, p_shape):
        assert n_shape == ()
        assert len(p_shape) == 1
        return p_shape

    @property
    def is_random(self):
        return True


multinomial = Multinomial()


def sum(x, axis=None):
    x = makerv(x)
    sum_op = Sum(axis)

    return sum_op(x)


class Sum(CondDist):
    def __init__(self, axis):
        self.axis = axis
        super().__init__("sum")

    def get_shape(self, x_shape):
        if self.axis is None:
            return ()
        else:
            return x_shape[: self.axis] + x_shape[self.axis + 1 :]

    @property
    def is_random(self):
        return False

    def __repr__(self):
        return f"Sum(axis={self.axis})"

    def __str__(self):
        return f"sum(axis={self.axis})"


def slice_length(size, slice):
    return len(np.ones(size)[slice])


class Index(CondDist):
    """
    Index into a RV
    Note: slices must be FIXED when array is created
    """

    def __init__(self, *slices):
        self.slices = slices
        super().__init__("index")

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

    @property
    def is_random(self):
        return False

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


class CondProb(CondDist):
    def __init__(self, cond_dist):
        assert isinstance(cond_dist, CondDist), "CondProb must be called on a CondDist"
        assert (
            cond_dist.is_random
        ), "CondProb must be called on a CondDist with cond_dist.is_random=True"
        self.base_cond_dist = cond_dist
        super().__init__("CondProb")

    def get_shape(self, value, *parent_values):
        return ()

    @property
    def is_random(self):
        return False

    def __repr__(self):
        return "CondProb(base_cond_dist=" + repr(self.base_cond_dist) + ")"

    def __str__(self):
        return "CondProb(" + str(self.base_cond_dist) + ")"


# See: https://mc-stan.org/docs/stan-users-guide/summing-out-the-responsibility-parameter.html


class Mixture(CondDist):
    def __init__(self, component_cond_dist, in_axes):
        self.component_cond_dist = component_cond_dist
        self.in_axes = in_axes
        super().__init__("Mixture")

    def get_shape(self, weights_shape, *parents_shapes):
        assert len(weights_shape) == 1, "weights must be 1D"
        axis_size = weights_shape[0]
        remaining_shapes, axis_size = get_sliced_shapes(
            parents_shapes, self.in_axes, axis_size
        )
        return self.component_cond_dist.get_shape(*remaining_shapes)

    def __repr__(self):
        return f"Mixture(component_cond_dist={repr(self.component_cond_dist)}, in_axes={repr(self.in_axes)})"

    def __str__(self):
        return f"Mixture({str(self.component_cond_dist)}, {str(self.in_axes)})"

    @property
    def is_random(self):
        return True


def mix(mixture_var, fun):
    """
    Take a discrete variable that defines a mixture and a function that maps that variable to a dists
    e.g.
    y = make_mixture([0.25, 0.75], lambda z: normal_scale(-0.5+z, 3.3))
    """
    if mixture_var.cond_dist == bernoulli:
        vals = np.arange(2)
    else:
        raise NotImplementedError("handle other dists")
    prob_fun = CondProb(mixture_var.cond_dist)
    prob_axes = (0,) + (None,) * len(mixture_var.parents)
    weights = vmap(prob_fun, prob_axes)(vals, *mixture_var.parents)

    # QUESTION:
    # will this work inside of a vmap or similar?
    # I think the main thing that won't work inside of vmap is something that examines
    # the parents of the input arguments
    vmapped_rv = vmap(fun, 0)(vals)
    assert isinstance(vmapped_rv, RV), "output must be an RV"
    vmapped_cond_dist = vmapped_rv.cond_dist
    component_cond_dist = vmapped_cond_dist.base_cond_dist
    assert component_cond_dist.is_random, "output must be random"
    parents = vmapped_rv.parents
    in_axes = vmapped_cond_dist.in_axes
    return Mixture(component_cond_dist, in_axes)(weights, *parents)


# class MixtureDist(CondDist):
#     def __init__(self, index_dist, component_dist, in_axes, num_index_params=1):
#         self.index_dist = index_dist
#         self.component_dist = component_dist
#         self.num_index_params = num_index_params
#         self.in_axes = in_axes
#         super().__init__("mixture")
#
#     def get_shape(self, *parents_shapes):
#         index_shapes = parents_shapes[: self.num_index_params]
#         component_shapes = parents_shapes[self.num_index_params :]
#         dummy_shapes, _ = get_sliced_shapes(
#             component_shapes, self.in_axes, in_axis=None
#         )
#
#         return self.component_dist.get_shape(*dummy_shapes)
#
#     def __repr__(self):
#         return (
#             "MixtureDist(index_dist="
#             + repr(self.index_dist)
#             + ", component_dist="
#             + repr(self.component_dist)
#             + ",in_axes="
#             + repr(self.in_axes)
#             + ")"
#         )
#
#     def __str__(self):
#         new_in_axes = jax.tree_util.tree_map(
#             lambda x: blank if x is None else x,
#             self.in_axes,
#             is_leaf=util.is_leaf_with_none,
#         )
#         return (
#             "mixture("
#             + str(new_in_axes)
#             + ","
#             + str(self.index_dist)
#             + ","
#             + str(self.component_dist)
#             + ")"
#         )


# Remember, transforms don't have any functionality!


# class Transform:
#     def get_shape(self, base_shape):
#         pass
#
#     def __call__(self, cond_dist):
#         print(f"{cond_dist=}")
#         assert isinstance(
#             cond_dist, CondDist
#         ), "transforms can only be called on CondDist objects"
#         return TransformedCondDist(cond_dist, self)
#
#
# class TransformedCondDist(CondDist):
#     "given base_dist(x|y) and transform(x) represent transform(x)|y"
#
#     def __init__(self, base_cond_dist, transform):
#         super().__init__("transformed")
#         self.base_cond_dist = base_cond_dist
#         self.transform = transform
#
#     def get_shape(self, *parent_shapes):
#         base_shape = self.base_cond_dist.get_shape(*parent_shapes)
#         new_shape = self.transform.get_shape(base_shape)
#         return new_shape
#
#     @property
#     def is_random(self):
#         return self.base_cond_dist.is_random
#
#     def __repr__(self):
#         return (
#             "transformed("
#             + repr(self.base_cond_dist)
#             + ","
#             + repr(self.transform)
#             + ")"
#         )
#
#     def __str__(self):
#         return str(self.transform) + " ∘ " + str(self.base_cond_dist)
#
#
# class ScalarTransform(Transform):
#     def __init__(self, name):
#         self.name = name
#
#     def get_shape(self, base_shape):
#         assert base_shape == ()
#         return ()
#
#     def __repr__(self):
#         return self.name
#
#     def __str__(self):
#         return self.name
#
#
# inverse_softplus = ScalarTransform("inverse_softplus")
# softplus = ScalarTransform("softplus")


################################################################################
# Low-level vmap operation: Turns one cond_dist into another
################################################################################


def split_shape(shape, i):
    if i is None:
        new_shape = shape
        new_axis_size = None
    else:
        lo, mid, hi = (shape[:i], shape[i], shape[i + 1 :])
        new_shape = lo + hi
        new_axis_size = shape[i]
    return new_shape, new_axis_size


class Blank:
    def __repr__(self):
        return "∅"

    def __str__(self):
        return "∅"


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


blank = Blank()


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
        super().__init__("VMapDist")

    def get_shape(self, *parents_shapes):
        remaining_shapes, axis_size = get_sliced_shapes(
            parents_shapes, self.in_axes, self.axis_size
        )
        dummy_shape = self.base_cond_dist.get_shape(*remaining_shapes)
        return (axis_size,) + dummy_shape

    @property
    def is_random(self):
        return self.base_cond_dist.is_random

    def __repr__(self):
        return "VMapDist(base_cond_dist=" + repr(self.base_cond_dist) + ")"

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
            + ", \n"
            + str(self.base_cond_dist)
            + ")"
        )


################################################################################
# Full-blown VMap
################################################################################


def vmap_dummy_args(in_axes, axis_size, *args):
    assert len(in_axes) == len(args)
    dummy_args = []
    for i, a in zip(in_axes, args):
        new_shape, new_axis_size = split_shape(a.shape, i)
        # print(f"{axis_size=} {new_axis_size=}")
        dummy_args.append(AbstractRV(new_shape))
        if axis_size is None:
            axis_size = new_axis_size
        elif new_axis_size is not None:
            assert axis_size == new_axis_size, "incoherent axis size"
    return dummy_args, axis_size


def vmap_generated_nodes(f, *dummy_args):
    # get all the nodes generated by a function
    # this is tricky because that function quite possibly includes captured closure variables
    # our strategy is to run the function twice on different copies of the arguments
    # anything that's included in one but not the other is new

    # don't strictly have to do this but whatever
    for p in dummy_args:
        assert isinstance(p, AbstractRV)

    # f is assumed to be flat!
    dummy_output1 = f(*dummy_args)
    assert isinstance(dummy_output1, list), "vmap_eval must be called on flat functions"

    # TODO: This will search the WHOLE upstream DAG, should try to make more efficient
    dummy_nodes1 = dag.upstream_nodes(dummy_output1)

    dummy_output2 = f(*dummy_args)
    assert isinstance(dummy_output2, list), "vmap_eval must be called on flat functions"
    assert len(dummy_output1) == len(dummy_output2)
    for d1, d2 in zip(dummy_output1, dummy_output2):
        assert d1.shape == d2.shape

    def excluded_node(node):
        return node in dummy_nodes1

    dummy_nodes2 = dag.upstream_nodes(dummy_output2, block_condition=excluded_node)
    return tuple(dummy_nodes2), dummy_output2


def vmap_eval(f, in_axes, axis_size, *args):
    # f must be FLAT!
    # that means that it takes a number of arguments, each of which is JUST A RV
    # and it returns a LIST of arguments, each of which is JUST A RV

    args = list(makerv(a) for a in args)

    dummy_args, axis_size = vmap_dummy_args(in_axes, axis_size, *args)

    dummy_nodes, dummy_outputs = vmap_generated_nodes(f, *dummy_args)

    # TODO: disabled for now — figure out why!
    # I think the problem is that if inputs are passed directly to outputs then
    # they don't count as generated
    # for dummy_output in dummy_outputs:
    #    print(f"{dummy_output=}")
    #    assert dummy_output in dummy_nodes, f"{dummy_output} not in dummy_nodes"

    for dummy_arg in dummy_args:
        assert dummy_arg not in dummy_nodes

    dummy_to_real = util.WriteOnceDict()
    dummy_mapped_axis = util.WriteOnceDict()
    for dummy_arg, i, arg in zip(dummy_args, in_axes, args):
        dummy_to_real[dummy_arg] = arg
        dummy_mapped_axis[dummy_arg] = i

    # trouble: what about non-dummy arguments!

    for dummy_node in dummy_nodes:
        dummy_parents = dummy_node.parents
        # replace dummies with real nodes
        parents = [dummy_to_real[p] if p in dummy_to_real else p for p in dummy_parents]
        # lookup mapped axes or just use None
        my_in_axes = [
            dummy_mapped_axis[p] if p in dummy_mapped_axis else None
            for p in dummy_parents
        ]

        # slight optimization: don't vmap constant nodes (could disable)
        if isinstance(dummy_node.cond_dist, Constant):
            assert my_in_axes == []
            real_node = dummy_node
            dummy_to_real[dummy_node] = real_node
            dummy_mapped_axis[dummy_node] = None
        else:
            cond_dist = VMapDist(
                dummy_node.cond_dist, in_axes=my_in_axes, axis_size=axis_size
            )
            real_node = RV(cond_dist, *parents)
            dummy_to_real[dummy_node] = real_node
            dummy_mapped_axis[dummy_node] = 0

    output = [dummy_to_real[dummy] for dummy in dummy_outputs]
    return output


class vmap:  # not a RV!
    def __init__(self, f, in_axes, axis_size=None):
        self.f = f
        self.in_axes = in_axes
        self.axis_size = axis_size

    def __call__(self, *args):
        # no greedy casting because this leads to ambiguity
        # if the user sends [(1,2),(3,4)] is that a list of two
        # arrays?
        args = jax.tree_util.tree_map(makerv, args)

        def get_dummy(i, x):
            if i is None:
                return AbstractRV(x.shape)
            else:
                lo, mid, hi = (x.shape[:i], x.shape[i], x.shape[i + 1 :])
                return AbstractRV(lo + hi)

        dummy_args = util.tree_map_recurse_at_leaf(
            get_dummy, self.in_axes, args, is_leaf=util.is_leaf_with_none
        )

        new_in_axes = util.tree_map_recurse_at_leaf(
            lambda i, x: i, self.in_axes, dummy_args, is_leaf=util.is_leaf_with_none
        )
        flat_in_axes, axes_treedef = jax.tree_util.tree_flatten(
            new_in_axes, is_leaf=util.is_leaf_with_none
        )
        flat_f, flatten_inputs, unflatten_output = util.flatten_fun(self.f, *dummy_args)
        flat_args = flatten_inputs(*args)
        flat_output = vmap_eval(flat_f, flat_in_axes, self.axis_size, *flat_args)
        output = unflatten_output(flat_output)
        return output


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

all_cond_dist_classes = [Sum, Index, CondProb, Mixture, VMapDist]


################################################################################
# RVs are very simple: Just remember parents and cond_dist and shape
################################################################################


def makerv(a):
    if isinstance(a, RV):
        return a
    else:
        cond_dist = Constant(a)
        return RV(cond_dist)


class RV(dag.Node):
    def __init__(self, cond_dist, *parents):
        super().__init__(*parents)
        parents_shapes = tuple(p.shape for p in parents)
        self._shape = cond_dist.get_shape(*parents_shapes)
        self.cond_dist = cond_dist

    def __add__(self, b):
        return add(self, b)

    __radd__ = __add__

    def __sub__(self, b):
        return sub(self, b)

    def __rsub__(self, b):
        return sub(b, self)

    def __mul__(self, b):
        return mul(self, b)

    __rmul__ = __mul__

    def __truediv__(self, b):
        return div(self, b)

    def __rtruediv__(self, b):
        return div(b, self)

    def __pow__(self, b):
        return pow(self, b)

    def __rpow__(self, a):
        return pow(a, self)

    def __matmul__(self, a):
        return matmul(self, a)

    def __getitem__(self, idx):
        return index(self, idx)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

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


class AbstractCondDist(CondDist):
    def __init__(self, shape):
        self.shape = shape
        super().__init__(name="abstract")

    def get_shape(self):
        return self.shape


class AbstractRV(RV):
    def __init__(self, shape):
        super().__init__(AbstractCondDist(shape))

    def __repr__(self):
        return "Abstract" + str(self.shape)

    def __str__(self):
        return str(self.shape)


class plate:
    def __init__(self, *args, N=None, in_axes=-1):
        self.in_axes = in_axes
        self.size = N
        self.args = tuple(makerv(a) for a in args)

    def __call__(self, f):
        assert hasattr(f, "__call__"), "must be callable"

        if self.in_axes == -1:
            if self.args == ():
                in_axes = None
            else:
                in_axes = 0
        else:
            in_axes = self.in_axes
        return vmap(f, in_axes, axis_size=self.size)(*self.args)


def print_upstream(*vars):
    vars = jax.tree_util.tree_leaves(vars)
    nodes = dag.upstream_nodes(vars)

    # get maximum # parents
    max_pars = 0
    max_shape = 0
    for node in nodes:
        max_pars = max(max_pars, len(node.parents))
        max_shape = max(max_shape, len(str(node.shape)))

    digits = 1 + int(np.log10(len(nodes) - 1))
    par_str_len = (digits + 1) * max_pars - 1

    id = 0
    node_to_id = {}
    for node in nodes:
        par_ids = [node_to_id[p] for p in node.parents]

        par_id_str = util.comma_separated(par_ids, util.num2str, False)
        par_id_str = par_id_str + " " * (par_str_len - len(par_id_str))

        shape_str = str(node.shape)
        shape_str += " " * (max_shape - len(shape_str))

        print(
            util.num2str(id)
            + ": "
            + shape_str
            + " ["
            + par_id_str
            + "] "
            + str(node.cond_dist)
        )
        node_to_id[node] = id
        id += 1


def label_fn(observed_vars=(), labels=None):
    if labels is None:
        labels = {}

    def fn(node):
        style = "filled" if node in observed_vars else None

        if node in labels:
            label = labels[node] + ": " + str(node.cond_dist)
        else:
            label = str(node.cond_dist)

        if node.cond_dist.is_random:
            shape = "oval"
        else:
            shape = "plaintext"

        return dict(style=style, label=label, shape=shape)

    return fn


def viz_generic(vars, label_fn):
    if isinstance(vars, dict):  # hack to handle replacements. Probably bad
        vars = tuple(vars.values())

    vars = jax.tree_util.tree_leaves(vars)
    nodes = dag.upstream_nodes(vars)

    import graphviz  # doctest: +NO_EXE

    graphviz.set_jupyter_format("png")

    dot = graphviz.Digraph()

    id = 0
    node_to_id = {}
    for node in nodes:
        dot.node(str(id), **label_fn(node))

        for p in node.parents:
            dot.edge(str(node_to_id[p]), str(id))

        node_to_id[node] = id
        id += 1
    return dot


def viz(vars, observed_vars=(), labels=None):
    fn = label_fn(observed_vars=observed_vars, labels=labels)
    return viz_generic(vars, fn)


viz_upstream = viz  # TODO: delete after changing calls

# TODO: Fix to remove new_infer
# def viz_samples(vars, precision=2):
#     vars = jax.tree_util.tree_leaves(vars)
#     nodes = dag.upstream_nodes(vars)
#
#     from . import new_infer as infer
#
#     samps = infer.sample(nodes, niter=1)
#
#     def fn(node):
#         i = nodes.index(node)
#         s = np.array(samps[i][0])
#         label = np.array_str(s, precision=precision)
#         label = np.array2string(
#             s,
#             precision=precision,
#             formatter={"float_kind": lambda x: f"%.{precision}f" % x},
#         )
#         return dict(label=label)
#
#     return viz_generic(nodes, fn)
#
#
# def viz_samples_live(vars, reps=25, wait=1, precision=2):
#     from IPython.display import clear_output, display, Image, SVG
#     import IPython.display
#     import time
#
#     for i in range(reps):
#         clear_output(wait=True)
#         graph = viz_samples(vars, precision=precision)
#         IPython.display.display_png(graph)
#         time.sleep(wait)
