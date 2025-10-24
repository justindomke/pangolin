"""
This package defines a special subtype of RV that supports operator overloading
"""

from pangolin.ir import RV, Op, Constant, ScalarOp, VMap

# from pangolin.ir.scalar_ops import add, mul, sub, div, pow

# from pangolin.ir.op import SetAutoRV
from pangolin import ir

# from .indexing import index

from numpy.typing import ArrayLike

# from jax.typing import ArrayLike
import jax
import numpy as np
from typing import TypeVar, Type, Callable, cast, Sequence
from pangolin.util import comma_separated
import inspect
from typing import Generic
import os

RV_or_ArrayLike = RV | ArrayLike
# RV_or_ArrayLike = RV | jax.Array | np.ndarray | np.number | int | float
"""Represents either a `RV` or a value that can be cast to a NumPy array, such as a float or list of floats.
This will include JAX arrays (which is good) but also unfortunately accepts strings (which is bad)."""

####################################################################################################
# config
####################################################################################################

SCALAR_BROADCASTING = os.getenv("SCALAR_BROADCASTING", "off")

####################################################################################################
# The core InfixRV class. Like an RV except has infix operations
####################################################################################################

OpU = TypeVar("OpU", bound=Op)


class InfixRV(RV[OpU], Generic[OpU]):
    """An Infix RV is exactly like a standard `pangolin.ir.RV` except it supports infix operations.

    Examples
    --------
    >>> a = InfixRV(Constant(2))
    >>> b = InfixRV(Constant(3))
    >>> a + b
    InfixRV(Add(), InfixRV(Constant(2)), InfixRV(Constant(3)))
    >>> a**b
    InfixRV(Pow(), InfixRV(Constant(2)), InfixRV(Constant(3)))
    >>> -a
    InfixRV(Mul(), InfixRV(Constant(2)), InfixRV(Constant(-1)))

    See Also
    --------
    pangolin.ir.RV

    """

    __array_priority__ = 1000  # so x @ y works when x numpy.ndarray and y RV

    def __init__(self, op: OpU, *parents):
        super().__init__(op, *parents)

    def __neg__(self):
        return mul(self, -1)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __pow__(self, other):
        return pow(self, other)

    def __rpow__(self, other):
        return pow(other, self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __repr__(self):
        return "Infix" + super().__repr__()

    def __str__(self):
        return super().__str__()

    _IdxType = RV_or_ArrayLike | slice | type(Ellipsis)

    def __getitem__(self, idx: _IdxType | tuple[_IdxType, ...]):
        """
        @public
        You can index an `RV` with the `[]` operators, e.g. as `A[B,C]`.

        Note that indexing with this interface is different (and simpler) than NumPy or JAX:

        First, indexing is always fully-orthogonal. This is done to avoid the [utter insanity](https://numpy.org/doc/stable/user/basics.indexing.html) that is NumPy indexing with broadcasting, basic indexing, advanced indexing, and combinations of basic and advanced indexing. In this interface, if `A`, `B`, and `C` are RVs, then `A[B,C].shape == A.shape + B.shape`, and similarly if `B` or `C` are int / list of (list of) int / numpy array / slice.

        Second, all axes must be indexed. For example, if `A` is a RV with 3 axes, then `A[2]` will trigger an exception. The idea of this is to make code more legible and self-enforcing. Instead you must write `A[2, :, :]` or `A[2, ...]`.

        Examples
        --------
        >>> # Basic indexing
        >>> A = constant([9,8,7,6,5,4])
        >>> B = A[2]
        >>> B.op
        SimpleIndex()
        >>> B.parents[0] == A
        True
        >>> B.parents[1]
        InfixRV(Constant(2))

        >>> # indexing with a slice
        >>> B = A[1::2]
        >>> B.op
        SimpleIndex()
        >>> B.parents[0] == A
        True
        >>> B.parents[1]
        InfixRV(Constant([1,3,5]))

        >>> # indexing with a combination of constants and slices
        >>> A = constant([[3,4,5],[6,7,8]])
        >>> B = A[[1,0],::2]
        >>> B.op
        SimpleIndex()
        >>> B.parents[0] == A
        True
        >>> B.parents[1]
        InfixRV(Constant([1,0]))
        >>> B.parents[2]
        InfixRV(Constant([0,2]))
        """
        if not isinstance(idx, tuple):
            idx = (idx,)

        from .indexing import index

        return index(self, *idx)


####################################################################################################
# makerv
####################################################################################################


def constant(value: ArrayLike):
    """Create a constant RV

    Parameters
    ----------
    value: ArrayLike
        value for the constant. Should be a numpy (or JAX) array or something castable to that, e.g. int / float / list of list of ints/floats.

    Returns
    -------
    InfixRV
        RV with Constant Op

    Examples
    --------
    >>> constant(7)
    InfixRV(Constant(7))
    >>> constant([0,1,2])
    InfixRV(Constant([0,1,2]))

    """
    return InfixRV(Constant(value))


# def non_infix_rv(x):
#     if isinstance(x, RV):
#         if not isinstance(x, InfixRV):
#             return True
#     return False


def makerv(x) -> RV:
    """
    If the input is `RV`, then it just returns it. Otherwise, creates an InfixRV.

    Examples
    --------
    >>> x = makerv(1)
    >>> x
    InfixRV(Constant(1))
    >>> y = x + x
    >>> y
    InfixRV(Add(), InfixRV(Constant(1)), InfixRV(Constant(1)))
    >>> z = makerv(y)
    >>> z
    InfixRV(Add(), InfixRV(Constant(1)), InfixRV(Constant(1)))
    >>> y==z
    True
    """

    if isinstance(x, RV):
        return x
    else:
        return InfixRV(Constant(x))


# def make_infix_rv(x) -> InfixRV:
#     """
#     If the input is an `InfixRV`, then it just returns it. Otherwise, creates an InfixRV.
#     Fails is input is RV but not InfixRV
#     """
#     assert not non_infix_rv(x)

#     if isinstance(x,RV):
#         assert isinstance(x, InfixRV)
#         return x
#     else:
#         return InfixRV(Constant(x))

####################################################################################################
# Helpers to make scalar functions
####################################################################################################


def create_rv(op, *args) -> InfixRV:
    args = tuple(makerv(a) for a in args)
    # args = tuple(a if isinstance(a,RV) else constant(a) for a in args)
    op.get_shape(*[a.shape for a in args])  # checks shapes
    return InfixRV(op, *args)


def _scalar_op_doc(OpClass):
    op = OpClass
    expected_parents = op._expected_parents

    if op.random:
        __doc__ = f"""
        Creates a {str(OpClass.__name__)} distributed RV.
        """
    else:
        __doc__ = f"""
        Creates an RV by applying {str(OpClass.__name__)} to parents.
        """

    __doc__ += """
    
    Parameters
    ----------
    """

    for p in expected_parents:
        __doc__ += f"""
            {p}: RV_or_ArrayLike
                {expected_parents[p]}. Must be scalar.
    
            """

    __doc__ += f"""
    Returns
    -------
    z: InfixRV
        Random variable with `z.op` of type `pangolin.ir.{str(OpClass.__name__)}` and {len(expected_parents)} parent(s).
    """

    from .. import util

    # arguments 0.1, 0.2, 0.4, etc. work for *almost* all ops and round correctly
    args = [0.1 * 2**n for n in range(len(expected_parents))]
    args_str = [str(a) for a in args]
    par_args = [f"InfixRV(Constant({a}))" for a in args]

    __doc__ += f"""
    Examples
    --------
    >>> {util.camel_case_to_snake_case(str(OpClass.__name__))}{util.comma_separated(args_str, spaces=True)}
    InfixRV({str(OpClass.__name__)}(), {util.comma_separated(par_args, parens=False, spaces=True)})
    """

    if op._notes:
        __doc__ += f"""
        
    Notes
    -----
    
    """
        for note in op._notes:
            __doc__ += f"""
    {note}
    """

    # __doc__ += f"""

    # See Also
    # --------
    # `pangolin.ir.{str(OpClass.__name__)}`
    # """
    return __doc__


def vmap_scalars_simple(op: Op, *parent_shapes: ir._Shape) -> Op:
    """Given an all-scalar op (all inputs scalar, all outputs scalar), get a `VMap` op.
    This only accepts a very limited amount of broadcasting: All parents shapes must
    either be *scalar* or *exactly equal*.

    Parameters
    ----------
    op: Op
        the op to VMap
    shapes
        shapes for each parent, must all be *equal* or scalar

    Returns
    -------
    new_op: Op
        vmapped op (or possibly original op)

    Examples
    --------
    >>> vmap_scalars_simple(ir.Exp(), (3,))
    VMap(Exp(), (0,), 3)

    >>> vmap_scalars_simple(ir.Normal(), (3,), ())
    VMap(Normal(), (0, None), 3)

    >>> vmap_scalars_simple(ir.Normal(), (3,), ())
    VMap(Normal(), (0, None), 3)

    >>> vmap_scalars_simple(ir.StudentT(), (3,5), (), (3,5))
    VMap(VMap(StudentT(), (0, None, 0), 5), (0, None, 0), 3)
    """

    array_shape = None
    for shape in parent_shapes:
        if shape == ():
            continue

        if array_shape is None:
            array_shape = shape
        else:
            if shape != array_shape:
                raise ValueError(
                    f"Can't broadcast non-matching shapes {shape} and {array_shape}"
                )

    if array_shape is None:
        return op

    in_axes = [0 if shape == array_shape else None for shape in parent_shapes]

    new_op = op
    for size in reversed(array_shape):
        new_op = VMap(new_op, in_axes, size)

    assert new_op.get_shape(*parent_shapes) == array_shape, "Pangolin bug"

    return new_op


def vmap_scalars_numpy(op: Op, *parent_shapes: ir._Shape) -> Op:
    """Given an all-scalar op (all inputs scalar, all outputs scalar), get a `VMap` op.
    This implements most of numpy-style scalar broadcasting. The only limitation is that broadcasting of singleton dimensions
    against non-singleton dimensions is not supported.

    Parameters
    ----------
    op: Op
        the op to VMap
    shapes
        shapes for each parent, must all be *equal* or scalar

    Returns
    -------
    new_op: Op
        vmapped op (or possibly original op)

    Examples
    --------
    >>> vmap_scalars_numpy(ir.Exp(), (3,))
    VMap(Exp(), (0,), 3)

    >>> vmap_scalars_numpy(ir.Normal(), (3,), ())
    VMap(Normal(), (0, None), 3)

    >>> vmap_scalars_numpy(ir.Normal(), (), (2,3))
    VMap(VMap(Normal(), (None, 0), 3), (None, 0), 2)

    >>> vmap_scalars_numpy(ir.Normal(), (3,), (2,3))
    VMap(VMap(Normal(), (0, 0), 3), (None, 0), 2)
    """

    # will raise ValueError if not broadcastable
    array_shape = np.broadcast_shapes(*parent_shapes)

    if array_shape is None:
        return op

    new_op = op
    for n, size in enumerate(reversed(array_shape)):
        in_axes = []
        for parent_shape in parent_shapes:
            if len(parent_shape) < n + 1:
                my_axis = None
            else:
                my_axis = 0
            in_axes.append(my_axis)

        new_op = VMap(new_op, in_axes, size)

    assert new_op.get_shape(*parent_shapes) == array_shape, "Pangolin bug"

    return new_op

    # the obvious way to generalize this to handle singleten dimensions would be to change to
    #
    # (new_op, new_parents) = vmap_scalars_numpy(op: Op, *parents: ir.RV)
    #
    # where new_parents could be the same as old, or might include "squeeze" operations
    # (will also need to create ir.Squeeze)


def scalar_fun_factory(OpClass, /):
    import makefun

    op = OpClass()
    expected_parents = op._expected_parents  # type: ignore

    if SCALAR_BROADCASTING == "simple":

        def fun(*args, **kwargs):
            positional_args = args + tuple(kwargs[a] for a in kwargs)
            parent_shapes = [arg.shape for arg in positional_args]
            vmapped_op = vmap_scalars_simple(op, *parent_shapes)
            return create_rv(vmapped_op, *positional_args)

    elif SCALAR_BROADCASTING == "numpy":

        def fun(*args, **kwargs):
            positional_args = args + tuple(kwargs[a] for a in kwargs)
            parent_shapes = [arg.shape for arg in positional_args]
            vmapped_op = vmap_scalars_numpy(op, *parent_shapes)
            return create_rv(vmapped_op, *positional_args)

    elif SCALAR_BROADCASTING == "off":

        def fun(*args, **kwargs):
            return create_rv(op, *args, *[kwargs[a] for a in kwargs])

    else:
        raise Exception(f"Unknown scalar broadcasting model: {SCALAR_BROADCASTING}")

    func_sig = (
        f"{op.name}"
        + comma_separated([f"{a}:RV_or_ArrayLike" for a in expected_parents])
        + " -> InfixRV"
    )

    fun = makefun.create_function(func_sig, fun)
    fun.__doc__ = _scalar_op_doc(OpClass)
    return fun


def scalar_fun_factory1(OpClass) -> Callable[[RV_or_ArrayLike], InfixRV]:
    fun = scalar_fun_factory(OpClass)
    assert len(inspect.signature(fun).parameters) == 1
    return fun


def scalar_fun_factory2(
    OpClass,
) -> Callable[[RV_or_ArrayLike, RV_or_ArrayLike], InfixRV]:
    fun = scalar_fun_factory(OpClass)
    assert len(inspect.signature(fun).parameters) == 2
    return fun


def scalar_fun_factory3(
    OpClass,
) -> Callable[[RV_or_ArrayLike, RV_or_ArrayLike, RV_or_ArrayLike], InfixRV]:
    fun = scalar_fun_factory(OpClass)
    assert len(inspect.signature(fun).parameters) == 3
    return fun


####################################################################################################
# Arithmetic
####################################################################################################

add = scalar_fun_factory2(ir.Add)
sub = scalar_fun_factory2(ir.Sub)
mul = scalar_fun_factory2(ir.Mul)
div = scalar_fun_factory2(ir.Div)

####################################################################################################
# Trigonometry
####################################################################################################

arccos = scalar_fun_factory1(ir.Arccos)
arccosh = scalar_fun_factory1(ir.Arccosh)
arcsin = scalar_fun_factory1(ir.Arcsin)
arcsinh = scalar_fun_factory1(ir.Arcsinh)
arctan = scalar_fun_factory1(ir.Arctan)
arctanh = scalar_fun_factory1(ir.Arctanh)
cos = scalar_fun_factory1(ir.Cos)
cosh = scalar_fun_factory1(ir.Cosh)
sin = scalar_fun_factory1(ir.Sin)
sinh = scalar_fun_factory1(ir.Sinh)
tan = scalar_fun_factory1(ir.Tan)
tanh = scalar_fun_factory1(ir.Tanh)

####################################################################################################
# Other scalar function
####################################################################################################

pow = scalar_fun_factory2(ir.Pow)
abs = scalar_fun_factory1(ir.Abs)
exp = scalar_fun_factory1(ir.Exp)
inv_logit = scalar_fun_factory1(ir.InvLogit)
log = scalar_fun_factory1(ir.Log)
loggamma = scalar_fun_factory1(ir.Loggamma)
logit = scalar_fun_factory1(ir.Logit)
step = scalar_fun_factory1(ir.Step)

expit = scalar_fun_factory1(ir.InvLogit)
expit.__doc__ = "Equivalent to `inv_logit`"
sigmoid = scalar_fun_factory1(ir.InvLogit)
sigmoid.__doc__ = "Equivalent to `inv_logit`"


def sqrt(x):
    "sqrt(x) is an alias for pow(x,0.5)"
    return pow(x, 0.5)


####################################################################################################
# Multivariate funs
####################################################################################################


def matmul(a, b):
    """
    Matrix product of two arrays. The behavior follows that of
    [`numpy.matmul`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
    except that `a` and `b` must both be 1-D or 2-D arrays. In particular:
    * If `a` and `b` are both 1-D then this represents an inner-product.
    * If `a` is 1-D and `b` is 2-D then this represents vector/matrix multiplication
    * If `a` is 2-D and `b` is 1-D then this represents matrix/vector multiplication
    * If `a` and `b` are both 2-D then this represents matrix/matrix multiplication
    """
    return create_rv(ir.Matmul(), a, b)


def inv(a):
    """
    Take the inverse of a matrix. Input must be a 2-D square (invertible) array.
    """
    return create_rv(ir.Inv(), a)


def softmax(a):
    """
    Take [softmax](https://en.wikipedia.org/wiki/Softmax_function) function. (TODO: conform to
    syntax of [scipy.special.softmax](
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html))

    Parameters
    ----------
    a
        1-D vector
    """
    return create_rv(ir.Inv(), a)


def sum(x: RV, axis: int) -> InfixRV[ir.Sum]:
    """
    Take the sum of a random variable along a given axis

    Parameters
    ----------
    x
        an RV (or something that can be cast to a Constant RV)
    axis
        a non-negative integer (cannot be a random variable)
    """
    if not isinstance(axis, int):
        raise ValueError("axis argument for sum must be an integer")
    return create_rv(ir.Sum(axis), x)


####################################################################################################
# Scalar distributions
####################################################################################################

normal = scalar_fun_factory2(ir.Normal)
normal_prec = scalar_fun_factory2(ir.NormalPrec)
lognormal = scalar_fun_factory2(ir.Lognormal)
cauchy = scalar_fun_factory2(ir.Cauchy)
bernoulli = scalar_fun_factory1(ir.Bernoulli)
bernoulli_logit = scalar_fun_factory1(ir.BernoulliLogit)
binomial = scalar_fun_factory2(ir.Binomial)
uniform = scalar_fun_factory2(ir.Uniform)
beta = scalar_fun_factory2(ir.Beta)
beta_binomial = scalar_fun_factory3(ir.BetaBinomial)
exponential = scalar_fun_factory1(ir.Exponential)
gamma = scalar_fun_factory2(ir.Gamma)
poisson = scalar_fun_factory1(ir.Poisson)
student_t = scalar_fun_factory3(ir.StudentT)

####################################################################################################
# Multivariate dists
####################################################################################################


def multi_normal(mean, cov):
    """Create a multivariate normal distributed random variable. Call as `multi_normal(mean,cov)`"""
    return create_rv(ir.MultiNormal(), mean, cov)


def categorical(theta):
    """Create a [categorical](https://en.wikipedia.org/wiki/Categorical_distribution)-distributed
    where `theta` is a vector of non-negative reals that sums to one."""
    return create_rv(ir.Categorical(), theta)


def multinomial(n, p):
    """Create a [multinomial](https://en.wikipedia.org/wiki/Multinomial_distribution)-distributed
    random variable. Call as `multinomial(n,p)` where `n` is the number of repetitions and `p` is a
    vector of probabilities that sums to one."""
    return create_rv(ir.Multinomial(), n, p)


def dirichlet(alpha):
    """Create a [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution)-distributed
    random variable. Call as `dirichlet(alpha)` where `alpha` is a 1-D vector of positive reals.
    """
    return create_rv(ir.Dirichlet(), alpha)
