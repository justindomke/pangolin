"""
This package defines a special subtype of RV that supports operator overloading
"""

from __future__ import annotations
from pangolin.ir import Op, Constant, ScalarOp, VMap
from pangolin import ir, util, dag
from numpy.typing import ArrayLike
import jax
import numpy as np
from typing import Type, Callable, Sequence, TYPE_CHECKING, get_type_hints, Union, Any
import inspect
import typing
import inspect
from dataclasses import dataclass, fields
from enum import Enum
from contextlib import contextmanager
import makefun
import types
import functools
import warnings
from jaxtyping import PyTree

RVLike: typing.TypeAlias = Union[ArrayLike, "InfixRV"]

# RV_or_ArrayLike = RV | jax.Array | np.ndarray | np.number | int | float

# TODO: Change return types to InfixRV?
# TODO: Shorten InfixRV name?

########################################################################################
# config
########################################################################################


class Broadcasting(Enum):
    """
    Broadcasting behavior for scalar functions.

    Used to change broadcasting behavior via `config`.

    .. code-block:: python

        >>> config.broadcasting = Broadcasting.OFF    # no broadcasting
        >>> config.broadcasting = Broadcasting.SIMPLE # simple broadcasting (default)
        >>> config.broadcasting = Broadcasting.NUMPY  # numpy-style broadcasting


    Broadcasting applies to "all-scalar" functions where all inputs and outputs are scalar.
    If broadcasting is set to OFF, all inputs must actually be scalar, and the output is
    scalar. If broadcasting is set to SIMPLE, then inputs can be non-scalar, but all
    non-scalar inputs must have *exactly* the same shape, which is the shape of the output
    If broadcasting is set to NUMPY, then inputs are broadcast against each other
    NumPy-style and the resulting shape is the shape of the output.

    Suppose ``f(a,b,c)`` is a scalar function. Then here is how broadcasting behaves:

    +-----------------------------------------+--------+-----------+-----------+
    |                                         | Broadcasting mode              |
    +                                         +--------+-----------+-----------+
    |                                         | OFF    | SIMPLE    | NUMPY     |
    +-------------+-------------+-------------+--------+-----------+-----------+
    | ``a.shape`` | ``b.shape`` | ``c.shape`` | ``f(a,b,c).shape``             |
    +=============+=============+=============+========+===========+===========+
    | ``()``      | ``()``      | ``()``      | ``()`` | ``()``    | ``()``    |
    +-------------+-------------+-------------+--------+-----------+-----------+
    | ``(5,)``    | ``()``      | ``()``      | n/a    | ``(5,)``  | ``(5,)``  |
    +-------------+-------------+-------------+--------+-----------+-----------+
    | ``(5,)``    | ``(5,)``    | ``()``      | n/a    | ``(5,)``  | ``(5,)``  |
    +-------------+-------------+-------------+--------+-----------+-----------+
    | ``(5,)``    | ``(5,)``    | ``(5,)``    | n/a    | ``(5,)``  | ``(5,)``  |
    +-------------+-------------+-------------+--------+-----------+-----------+
    | ``(5,9)``   | ``()``      | ``()``      | n/a    | ``(5,9)`` | ``(5,9)`` |
    +-------------+-------------+-------------+--------+-----------+-----------+
    | ``(5,9)``   | ``(5,9)``   | ``()``      | n/a    | ``(5,9)`` | ``(5,9)`` |
    +-------------+-------------+-------------+--------+-----------+-----------+
    | ``(5,9)``   | ``(5,9)``   | ``(5,9)``   | n/a    | ``(5,9)`` | ``(5,9)`` |
    +-------------+-------------+-------------+--------+-----------+-----------+
    | ``(9,)``    | ``(5,9)``   | ``()``      | n/a    | n/a       | ``(5,9)`` |
    +-------------+-------------+-------------+--------+-----------+-----------+
    | ``(5,)``    | ``(6,)``    | ``()``      | n/a    | n/a       | n/a       |
    +-------------+-------------+-------------+--------+-----------+-----------+

    This interface does not (currently) support automatic broadcasting for non-scalar
    functions.


    Attributes:
        OFF: No broadcasting at all. Scalar functions only accept scalar arguments.
        SIMPLE: Simple broadcasting only. Arguments can be any shape, but non-scalar
            must have exactly the same shape.
        NUMPY: NumPy-style broadcasting. The only limitation is that broadcasting
            of singleton dimensions against non-singleton dimensions is not currently
            supported.

    Examples
    --------

    No broadcasting

    >>> from pangolin import interface as pi
    >>> pi.config.broadcasting = pi.Broadcasting.OFF
    >>> pi.student_t(0,1,1).shape # doctest: +ELLIPSIS
    ()
    >>> pi.student_t(0,np.ones(5),1).shape # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: StudentT op got parent shapes ((), (5,), ()) not all scalar.

    Simple broadcasting

    >>> pi.config.broadcasting = pi.Broadcasting.SIMPLE
    >>> pi.student_t(0,1,1).shape
    ()
    >>> pi.student_t(0,np.ones(5),1).shape
    (5,)
    >>> pi.student_t(0,np.ones((5,3)),1).shape
    (5, 3)
    >>> pi.student_t(0,np.ones((5,3)),np.ones((5,3))).shape
    (5, 3)
    >>> pi.student_t(0,np.ones(3),np.ones((5,3))).shape
    Traceback (most recent call last):
        ...
    ValueError: Can't broadcast non-matching shapes (5, 3) and (3,)

    NumPy broadcasting

    >>> pi.config.broadcasting = pi.Broadcasting.NUMPY
    >>> pi.student_t(0,1,1).shape
    ()
    >>> pi.student_t(0,np.ones(5),1).shape
    (5,)
    >>> pi.student_t(0,np.ones((5,3)),1).shape
    (5, 3)
    >>> pi.student_t(0,np.ones((5,3)),np.ones((5,3))).shape
    (5, 3)
    >>> pi.student_t(0,np.ones(3),np.ones((5,3))).shape
    (5, 3)

    Restore to default

    >>> pi.config.broadcasting = pi.Broadcasting.SIMPLE

    You can also change this behavior temporarily in a code block by using the
    `override` context manager.

    >>> from pangolin import interface as pi
    >>> with pi.override(broadcasting="off"):
    ...     x = pi.normal(np.zeros(3), np.ones((5,3)))
    Traceback (most recent call last):
    ...
    ValueError: Normal op got parent shapes ((3,), (5, 3)) not all scalar.
    >>> with pi.override(broadcasting="numpy"):
    ...     x = pi.normal(np.zeros(3), np.ones((5,3)))
    ...     x.shape
    (5, 3)

    """

    OFF = "off"
    SIMPLE = "simple"
    NUMPY = "numpy"


@dataclass
class Config:
    """Global configuration for Pangolin interface.

    See Also
    --------
    override : Context manager for temporary config changes.
    config: instance to use to actually make changes
    """

    broadcasting: Broadcasting = Broadcasting.SIMPLE


config: Config = Config()
config.__doc__ = "Singleton instance of `Config` class"


@contextmanager
def override(**kwargs):
    """
    Temporarily override config values.

    A context manager that sets `config` values for the duration of
    the block, then restores the original values on exit.

    Args:
        kwargs: Attribute names and their temporary values.
            See `config` for available options.

    Raises:
        AttributeError: If a key doesn't match a config attribute.
        ValueError: If a value is invalid for that config option.

    Example:
        >>> from pangolin.interface import override, Broadcasting
        >>> config.broadcasting
        <Broadcasting.SIMPLE: 'simple'>
        >>> with override(broadcasting="off"):
        ...     config.broadcasting
        <Broadcasting.OFF: 'off'>
        >>> config.broadcasting
        <Broadcasting.SIMPLE: 'simple'>
    """
    type_hints = get_type_hints(Config)

    originals = {}
    for key, value in kwargs.items():
        originals[key] = getattr(config, key)
        expected_type = type_hints[key]
        if isinstance(value, expected_type):
            setattr(config, key, value)
        else:
            cast_value = expected_type(value)
            setattr(config, key, cast_value)

    try:
        yield
    finally:
        for key, value in originals.items():
            setattr(config, key, value)


########################################################################################
# The core InfixRV class. Like an RV except has infix operations
########################################################################################


class VectorIndexProxy:
    def __init__(self, var: InfixRV):
        self.var = var

    def __getitem__(self, args):
        from .indexing import vector_index

        if isinstance(args, tuple):
            return vector_index(self.var, *args)
        else:
            return vector_index(self.var, args)


# OpU = TypeVar("OpU", bound=Op)


class InfixRV[O: Op](ir.RV[O]):
    """An Infix RV is exactly like a standard `pangolin.ir.RV` except it supports infix
    operations.

    This is a generic type, so you may write ``InfixRV[OpClass]`` as a type hint.

    Args:
        op: The Op defining this class.
        *parents


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

    def __init__(self, op: O, *parents: InfixRV):
        self.s = VectorIndexProxy(self)
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

    @property
    def T(self):
        return transpose(self)

    def __repr__(self):
        return "Infix" + super().__repr__()

    def __str__(self):
        return super().__str__()

    _IdxType = RVLike | slice | types.EllipsisType

    def __getitem__(self, idx: _IdxType | tuple[_IdxType, ...]):
        """
        You can index an `RV` with the ``[]`` operator, e.g. as ``A[B,C]``.

        Note that indexing with this interface is different (and simpler) than NumPy or JAX:

        First, indexing is by default fully-orthogonal. This is done to avoid the `utter insanity <https://numpy.org/doc/stable/user/basics.indexing.html>`_ that is NumPy indexing with broadcasting, basic indexing, advanced indexing, and combinations of basic and advanced indexing. In this interface, if ``A``, ``B``, and ``C`` are RVs, then ``A[B,C].shape == A.shape + B.shape``, and similarly if ``B`` or ``C`` are int / list of (list of) int / numpy array / slice.

        Second, all axes must be indexed. For example, if ``A`` is a RV with 3 axes, then ``A[2]`` will trigger an exception. The idea of this is to make code more legible and self-enforcing. Instead you must write ``A[2, :, :]`` or ``A[2, ...]``.

        Examples
        --------
        >>> # Basic indexing
        >>> A = constant([9,8,7,6,5,4])
        >>> B = A[2]
        >>> B.op
        Index()
        >>> B.parents[0] == A
        True
        >>> B.parents[1]
        InfixRV(Constant(2))

        >>> # indexing with a slice
        >>> B = A[1::2]
        >>> B.op
        Index()
        >>> B.parents[0] == A
        True
        >>> B.parents[1]
        InfixRV(Constant([1,3,5]))

        >>> # indexing with a combination of constants and slices
        >>> A = constant([[3,4,5],[6,7,8]])
        >>> B = A[[1,0],::2]
        >>> B.op
        Index()
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

    @property
    def parent_ops(self):
        """
        Just a shortcut for ``tuple(p.op for p in self.parents)``. Intended mostly for testing.
        """
        return tuple(p.op for p in self.parents)


########################################################################################
# constants and makerv
########################################################################################


def constant(value: ArrayLike) -> InfixRV[Constant]:
    """Create a constant RV

    Parameters
    ----------
    value
        value for the constant. Should be a numpy (or JAX) array or something castable
        to that, e.g. int / float / list of list of ints/floats.

    Returns
    -------
    out
        RV with Constant Op

    Examples
    --------
    >>> constant(7)
    InfixRV(Constant(7))
    >>> constant([0,1,2])
    InfixRV(Constant([0,1,2]))
    """
    return InfixRV(Constant(value))


def makerv(x: RVLike) -> InfixRV:
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

    if isinstance(x, ir.RV) and not isinstance(x, InfixRV):
        raise ValueError("makerv called with base ir RV (only handles InfixRV)")

    if isinstance(x, InfixRV):
        return x
    else:
        return InfixRV(Constant(x))


def get_shape(arg: RVLike):
    """If `arg` has a shape, attribute return it. Otherwise return the shape it would have if cast to an array. (No array is actually created.)

        Parameters
    ----------
    arg : array_like
        Input data, in any form that can be converted to an array. This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists, and ndarrays.

    Returns
    -------
    shape : tuple of ints
        Shape of the array that would result from casting `arg` to an array.
        Each element of the tuple gives the size of the corresponding dimension.

    See Also
    --------
    numpy.shape : Return the shape of an array

    Examples
    --------
    >>> get_shape([[1, 2, 3], [4, 5, 6]])
    (2, 3)

    >>> get_shape(np.array([[1, 2, 3], [4, 5, 6]]))
    (2, 3)

    >>> get_shape(constant([[1, 2, 3], [4, 5, 6]]))
    (2, 3)

    >>> get_shape(42)  # Scalar
    ()

    >>> get_shape([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    (2, 2, 2)

    >>> get_shape([[], [], []])
    (3, 0)

    """

    if isinstance(arg, InfixRV):
        return arg.shape
    else:
        return np.shape(arg)


########################################################################################
# Multivariate funs
########################################################################################


def matmul(a: RVLike, b: RVLike) -> InfixRV[ir.Matmul]:
    """
    Matrix product of two arrays. The behavior follows that of
    `numpy.matmul <https://numpy.org/doc/stable/reference/generated/numpy.matmul.html>`_
    except that ``a`` and ``b`` must both be 1-D or 2-D arrays. In particular:

    * If ``a`` and ``b`` are both 1-D then this represents an inner-product.
    * If ``a`` is 1-D and ``b`` is 2-D then this represents vector/matrix multiplication
    * If ``a`` is 2-D and ``b`` is 1-D then this represents matrix/vector multiplication
    * If ``a`` and ``b`` are both 2-D then this represents matrix/matrix multiplication

    Parameters
    ----------
    a
        first argument (1d or 2d array)
    b
        second argument (1d or 2d array, matching shape of ``a``)
    """
    return create_rv(ir.Matmul(), a, b)


def transpose(a: RVLike) -> InfixRV[ir.Transpose]:
    """
    Tranpose a matrix. Input must be a 2-D array.
    """
    return create_rv(ir.Transpose(), a)


def diag(a: RVLike) -> InfixRV[ir.Diag]:
    """
    Get the diagonal of a matrix. Input must be a 2-D square array. Does not construct diagonal matrices. (Use `diag_matrix`)
    """
    return create_rv(ir.Diag(), a)


def diag_matrix(a: RVLike) -> InfixRV[ir.DiagMatrix]:
    """
    Get the diagonal of a matrix. Input must be a 1-D array. Does not extract diagonals. (Use `diag`)
    """
    return create_rv(ir.DiagMatrix(), a)


def inv(a: RVLike) -> InfixRV[ir.Inv]:
    """
    Take the inverse of a matrix. Input must be a 2-D square (invertible) array.
    """
    return create_rv(ir.Inv(), a)


def cholesky(a: RVLike) -> InfixRV[ir.Cholesky]:
    """
    Take the inverse of a matrix. Input must be a 2-D square (invertible) array.
    """
    return create_rv(ir.Cholesky(), a)


def softmax(a: RVLike) -> InfixRV[ir.Softmax]:
    """
    Take `softmax <https://en.wikipedia.org/wiki/Softmax_function>`_ function.
    (TODO: conform to
    syntax of `scipy.special.softmax
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html>`_

    Parameters
    ----------
    a
        1-D vector
    """
    return create_rv(ir.Softmax(), a)


def sum(x: InfixRV, axis: int) -> InfixRV[ir.Sum]:
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


########################################################################################
# Multivariate dists
########################################################################################


def multi_normal(mean: RVLike, cov: RVLike) -> InfixRV[ir.MultiNormal]:
    """
    Create a multivariate normal distributed random variable.
    Call as ``multi_normal(mean, cov)``

    Parameters
    ----------
    mean
        mean (1D array)
    cov
        covariance (2D positive-definite array)
    """
    return create_rv(ir.MultiNormal(), mean, cov)


def categorical(theta: RVLike) -> InfixRV[ir.Categorical]:
    """
    Create a `categorical <https://en.wikipedia.org/wiki/Categorical_distribution>`_
    distributed `RV` where ``theta`` is a vector of non-negative reals that sums to one.

    Parameters
    ----------
    theta
        positive event probabilities (should sum to one)
    """
    return create_rv(ir.Categorical(), theta)


def multinomial(n: RVLike, p: RVLike) -> InfixRV[ir.Multinomial]:
    """
    Create a `multinomial <https://en.wikipedia.org/wiki/Multinomial_distribution>`_
    distributed random variable. Call as ``multinomial(n,p)`` where ``n`` is the number of repetitions and ``p`` is a vector of probabilities that sums to one.

    Parameters
    ----------
    n
        number of repetitions (scalar)
    p
        vector of probabilities (should sum to one)

    """
    return create_rv(ir.Multinomial(), n, p)


def dirichlet(alpha: RVLike) -> InfixRV[ir.Dirichlet]:
    """
    Create a
    `Dirichlet <https://en.wikipedia.org/wiki/Dirichlet_distribution>`__
    distributed random variable.
    Call as ``dirichlet(alpha)`` where ``alpha`` is a 1-D vector of positive reals.

    Parameters
    ----------
    alpha
        concentration (vector of positive numbers)
    """
    return create_rv(ir.Dirichlet(), alpha)


def wishart(nu: RVLike, S: RVLike) -> InfixRV[ir.Wishart]:
    """
    Create a
    `Wishart <https://en.wikipedia.org/wiki/Wishar_distribution>`__
    distributed random variable.

    Args:
        nu: degress of freedom (scalar)
        S: scale matrix (symmetric posisitive definite)

    """
    # TODO: Support regular / inverse / cholesky wishart
    return create_rv(ir.Wishart(), nu, S)


########################################################################################
# vmapping
########################################################################################

FlatCallable = Callable[..., list[InfixRV]]  # don't know how to enforce that inputs are RV

Shape = ir.Shape


# class FlatCallable(Protocol):
#     def __call__(self, *args: RV) -> list[InfixRV]: ...


class AbstractOp(Op):
    """
    Create an abstract Op. An `AbstractOp` doesn't actually do anything and expects no parents. It just has a fixed shape.

    Parameters
    ----------
    shape
        the shape for the output
    """

    _random = False

    def __init__(self, shape: Shape = ()):
        self.shape = shape
        super().__init__()

    def _get_shape(self, *parents_shapes):
        return self.shape

    def __eq__(self, other):
        """
        Equality for abstract ops is very restricted
        """
        return id(self) == id(other)

    def __hash__(self):
        return object.__hash__(self)

    def __repr__(self):
        if self.shape == ():
            return "AbstractOp()"
        else:
            return f"AbstractOp({self.shape})"

    def __str__(self):
        if self.shape == ():
            return "abstract_op"
        else:
            return f"abstract_op({self.shape})"


def vmap_subgraph(
    dummy_roots: Sequence[InfixRV],
    dummy_nodes: Sequence[InfixRV],
    dummy_outputs: Sequence[InfixRV],
    roots: Sequence[InfixRV],
    roots_axes: Sequence[int | None],
    axis_size: int | None,
) -> list[InfixRV]:
    """
    Takes a graph of "dummy" RVs that represent some non-vmapped computation, then creates a parallel graph of "real" RVs that represent a vmapped computation


    Parameters
    ----------
    dummy_roots
        Root notes for non-vmapped graph.
    dummy_nodes
        Rest of nodes for non-vmapped graph
    dummy_outputs
        Output nodes for non-vmapped graph (must be in dummy_node)
    roots
        Root notes for the desired vmapped graph
    roots_axes
        the axes along which the roots should be vectorized
    axis_size
        the axis size for all mapped nodes (optional unless no args vmapped)

    Returns
    -------
    real_outputs
        vmapped nodes corresponding to `dummy_outputs`, but with everything vectorized

    Examples
    --------
    >>> a_dummy = InfixRV(AbstractOp())
    >>> b_dummy = InfixRV(AbstractOp())
    >>> c_dummy = InfixRV(ir.Add(), a_dummy, b_dummy)
    >>> a = InfixRV(Constant([0, 1, 2]))
    >>> b = InfixRV(Constant([4, 5, 6]))
    >>> [c] = vmap_subgraph([a_dummy, b_dummy], [c_dummy], [c_dummy], [a, b], [0, 0], 3)
    >>> print(str(c_dummy))
    add(abstract_op, abstract_op)
    >>> print(str(c))
    vmap(add, [0, 0], 3)([0 1 2], [4 5 6])
    >>> print(repr(c_dummy))
    InfixRV(Add(), InfixRV(AbstractOp()), InfixRV(AbstractOp()))
    >>> print(repr(c))
    InfixRV(VMap(Add(), [0, 0], 3), InfixRV(Constant([0,1,2])), InfixRV(Constant([4,5,6])))

    >>> d_dummy = InfixRV(ir.Mul(), a_dummy, c_dummy)
    >>> print_upstream(d_dummy)
    shape | statement
    ----- | ---------
    ()    | a = abstract_op
    ()    | b = abstract_op
    ()    | c = add(a,b)
    ()    | d = mul(a,c)
    >>> [d] = vmap_subgraph([a_dummy, b_dummy], [c_dummy, d_dummy], [d_dummy], [a, b], [0, 0], 3)
    >>> print_upstream(d)
    shape | statement
    ----- | ---------
    (3,)  | a = [0 1 2]
    (3,)  | b = [4 5 6]
    (3,)  | c = vmap(add, [0, 0], 3)(a,b)
    (3,)  | d = vmap(mul, [0, 0], 3)(a,c)
    """
    # TODO: Should we allow axis_size=None here?

    if any(a in dummy_roots for a in dummy_nodes):
        raise ValueError("dummy_roots cannot be included in dummy_nodes")

    if not all(a in dummy_nodes for a in dummy_outputs):
        raise ValueError("dummy_outputs must all be included in dummy_nodes")

    dummy_to_real = util.WriteOnceDefaultDict(lambda p: p)
    dummy_mapped_axis = util.WriteOnceDefaultDict(lambda p: None)
    for dummy_arg, i, arg in zip(dummy_roots, roots_axes, roots, strict=True):
        dummy_to_real[dummy_arg] = arg
        dummy_mapped_axis[dummy_arg] = i

    for dummy_node in dummy_nodes:
        dummy_parents = dummy_node.parents
        parents = tuple(dummy_to_real[p] for p in dummy_parents)
        my_in_axes = tuple(dummy_mapped_axis[p] for p in dummy_parents)

        no_mapped_axes = all(axis is None for axis in my_in_axes)
        # if no mapped axes AND non-random AND not in output, no need to map
        if no_mapped_axes and not dummy_node.op.random and dummy_node not in dummy_outputs:
            new_op = dummy_node.op
            new_axis = None
        else:
            new_op = VMap(dummy_node.op, in_axes=my_in_axes, axis_size=axis_size)
            new_axis = 0

        dummy_to_real[dummy_node] = create_rv(new_op, *parents)
        dummy_mapped_axis[dummy_node] = new_axis

    # if any(dummy_mapped_axis[a] is None for a in dummy_outputs):
    #    raise ValueError("Output not vmapped")

    real_nodes = [dummy_to_real[dummy_node] for dummy_node in dummy_outputs]
    return real_nodes


def vmap_dummy_args(
    args: Sequence[InfixRV], in_axes: Sequence[int | None], axis_size: int | None
) -> tuple[tuple[InfixRV[AbstractOp], ...], int]:
    """
    Given a "full" arguments, get a list of dummy/sliced arguments.

    Parameters
    ----------
    args
        Sequence of RVs for which sliced "dummies" are required.
    in_axes
        What axis to map each argument over. Should have same length as `args`.
    axis_size
        Anticipated axis size (or None if it should be inferred)

    Returns
    -------
    dummy_args
        tuple of abstract RVs with sliced shapes
    axis_size
        inferred axis_size (or arg if provided)

    Examples
    --------
    >>> A = InfixRV(Constant([[0,1,2],[4,5,6]]))
    >>> B = InfixRV(Constant([7,8,9]))
    >>> dummy_args, axis_size = vmap_dummy_args([A, B], [1, 0], None)
    >>> dummy_args
    (InfixRV(AbstractOp((2,))), InfixRV(AbstractOp()))
    """

    if not util.all_unique(args):
        raise ValueError("vmap_dummy_args requires all unique arguments")

    dummy_args = []
    for i, a in zip(in_axes, args, strict=True):
        new_shape, new_axis_size = ir.split_shape(a.shape, i)

        # once upon a time we did this—but don't remember the point of it
        # if isinstance(a.op, VMap) and i == 0:
        #     new_op = a.op.base_op  # why do we care?
        # else:
        #     new_op = AbstractOp(new_shape, a.op.random)

        new_op = AbstractOp(new_shape)
        my_dummy = create_rv(new_op)  # no parents!

        dummy_args.append(my_dummy)
        if axis_size is None:
            axis_size = new_axis_size
        elif new_axis_size is not None:
            assert axis_size == new_axis_size, "incoherent axis size"
    if axis_size is None:
        raise ValueError("axis_size could not be inferred")

    return tuple(dummy_args), axis_size


def generated_nodes(fun: FlatCallable, *args: InfixRV) -> tuple[list[InfixRV], list[InfixRV]]:
    """
    Given a "flat" function and some number of RV arguments, get all the nodes that the function
    creates. This *includes* nodes that do not depend on the inputs.

    Parameters
    ----------
    fun
        A function that takes some number of `RV` arguments and returns a list of `RV`s
    *args
        arguments to call the function on.

    Returns
    -------
    all_vars
        All `RV`s that are generated by this function and downstream of `args`
    out
        The outputs of the original function (typically overlaps with `all_vars`)

    Examples
    --------
    >>> def fun(x,y):
    ...     a = InfixRV(ir.Exp(), x)
    ...     b = InfixRV(ir.Add(), a, y)
    ...     return [b]
    >>> x = InfixRV(ir.Constant(0))
    >>> y = InfixRV(ir.Constant(1))
    >>> all_vars, out = generated_nodes(fun, x, y)
    >>> len(all_vars)
    2
    >>> all_vars[0]
    InfixRV(Exp(), InfixRV(Constant(0)))
    >>> all_vars[1]
    InfixRV(Add(), InfixRV(Exp(), InfixRV(Constant(0))), InfixRV(Constant(1)))
    >>> out
    [InfixRV(Add(), InfixRV(Exp(), InfixRV(Constant(0))), InfixRV(Constant(1)))]
    """
    for a in args:
        assert isinstance(a, InfixRV), "arguments must be InfixRV"

    # all generated nodes must have higher n
    n_before_call = InfixRV._n

    def is_abstract(rv: InfixRV) -> bool:
        return rv._n >= n_before_call

    def not_abstract(var: InfixRV):
        return not is_abstract(var)

    abstract_out: list[InfixRV[AbstractOp]] = fun(*args)

    if not isinstance(abstract_out, list):
        raise ValueError("generated_nodes must take a function that returns a list")
    if any(a in args for a in abstract_out):
        raise ValueError("fun passed to generated_nodes cannot return input values")
    for a in abstract_out:
        if a in args:
            raise ValueError("fun passed to generated_nodes cannot return inputs.")
        if not isinstance(a, InfixRV):
            raise ValueError(f"fun passed to generated_nodes returned non-RV output (got {type(a)}")

    all_abstract_vars = dag.upstream_nodes(abstract_out, node_block=not_abstract)

    all_abstract_vars = sorted(all_abstract_vars, key=lambda node: node._n)

    # convert abstract nodes to concrete
    abstract_to_concrete: dict[InfixRV, InfixRV] = {}
    for abstract_var in all_abstract_vars:
        if abstract_var in args:
            where_var = args.index(abstract_var)
            concrete_var = args[where_var]
        else:
            new_parents = tuple(abstract_to_concrete[p] if is_abstract(p) else p for p in abstract_var.parents)
            concrete_var = create_rv(abstract_var.op, *new_parents)
        abstract_to_concrete[abstract_var] = concrete_var

    all_vars = [abstract_to_concrete[v] for v in all_abstract_vars if v not in args]
    out = [abstract_to_concrete[v] if v in abstract_to_concrete else v for v in abstract_out]

    return all_vars, out


def vmap_eval_flat(
    f: FlatCallable,
    in_axes: Sequence[int | None],
    axis_size: int | None,
    *args: RVLike,
) -> list[InfixRV]:
    """
    This function (but not vmap itself) works on "flat" function f, meaning that each
    argument of the function is just a RV. And the function must return
    a list of arguments which again are each just a RV.

    Parameters
    ----------
    f: FlatCallable
        The function to be vmapped. Must be "flat"
    in_axes: Sequence[int | None]
        axes to vmap for each argument
    axis_size: int | None
        length of vmap.
    args: RV_or_ArrayLike
        arguments. unlike most functions

    Returns
    -------
    vmapped_outputs
        The result of the vmap

    Examples
    --------
    >>> def f(a,b):
    ...     return [a+b]
    >>> A = constant([0,1])
    >>> B = constant([2,3])
    >>> [C] = vmap_eval_flat(f, (0,0), 2, A, B)
    >>> print(repr(C))
    InfixRV(VMap(Add(), [0, 0], 2), InfixRV(Constant([0,1])), InfixRV(Constant([2,3])))
    """

    # make sure inputs are RVs
    rv_args = tuple(makerv(a) for a in args)
    dummy_args, axis_size = vmap_dummy_args(rv_args, in_axes, axis_size)
    dummy_nodes, dummy_outputs = generated_nodes(f, *dummy_args)

    return vmap_subgraph(dummy_args, dummy_nodes, dummy_outputs, rv_args, in_axes, axis_size)


def get_dummy_args(in_axes, args) -> tuple[list[int], list[InfixRV]]:
    """Converts PyTree args and axes to flat args and axes

    Parameters
    ----------
    in_axes
        PyTree of input axes
    args
        PyTree of inputs

    Returns
    -------
    flat_in_axes
        Flat input axes
    flat_args
        flat args

    Examples
    --------
    >>> get_dummy_args(0, makerv([0,1,2]))
    InfixRV(AbstractOp())
    >>> get_dummy_args([0,0], [makerv([0,1,2]), makerv([3,4,5])])
    [InfixRV(AbstractOp()), InfixRV(AbstractOp())]
    >>> get_dummy_args(0, [makerv([0,1,2]), makerv([3,4,5])])
    [InfixRV(AbstractOp()), InfixRV(AbstractOp())]
    >>> get_dummy_args(0, {1:makerv([0,1,2])})
    {1: InfixRV(AbstractOp())}
    >>> get_dummy_args(0, {'cat':makerv([0,1,2])})
    {'cat': InfixRV(AbstractOp())}
    >>> get_dummy_args({'cat':0}, {'cat':makerv([0,1,2])})
    {'cat': InfixRV(AbstractOp())}
    >>> get_dummy_args({'cat':0, 'dog':None}, {'cat':makerv([0,1,2]), 'dog':makerv(3)})
    {'cat': InfixRV(AbstractOp()), 'dog': InfixRV(AbstractOp())}

    >>> A = constant([0,1,2])
    >>> B = constant([3,4,5])
    >>> x = {"dog": A, "cat": B}
    >>> in_axes = {"dog": 0, "cat": 0}
    >>> get_dummy_args(in_axes, x)
    {'cat': InfixRV(AbstractOp()), 'dog': InfixRV(AbstractOp())}
    """

    def get_dummy(i, x):
        if i is None:
            new_shape = x.shape
        else:
            lo, mid, hi = (x.shape[:i], x.shape[i], x.shape[i + 1 :])
            new_shape = lo + hi

        # In old code tried to preserve x.op when isinstance(x.op, VMap)
        op = AbstractOp(new_shape)
        return create_rv(op)

    dummy_args = util.tree_map_recurse_at_leaf(get_dummy, in_axes, args, is_leaf=util._is_leaf_with_none)

    return dummy_args


# RVCallable = Callable[[PyTree[InfixRV]], PyTree[InfixRV]]


# def vmap(f: Callable, in_axes: Any = 0, axis_size: int | None = None) -> Callable:
def vmap[*Args](
    f: Callable[[*Args], PyTree[InfixRV]], in_axes: Any = 0, axis_size: int | None = None
) -> Callable[[*Args], PyTree[InfixRV]]:
    """
    Vectorizing map. Create a function which maps ``f`` over argument axes.

    This function matches exactly the interface of `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`_, although it doesn't provide some of the optional arguments ``jax.vmap`` does.

    Parameters
    ----------
    f
        The function to vmap. Should take a pytree of `RV` as inputs and return a pytree of `RV` as outputs.
    in_axes
        An int, None, or pytree with roots that are int or None. Specifies which axis of each RV should be mapped (if int) or that no axis shuld be mapped (if None). Can be a pytree matching the structure of all arguments to ``f``. Or, can be a pytree that is a prefix to the pytree representing all arguments. By default, in_axes is zero, meaning all RVs are mapped over the first axis.
    axis_size
        An integer indicating the size of the axis to be mapped. This is optional unless all leaves of ``in_axes`` are ``None``.

    Returns
    -------
    vec_f
        batched/vectorized version of ``f`` with arguments matching those of ``f`` with extra axes at positions indicated by ``in_axes`` and a return value that corresponds to that of ``f`` but with an extra axis in the first position.

    Examples
    --------
    Here's the simplest possible example.

    >>> def fun(a):
    ...     return exp(a)
    >>> A = constant([0,1,2])
    >>> vmap(fun)(A)
    InfixRV(VMap(Exp(), [0], 3), InfixRV(Constant([0,1,2])))

    Multiple inputs are OK.

    >>> def fun(a,b):
    ...     return a*b
    >>> A = constant([1,2,3])
    >>> B = constant([4,5,6])
    >>> vmap(fun)(A, B)
    InfixRV(VMap(Mul(), [0, 0], 3), InfixRV(Constant([1,2,3])), InfixRV(Constant([4,5,6])))

    Unmapped inputs are OK.

    >>> def fun(a,b):
    ...     return a*b
    >>> A = constant([1,2,3])
    >>> vmap(fun, [0, None])(A, constant(7))
    InfixRV(VMap(Mul(), [0, None], 3), InfixRV(Constant([1,2,3])), InfixRV(Constant(7)))

    Multiple outputs are OK.

    >>> def fun(a):
    ...     return [exp(a), log(a)]
    >>> [out1, out2] = vmap(fun)(A)
    >>> out1
    InfixRV(VMap(Exp(), [0], 3), InfixRV(Constant([1,2,3])))
    >>> out2
    InfixRV(VMap(Log(), [0], 3), InfixRV(Constant([1,2,3])))

    Pytree inputs and pytree in_axes are OK

    >>> def fun(x):
    ...     return x['cat']*x['dog']
    >>> x = {'cat': A, 'dog': constant(3)}
    >>> in_axes = {'cat': 0, 'dog': None}
    >>> vmap(fun, in_axes)(x)
    InfixRV(VMap(Mul(), [0, None], 3), InfixRV(Constant([1,2,3])), InfixRV(Constant(3)))

    Pytree outputs are OK

    >>> def fun(a, b):
    ...     return {"add": a+b, "mul": a*b}
    >>> vmap(fun)(A, B)
    {'add': InfixRV(VMap(Add(), [0, 0], 3), InfixRV(Constant([1,2,3])), InfixRV(Constant([4,5,6]))), 'mul': InfixRV(VMap(Mul(), [0, 0], 3), InfixRV(Constant([1,2,3])), InfixRV(Constant([4,5,6])))}

    Pytree in_axis prefixes are OK

    >>> def fun(x):
    ...     [a, (b,c)] = x
    ...     return (a*b)+c
    >>> x = [A, (constant(7), constant(8))]
    >>> in_axes1 = [0, (None, None)] # axes for each leaf
    >>> in_axes2 = [0, None]         # single None for (b,c) tuple!
    >>> out1 = vmap(fun, in_axes1)(x)
    >>> out2 = vmap(fun, in_axes2)(x)
    >>> out1.op == out2.op
    True
    >>> print_upstream(out1)
    shape | statement
    ----- | ---------
    (3,)  | a = [1 2 3]
    ()    | b = 7
    (3,)  | c = vmap(mul, [0, None], 3)(a,b)
    ()    | d = 8
    (3,)  | e = vmap(add, [0, None], 3)(c,d)
    >>> print_upstream(out2)
    shape | statement
    ----- | ---------
    (3,)  | a = [1 2 3]
    ()    | b = 7
    (3,)  | c = vmap(mul, [0, None], 3)(a,b)
    ()    | d = 8
    (3,)  | e = vmap(add, [0, None], 3)(c,d)

    See Also
    --------
    pangolin.ir.VMap
    """

    # TODO: support negative in_axes

    def call(*args):
        if len(args) == 1:
            # handles vmap(f, 0)(x) instead vmap(f,(0,))(x)
            my_in_axes = (in_axes,)
        elif isinstance(in_axes, list):
            # handles vmap(f, [0,1])(x,y) instead of vmap(f,(0,1))(x,y)
            my_in_axes = tuple(in_axes)
        else:
            my_in_axes = in_axes

        # no greedy casting because this leads to ambiguity
        # if the user sends [(1,2),(3,4)] is that a list of two
        # arrays?
        args = jax.tree_util.tree_map(makerv, args)

        dummy_args = get_dummy_args(my_in_axes, args)

        flat_in_axes, flat_args = util.dual_flatten(my_in_axes, args)

        flat_f, flatten_inputs, unflatten_output = util.flatten_fun(f, *dummy_args, is_leaf=util._is_leaf_with_none)
        flat_args = flatten_inputs(*args)
        flat_output = vmap_eval_flat(flat_f, flat_in_axes, axis_size, *flat_args)
        output = unflatten_output(flat_output)
        return output

    return call


def convert_args(rv_type: Type[InfixRV], *args: InfixRV):
    """
    Given some set of (interdependent) RVs, get a new set where all are converted to a new type
    but all inter-RV parent links are preserved.

    Parameters
    ----------
    rv_type
        Some subclass of `RV`
    args:RV
        arguments, all of type RV

    Returns
    -------
    new_args:tuple[rv_type]
        converted args
    """
    abstract_args = {}
    for a in args:
        new_parents: list[InfixRV] = [abstract_args[p] if p in abstract_args else p for p in a.parents]
        abstract_a = rv_type(a.op, *new_parents)
        abstract_args[a] = abstract_a
    return tuple(abstract_args[a] for a in args)


def vmap_flat(f: FlatCallable, in_axes: tuple[int | None, ...], axis_size: int | None):
    """
    vmap a flat function (one that takes some number of RV arguments and returns a list of RV
    arguments)
    """

    def vec_f(*args):
        args = list(makerv(a) for a in args)
        dummy_args, my_axis_size = vmap_dummy_args(args, in_axes, axis_size)
        dummy_nodes, dummy_outputs = generated_nodes(f, *dummy_args)
        return vmap_subgraph(dummy_args, dummy_nodes, dummy_outputs, args, in_axes, my_axis_size)

    return vec_f


########################################################################################
# Auto-generate interface functions for all scalar Ops
########################################################################################


def create_rv[O: Op](op: O, *args) -> InfixRV[O]:
    args = tuple(makerv(a) for a in args)
    # args = tuple(a if isinstance(a,RV) else constant(a) for a in args)
    op.get_shape(*[a.shape for a in args])  # checks shapes
    return InfixRV(op, *args)


def _scalar_op_doc(OpClass):
    op = OpClass
    # expected_parents = op._expected_parents
    op_info = ir._OpInfo(OpClass)
    expected_parents = op_info.expected_parents

    if op.random:
        __doc__ = f"Creates a {str(OpClass.__name__)} distributed RV."
    else:
        __doc__ = f"Creates an RV by applying {str(OpClass.__name__)} to parents."

    __doc__ += """
    
    All arguments must either be scalar or mutually broadcastable according to
    ``config.broadcasting``

    Args:
    """

    for p in expected_parents:
        __doc__ += f"""
        {p}: {expected_parents[p]}
    """

    __doc__ += f"""
    Returns:
        Random variable with ``z.op`` of type `pangolin.ir.{str(OpClass.__name__)}` or `pangolin.ir.VMap` if broadcasting is triggered and {len(expected_parents)} parent(s).
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


def broadcast_shapes_simple(*shapes: ir.Shape) -> None | ir.Shape:
    new_shape = None
    for shape in shapes:
        if shape == ():
            continue

        if new_shape is None:
            new_shape = shape
        else:
            if shape != new_shape:
                raise ValueError(f"Can't broadcast non-matching shapes {shape} and {new_shape}")
    return new_shape


def broadcast_shapes_numpy(*shapes: ir.Shape) -> None | ir.Shape:
    new_shape = np.broadcast_shapes(*shapes)
    return new_shape


def vmap_scalars_simple[O: Op](op: O, *parent_shapes: ir.Shape) -> VMap | O:
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
    VMap(Exp(), [0], 3)

    >>> vmap_scalars_simple(ir.Normal(), (3,), ())
    VMap(Normal(), [0, None], 3)

    >>> vmap_scalars_simple(ir.Normal(), (3,), ())
    VMap(Normal(), [0, None], 3)

    >>> vmap_scalars_simple(ir.StudentT(), (3,5), (), (3,5))
    VMap(VMap(StudentT(), [0, None, 0], 5), [0, None, 0], 3)
    """

    # TODO: Always return VMap

    array_shape = broadcast_shapes_simple(*parent_shapes)

    if array_shape is None:
        return op

    in_axes = tuple(0 if shape == array_shape else None for shape in parent_shapes)

    new_op = op
    for size in reversed(array_shape):
        new_op = VMap(new_op, in_axes, size)

    assert new_op.get_shape(*parent_shapes) == array_shape, "Pangolin bug"

    return new_op


def vmap_scalars_numpy[O: Op](op: O, *parent_shapes: ir.Shape) -> O | ir.VMap:
    """Given an all-scalar op (all inputs scalar, all outputs scalar), get a `VMap` op.
    This implements most of numpy-style scalar broadcasting. The only limitation is that
    broadcasting of singleton dimensions against non-singleton dimensions is not
    supported.

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
    VMap(Exp(), [0], 3)

    >>> vmap_scalars_numpy(ir.Normal(), (3,), ())
    VMap(Normal(), [0, None], 3)

    >>> vmap_scalars_numpy(ir.Normal(), (), (2,3))
    VMap(VMap(Normal(), [None, 0], 3), [None, 0], 2)

    >>> vmap_scalars_numpy(ir.Normal(), (3,), (2,3))
    VMap(VMap(Normal(), [0, 0], 3), [None, 0], 2)
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

        new_op = VMap(new_op, tuple(in_axes), size)

    assert new_op.get_shape(*parent_shapes) == array_shape, "Pangolin bug"

    return new_op

    # the obvious way to generalize this to handle singleten dimensions would be to change to
    #
    # (new_op, new_parents) = vmap_scalars_numpy(op: Op, *parents: ir.RV)
    #
    # where new_parents could be the same as old, or might include "squeeze" operations
    # (will also need to create ir.Squeeze)


########################################################################################
# Broadcasting
########################################################################################


def validate_scalar_args(*args, **kwargs):
    for a in args:
        if a.shape != ():
            raise ValueError(f"Non-scalar argument: {a}")
    for name in kwargs:
        if kwargs[name].shape != ():
            raise ValueError(f"Non-scalar argument: {name}={kwargs[name]}")


def bind_args(fun, *args, **kwargs):
    """
    Flattens positional and keyword arguments into positional only. The function must not have keyword-only arguments.
    """
    sig = inspect.signature(fun)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.args


def broadcast(scalar_fun):
    """
    Args:
        scalar_fun: Function accepting and returning all-scalar arguments. Can have keyword arguments, but cannot have keyword-only arguments.

    Returns:
        "broadcast" version of ``scalar_fun`` that accepts and returns array-shaped random variables according to the current broadcasting rules.


    Examples
    --------
    Take a simple function with scalar input and output

    >>> fun = lambda a: InfixRV(ir.Exponential(), a)

    If you just have a scalar input, this function works fine in all three broadcasting modes

    >>> with override(broadcasting='off'):
    ...     broadcast(fun)(3)
    InfixRV(Exponential(), InfixRV(Constant(3)))
    >>> with override(broadcasting='simple'):
    ...      broadcast(fun)(3)
    InfixRV(Exponential(), InfixRV(Constant(3)))
    >>> with override(broadcasting='numpy'):
    ...      broadcast(fun)(3)
    InfixRV(Exponential(), InfixRV(Constant(3)))

    If you have a vector input, this only works with simple or numpy broadcasting

    >>> with override(broadcasting='off'):
    ...     broadcast(fun)([3,4])
    Traceback (most recent call last):
    ...
    ValueError: Non-scalar argument: [3 4]
    >>> with override(broadcasting='simple'):
    ...     broadcast(fun)([3,4])
    InfixRV(VMap(Exponential(), [0], 2), InfixRV(Constant([3,4])))
    >>> with override(broadcasting='numpy'):
    ...     broadcast(fun)([3,4])
    InfixRV(VMap(Exponential(), [0], 2), InfixRV(Constant([3,4])))

    Now take this function with one vector and one matrix argument.

    >>> fun = lambda a, b: InfixRV(ir.Normal(), a, b)
    >>> mean = [1,2,3]
    >>> scale = [[1,2,3],[4,5,6]]

    This will only work with numpy-style broadcasting

    >>> with override(broadcasting='off'):
    ...     broadcast(fun)(mean, scale)
    Traceback (most recent call last):
    ...
    ValueError: Non-scalar argument: [1 2 3]
    >>> with override(broadcasting='simple'):
    ...     broadcast(fun)(mean, scale)
    Traceback (most recent call last):
    ...
    ValueError: Can't broadcast non-matching shapes (2, 3) and (3,)
    >>> with override(broadcasting='numpy'):
    ...     z = broadcast(fun)(mean, scale)
    >>> print(z)
    vmap(vmap(normal, [0, 0], 3), [None, 0], 2)([1 2 3], [[1 2 3] [4 5 6]])
    """

    @functools.wraps(scalar_fun)
    def handler(*args, **kwargs):
        args = bind_args(scalar_fun, *args, **kwargs)

        args = tuple(makerv(a) for a in args)
        arg_shapes = [a.shape for a in args]

        if config.broadcasting == Broadcasting.OFF:
            validate_scalar_args(*args)
            return scalar_fun(*args)
        elif config.broadcasting == Broadcasting.SIMPLE:
            array_shape = broadcast_shapes_simple(*arg_shapes)

            if array_shape is None:
                return scalar_fun(*args)

            in_axes = tuple(0 if shape == array_shape else None for shape in arg_shapes)
            new_fun = lambda args: scalar_fun(*args)  # always takes a single tuple
            for size in reversed(array_shape):
                new_fun = vmap(new_fun, in_axes=in_axes, axis_size=size)
            return new_fun(args)
        elif config.broadcasting == Broadcasting.NUMPY:
            array_shape = np.broadcast_shapes(*arg_shapes)

            if array_shape is None:
                return scalar_fun(*args)

            new_fun = lambda args: scalar_fun(*args)  # always takes a single tuple
            for n, size in enumerate(reversed(array_shape)):
                in_axes = []
                for arg_shape in arg_shapes:
                    if len(arg_shape) < n + 1:
                        my_axis = None
                    else:
                        my_axis = 0
                    in_axes.append(my_axis)

                new_fun = vmap(new_fun, tuple(in_axes), size)
            return new_fun(args)
        else:
            raise Exception(f"Unknown scalar broadcasting model: {config.broadcasting}")

    return handler


########################################################################################
# Generate interface for scalar functions
########################################################################################

type ScalarInterfaceFun[ScalarO: ScalarOp] = Callable[..., InfixRV[ScalarO | VMap]]


def get_base_op_from_scalar_fun[ScalarO: ScalarOp](fun: ScalarInterfaceFun[ScalarO]) -> Type[ScalarO]:
    hints = get_type_hints(fun)
    return_type = hints.get("return")

    if not return_type:
        raise ValueError(f"{fun.__name__} needs a return annotation")

    all_args = typing.get_args(return_type)
    assert len(all_args) == 1

    union_arg = all_args[0]
    OpClass = typing.get_args(union_arg)[0]
    return OpClass


def _scalar_op_signature(fun: ScalarInterfaceFun):
    sig = inspect.signature(fun)
    params = list(sig.parameters.values())

    OpClass = get_base_op_from_scalar_fun(fun)
    expected_parents = OpClass._expected_parents

    if isinstance(expected_parents, int):
        assert len(params) == expected_parents
        return sig

    assert len(params) == len(expected_parents), f"parent mismatch for {fun}"

    new_params = []
    for param, parent_name in zip(params, expected_parents, strict=True):
        new_p = param.replace(name=parent_name)
        new_params.append(new_p)

    new_sig = sig.replace(parameters=new_params)
    return new_sig


# def _scalar_handler(op):
#     def handler(*args, **kwargs):
#         if config.broadcasting == Broadcasting.OFF:
#             return create_rv(op, *args, *[kwargs[a] for a in kwargs])
#         elif config.broadcasting == Broadcasting.SIMPLE:
#             positional_args = args + tuple(kwargs[a] for a in kwargs)
#             parent_shapes = [get_shape(arg) for arg in positional_args]
#             vmapped_op = vmap_scalars_simple(op, *parent_shapes)
#             return create_rv(vmapped_op, *positional_args)
#         elif config.broadcasting == Broadcasting.NUMPY:
#             positional_args = args + tuple(kwargs[a] for a in kwargs)
#             parent_shapes = [get_shape(arg) for arg in positional_args]
#             vmapped_op = vmap_scalars_numpy(op, *parent_shapes)
#             return create_rv(vmapped_op, *positional_args)
#         else:
#             raise Exception(f"Unknown scalar broadcasting model: {config.broadcasting}")

#     return handler


def scalar_op(fun: ScalarInterfaceFun):
    OpClass = get_base_op_from_scalar_fun(fun)
    op = OpClass()

    if fun.__name__ != util.camel_case_to_snake_case(OpClass.__name__):
        raise Exception(f"{fun.__name__} does not match to {OpClass.__name__}")

    # assert fun.__name__ == util.camel_case_to_snake_case(OpClass.__name__)

    def handler(*args, **kwargs):
        positional_args = tuple(makerv(a) for a in args) + tuple(makerv(kwargs[a]) for a in kwargs)
        return InfixRV(op, *positional_args)

    wrapper = makefun.create_function(
        _scalar_op_signature(fun),
        # _scalar_handler(op),
        handler,
        func_name=fun.__name__,
        qualname=fun.__qualname__,
        module_name=fun.__module__,
        doc=_scalar_op_doc(OpClass),
    )

    return wrapper


@scalar_op
def add(a: RVLike, b: RVLike) -> InfixRV[ir.Add | ir.VMap[ir.Add]]: ...
@scalar_op
def sub(a: RVLike, b: RVLike) -> InfixRV[ir.Sub | ir.VMap[ir.Sub]]: ...
@scalar_op
def mul(a: RVLike, b: RVLike) -> InfixRV[ir.Mul | ir.VMap[ir.Mul]]: ...
@scalar_op
def div(a: RVLike, b: RVLike) -> InfixRV[ir.Div | ir.VMap[ir.Div]]: ...
@scalar_op
def pow(a: RVLike, b: RVLike) -> InfixRV[ir.Pow | ir.VMap[ir.Pow]]: ...
@scalar_op
def abs(a: RVLike) -> InfixRV[ir.Abs | ir.VMap[ir.Abs]]: ...
@scalar_op
def arccos(a: RVLike) -> InfixRV[ir.Arccos | ir.VMap[ir.Arccos]]: ...
@scalar_op
def arccosh(a: RVLike) -> InfixRV[ir.Arccosh | ir.VMap[ir.Arccosh]]: ...
@scalar_op
def arcsin(a: RVLike) -> InfixRV[ir.Arcsin | ir.VMap[ir.Arcsin]]: ...
@scalar_op
def arcsinh(a: RVLike) -> InfixRV[ir.Arcsinh | ir.VMap[ir.Arcsinh]]: ...
@scalar_op
def arctan(a: RVLike) -> InfixRV[ir.Arctan | ir.VMap[ir.Arctan]]: ...
@scalar_op
def arctanh(a: RVLike) -> InfixRV[ir.Arctanh | ir.VMap[ir.Arctanh]]: ...
@scalar_op
def cos(a: RVLike) -> InfixRV[ir.Cos | ir.VMap[ir.Cos]]: ...
@scalar_op
def cosh(a: RVLike) -> InfixRV[ir.Cosh | ir.VMap[ir.Cosh]]: ...
@scalar_op
def exp(a: RVLike) -> InfixRV[ir.Exp | ir.VMap[ir.Exp]]: ...
@scalar_op
def inv_logit(a: RVLike) -> InfixRV[ir.InvLogit | ir.VMap[ir.InvLogit]]: ...
@scalar_op
def log(a: RVLike) -> InfixRV[ir.Log | ir.VMap[ir.Log]]: ...
@scalar_op
def loggamma(a: RVLike) -> InfixRV[ir.Loggamma | ir.VMap[ir.Loggamma]]: ...
@scalar_op
def logit(a: RVLike) -> InfixRV[ir.Logit | ir.VMap[ir.Logit]]: ...
@scalar_op
def sin(a: RVLike) -> InfixRV[ir.Sin | ir.VMap[ir.Sin]]: ...
@scalar_op
def sinh(a: RVLike) -> InfixRV[ir.Sinh | ir.VMap[ir.Sinh]]: ...
@scalar_op
def step(a: RVLike) -> InfixRV[ir.Step | ir.VMap[ir.Step]]: ...
@scalar_op
def tan(a: RVLike) -> InfixRV[ir.Tan | ir.VMap[ir.Tan]]: ...
@scalar_op
def tanh(a: RVLike) -> InfixRV[ir.Tanh | ir.VMap[ir.Tanh]]: ...
@scalar_op
def normal(a: RVLike, b: RVLike) -> InfixRV[ir.Normal | ir.VMap[ir.Normal]]: ...
@scalar_op
def normal_prec(a: RVLike, b: RVLike) -> InfixRV[ir.NormalPrec | ir.VMap[ir.NormalPrec]]: ...
@scalar_op
def lognormal(a: RVLike, b: RVLike) -> InfixRV[ir.Lognormal | ir.VMap[ir.Lognormal]]: ...
@scalar_op
def cauchy(a: RVLike, b: RVLike) -> InfixRV[ir.Cauchy | ir.VMap[ir.Cauchy]]: ...
@scalar_op
def bernoulli(a: RVLike) -> InfixRV[ir.Bernoulli | ir.VMap[ir.Bernoulli]]: ...
@scalar_op
def bernoulli_logit(a: RVLike) -> InfixRV[ir.BernoulliLogit | ir.VMap[ir.BernoulliLogit]]: ...
@scalar_op
def binomial(a: RVLike, b: RVLike) -> InfixRV[ir.Binomial | ir.VMap[ir.Binomial]]: ...
@scalar_op
def uniform(a: RVLike, b: RVLike) -> InfixRV[ir.Uniform | ir.VMap[ir.Uniform]]: ...
@scalar_op
def beta(a: RVLike, b: RVLike) -> InfixRV[ir.Beta | ir.VMap[ir.Beta]]: ...
@scalar_op
def beta_binomial(a: RVLike, b: RVLike, c: RVLike) -> InfixRV[ir.BetaBinomial | ir.VMap[ir.BetaBinomial]]: ...
@scalar_op
def exponential(a: RVLike) -> InfixRV[ir.Exponential | ir.VMap[ir.Exponential]]: ...
@scalar_op
def gamma(a: RVLike, b: RVLike) -> InfixRV[ir.Gamma | ir.VMap[ir.Gamma]]: ...
@scalar_op
def poisson(a: RVLike) -> InfixRV[ir.Poisson | ir.VMap[ir.Poisson]]: ...
@scalar_op
def student_t(a: RVLike, b: RVLike, c: RVLike) -> InfixRV[ir.StudentT | ir.VMap[ir.StudentT]]: ...


# add extra doctests for student_t to check kwargs REALLY work
student_t.__doc__ = (
    typing.cast(str, student_t.__doc__)
    + """
    Also, let's check that we can mix positional and keyword arguments.

    >>> student_t(nu=0.1, mu=0.2, sigma=0.4)
    InfixRV(StudentT(), InfixRV(Constant(0.1)), InfixRV(Constant(0.2)), InfixRV(Constant(0.4)))

    >>> student_t(0.1, mu=0.2, sigma=0.4)
    InfixRV(StudentT(), InfixRV(Constant(0.1)), InfixRV(Constant(0.2)), InfixRV(Constant(0.4)))

    >>> student_t(0.1, sigma=0.4, mu=0.2)
    InfixRV(StudentT(), InfixRV(Constant(0.1)), InfixRV(Constant(0.2)), InfixRV(Constant(0.4)))

    >>> student_t(sigma=0.4, nu=0.1, mu=0.2)
    InfixRV(StudentT(), InfixRV(Constant(0.1)), InfixRV(Constant(0.2)), InfixRV(Constant(0.4)))
    """
)


def expit(a):
    "An alias for `inv_logit`"
    return inv_logit(a)


def sigmoid(a):
    "An alias for `inv_logit`"
    return inv_logit(a)


def sqrt(x: RVLike) -> InfixRV[ir.Pow | VMap[ir.Pow]]:
    """
    Take a square root.

    sqrt(x) is an alias for pow(x, 0.5)
    """
    return pow(x, 0.5)
