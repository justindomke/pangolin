"""
This package defines a special subtype of RV that supports operator overloading
"""

from __future__ import annotations
from pangolin.ir import RV, Op, Constant, ScalarOp, VMap
from pangolin import ir
from numpy.typing import ArrayLike
import jax
import numpy as np
from typing import TypeVar, Type, Callable, Sequence, TYPE_CHECKING
from pangolin.util import comma_separated, camel_case_to_snake_case
import inspect
from typing import Generic, TypeAlias, Final
import os
import sys
import inspect
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

RVLike: TypeAlias = RV | ArrayLike

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

    Suppose `f(a,b,c)` is a scalar function. Then here is how broadcasting behaves:

    +-----------------------------------+--------+----------+-----------+
    |                                   | Broadcasting mode             |
    +                                   +--------+----------+-----------+
    |                                   | OFF    | SIMPLE   | NUMPY     |
    +-----------+-----------+-----------+--------+----------+-----------+
    | `a.shape` | `b.shape` | `c.shape` | `f(a,b,c).shape`              |
    +===========+===========+===========+========+==========+===========+
    | `()`      | `()`      | `()`      | `()`   | `()`     | `()`      |
    +-----------+-----------+-----------+--------+----------+-----------+
    | `(5,)`    | `()`      | `()`      | ✖️     | `(5,)`   | `(5,)`    |
    +-----------+-----------+-----------+--------+----------+-----------+
    | `(5,)`    | `(5,)`    | `()`      | ✖️     | `(5,)`   | `(5,)`    |
    +-----------+-----------+-----------+--------+----------+-----------+
    | `(5,)`    | `(5,)`    | `(5,)`    | ✖️     | `(5,)`   | `(5,)`    |
    +-----------+-----------+-----------+--------+----------+-----------+
    | `(5,9)`   | `()`      | `()`      | ✖️     | `(5,9)`  | `(5,9)`   |
    +-----------+-----------+-----------+--------+----------+-----------+
    | `(5,9)`   | `(5,9)`   | `()`      | ✖️     | `(5,9)`  | `(5,9)`   |
    +-----------+-----------+-----------+--------+----------+-----------+
    | `(5,9)`   | `(5,9)`   | `(5,9)`   | ✖️     | `(5,9)`  | `(5,9)`   |
    +-----------+-----------+-----------+--------+----------+-----------+
    | `(9,)`    | `(5,9)`   | `()`      | ✖️     | ✖️       | `(5,9)`   |
    +-----------+-----------+-----------+--------+----------+-----------+
    | `(5,)`    | `(6,)`    | `()`      | ✖️     | ✖️       | ✖️        |
    +-----------+-----------+-----------+--------+----------+-----------+

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

    >> pi.config.broadcasting = pi.Broadcasting.SIMPLE

    You can also change this behavior temporarily in a code block by using the
    `broadcasting_off`, `broadcasting_simple`, and `broadcasting_numpy`
    context managers.

    >>> from pangolin import interface as pi
    >>> with pi.broadcasting_off():
    ...     x = pi.normal(np.zeros(3), np.ones((5,3)))
    Traceback (most recent call last):
    ...
    ValueError: Normal op got parent shapes ((3,), (5, 3)) not all scalar.
    >>> with pi.broadcasting_numpy():
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

    Access via `config`.

    Attributes
    ----------
    broadcasting
        Controls broadcasting behavior for scalar operations.
        Default: `ScalarBroadcasting.OFF`.

    Warning:
        Do not instantiate this class directly. Use the `config` module attribute.

    See Also
    --------
    override : Context manager for temporary config changes.
    """

    broadcasting: Broadcasting = Broadcasting.SIMPLE


config: Final[Config] = Config()
config.__doc__ = """The singleton configuration instance. Use this instead of instantiating `Config`."""


@contextmanager
def override(**kwargs):
    """
    Temporarily override config values.

    A context manager that sets config values for the duration of
    the block, then restores the original values on exit.

    Args:
        kwargs: Config attribute names and their temporary values.
            See :class:`Config` for available options.

    Raises:
        AttributeError: If a key doesn't match a config attribute.
        ValueError: If a value is invalid for that config option.

    Example:
        >>> from pangolin.interface import override, Broadcasting
        >>> config.broadcasting
        <Broadcasting.SIMPLE: 'simple'>
        >>> with override(broadcasting=Broadcasting.OFF):
        ...     config.broadcasting
        <Broadcasting.OFF: 'off'>
        >>> config.broadcasting
        <Broadcasting.SIMPLE: 'simple'>
    """
    originals = {}
    for key, value in kwargs.items():
        originals[key] = getattr(config, key)
        setattr(config, key, value)
    try:
        yield
    finally:
        for key, value in originals.items():
            setattr(config, key, value)


def broadcasting_off():
    """
    Temprarily turn scalar broadcasting off. Alias for `override(broadcasting=Broadcasting.OFF)`
    """
    return override(broadcasting=Broadcasting.OFF)


def broadcasting_simple():
    """
    Temprarily use simple scalar broadcasting. Alias for `override(broadcasting=Broadcasting.SIMPLE)`
    """
    return override(broadcasting=Broadcasting.SIMPLE)


def broadcasting_numpy():
    """
    Temprarily use numpy-scalar broadcasting. Alias for `override(broadcasting=Broadcasting.NUMPY)`
    """
    return override(broadcasting=Broadcasting.NUMPY)


########################################################################################
# The core InfixRV class. Like an RV except has infix operations
########################################################################################


OpU = TypeVar("OpU", bound=Op)


class InfixRV(RV[OpU], Generic[OpU]):
    """An Infix RV is exactly like a standard `pangolin.ir.RV` except it supports infix
    operations.

    This is a generic type, so you may write `InfixRV[OpClass]` as a type hint.

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

    def __init__(self, op: OpU, *parents: RV):
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

    _IdxType = RVLike | slice | type(Ellipsis)

    def __getitem__(self, idx: _IdxType | tuple[_IdxType, ...]):
        """
        You can index an `RV` with the `[]` operators, e.g. as `A[B,C]`.

        Note that indexing with this interface is different (and simpler) than NumPy or JAX:

        First, indexing is by default fully-orthogonal. This is done to avoid the `utter insanity <https://numpy.org/doc/stable/user/basics.indexing.html>`_ that is NumPy indexing with broadcasting, basic indexing, advanced indexing, and combinations of basic and advanced indexing. In this interface, if `A`, `B`, and `C` are RVs, then `A[B,C].shape == A.shape + B.shape`, and similarly if `B` or `C` are int / list of (list of) int / numpy array / slice.

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


########################################################################################
# makerv
########################################################################################


def constant(value: ArrayLike) -> InfixRV[Constant]:
    """Create a constant RV

    Parameters
    ----------
    value: ArrayLike
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


# def non_infix_rv(x):
#     if isinstance(x, RV):
#         if not isinstance(x, InfixRV):
#             return True
#     return False


def makerv(x: RVLike) -> RV:
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

    if isinstance(arg, RV):
        return arg.shape
    else:
        return np.shape(arg)


########################################################################################
# Auto-generate interface functions for all scalar Ops
########################################################################################


def create_rv(op: OpU, *args) -> InfixRV[OpU]:
    args = tuple(makerv(a) for a in args)
    # args = tuple(a if isinstance(a,RV) else constant(a) for a in args)
    op.get_shape(*[a.shape for a in args])  # checks shapes
    return InfixRV(op, *args)


def _scalar_op_doc(OpClass):
    op = OpClass
    expected_parents = op._expected_parents

    if op.random:
        __doc__ = f"Creates a {str(OpClass.__name__)} distributed RV."
    else:
        __doc__ = f"Creates an RV by applying {str(OpClass.__name__)} to parents."

    __doc__ += """
    
    Args:
    """

    for p in expected_parents:
        __doc__ += f"""
        {p}: {expected_parents[p]}. Must be scalar.
    """

    __doc__ += f"""
    Returns:
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


def vmap_scalars_simple(op: Op, *parent_shapes: ir.Shape) -> Op:
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

    in_axes = tuple(0 if shape == array_shape else None for shape in parent_shapes)

    new_op = op
    for size in reversed(array_shape):
        new_op = VMap(new_op, in_axes, size)

    assert new_op.get_shape(*parent_shapes) == array_shape, "Pangolin bug"

    return new_op


def vmap_scalars_numpy(op: Op, *parent_shapes: ir.Shape) -> Op:
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

        new_op = VMap(new_op, tuple(in_axes), size)

    assert new_op.get_shape(*parent_shapes) == array_shape, "Pangolin bug"

    return new_op

    # the obvious way to generalize this to handle singleten dimensions would be to change to
    #
    # (new_op, new_parents) = vmap_scalars_numpy(op: Op, *parents: ir.RV)
    #
    # where new_parents could be the same as old, or might include "squeeze" operations
    # (will also need to create ir.Squeeze)


# def scalar_fun_factory(OpClass: type[ScalarOp], /):
#     import makefun

#     op = OpClass()
#     expected_parents = op._expected_parents  # type: ignore

#     def fun(*args, **kwargs):
#         if SCALAR_BROADCASTING[0] == "off":
#             return create_rv(op, *args, *[kwargs[a] for a in kwargs])
#         elif SCALAR_BROADCASTING[0] == "simple":
#             positional_args = args + tuple(kwargs[a] for a in kwargs)
#             parent_shapes = [get_shape(arg) for arg in positional_args]
#             vmapped_op = vmap_scalars_simple(op, *parent_shapes)
#             return create_rv(vmapped_op, *positional_args)
#         elif SCALAR_BROADCASTING[0] == "numpy":
#             positional_args = args + tuple(kwargs[a] for a in kwargs)
#             parent_shapes = [get_shape(arg) for arg in positional_args]
#             vmapped_op = vmap_scalars_numpy(op, *parent_shapes)
#             return create_rv(vmapped_op, *positional_args)
#         else:
#             raise Exception(f"Unknown scalar broadcasting model: {SCALAR_BROADCASTING}")

#     func_sig = (
#         f"{op.name}"
#         + comma_separated([f"{a}:'RVLike'" for a in expected_parents])
#         + " -> InfixRV["
#         + f"ir.{OpClass.__name__}"
#         + "]"
#     )

#     fun = makefun.create_function(
#         func_sig, fun, dict_globals={"RVLike", RVLike}
#     )  # useless?
#     fun.__doc__ = _scalar_op_doc(OpClass)
#     fun.__module__ = "pangolin.interface"
#     return fun


def scalar_fun_factory(OpClass: type[ScalarOp]):
    import makefun

    op = OpClass()
    expected_parents = op._expected_parents  # type: ignore

    def fun(*args, **kwargs):
        if config.broadcasting == Broadcasting.OFF:
            return create_rv(op, *args, *[kwargs[a] for a in kwargs])
        elif config.broadcasting == Broadcasting.SIMPLE:
            positional_args = args + tuple(kwargs[a] for a in kwargs)
            parent_shapes = [get_shape(arg) for arg in positional_args]
            vmapped_op = vmap_scalars_simple(op, *parent_shapes)
            return create_rv(vmapped_op, *positional_args)
        elif config.broadcasting == Broadcasting.NUMPY:
            positional_args = args + tuple(kwargs[a] for a in kwargs)
            parent_shapes = [get_shape(arg) for arg in positional_args]
            vmapped_op = vmap_scalars_numpy(op, *parent_shapes)
            return create_rv(vmapped_op, *positional_args)
        else:
            raise Exception(f"Unknown scalar broadcasting model: {config.broadcasting}")

    func_sig = (
        f"{op.name}"
        + comma_separated([f"{a}:'RVLike'" for a in expected_parents])
        + " -> InfixRV["
        + f"ir.{OpClass.__name__}"
        + "]"
    )

    fun = makefun.create_function(
        func_sig, fun, dict_globals={"RVLike", RVLike}
    )  # useless?
    fun.__doc__ = _scalar_op_doc(OpClass)
    fun.__module__ = "pangolin.interface"
    return fun


if TYPE_CHECKING:
    # stubs so type-checker doesn't fail on InfixRV
    def add(a: RVLike, b: RVLike) -> InfixRV[ir.Add]: ...
    def sub(a: RVLike, b: RVLike) -> InfixRV[ir.Sub]: ...
    def mul(a: RVLike, b: RVLike) -> InfixRV[ir.Mul]: ...
    def div(a: RVLike, b: RVLike) -> InfixRV[ir.Div]: ...
    def pow(a: RVLike, b: RVLike) -> InfixRV[ir.Pow]: ...


def _populate_scalar_funs():
    module = sys.modules[__name__]

    for name, cls in vars(ir).items():
        if (
            isinstance(cls, type)
            and issubclass(cls, ir.ScalarOp)
            and not inspect.isabstract(cls)
        ):
            setattr(
                module,
                camel_case_to_snake_case(name),
                scalar_fun_factory(cls),
            )


_populate_scalar_funs()


expit = scalar_fun_factory(ir.InvLogit)
expit.__doc__ = "Equivalent to `inv_logit`"
sigmoid = scalar_fun_factory(ir.InvLogit)
sigmoid.__doc__ = "Equivalent to `inv_logit`"


def sqrt(x: RVLike) -> InfixRV[ir.Pow]:
    """
    Take a square root.

    sqrt(x) is an alias for pow(x,0.5)
    """
    return pow(x, 0.5)


########################################################################################
# Multivariate funs
########################################################################################


def matmul(a: RVLike, b: RVLike) -> RV[ir.Matmul]:
    """
    Matrix product of two arrays. The behavior follows that of
    `numpy.matmul <https://numpy.org/doc/stable/reference/generated/numpy.matmul.html>`_
    except that `a` and `b` must both be 1-D or 2-D arrays. In particular:

    * If `a` and `b` are both 1-D then this represents an inner-product.
    * If `a` is 1-D and `b` is 2-D then this represents vector/matrix multiplication
    * If `a` is 2-D and `b` is 1-D then this represents matrix/vector multiplication
    * If `a` and `b` are both 2-D then this represents matrix/matrix multiplication

    Parameters
    ----------
    a
        first argument (1d or 2d array)
    b
        second argument (1d or 2d array, matching shape of `a`)
    """
    return create_rv(ir.Matmul(), a, b)


def inv(a: RVLike) -> RV[ir.Inv]:
    """
    Take the inverse of a matrix. Input must be a 2-D square (invertible) array.
    """
    return create_rv(ir.Inv(), a)


def softmax(a: RVLike) -> RV[ir.Softmax]:
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


########################################################################################
# Multivariate dists
########################################################################################


def multi_normal(mean: RVLike, cov: RVLike) -> RV[ir.MultiNormal]:
    """
    Create a multivariate normal distributed random variable.
    Call as `multi_normal(mean,cov)`

    Parameters
    ----------
    mean
        mean (1D array)
    cov
        covariance (2D positive-definite array)
    """
    return create_rv(ir.MultiNormal(), mean, cov)


def categorical(theta: RVLike) -> RV[ir.Categorical]:
    """
    Create a `categorical <https://en.wikipedia.org/wiki/Categorical_distribution>`_
    distributed `RV` where `theta` is a vector of non-negative reals that sums to one.

    Parameters
    ----------
    theta
        positive event probabilities (should sum to one)
    """
    return create_rv(ir.Categorical(), theta)


def multinomial(n: RVLike, p: RVLike) -> RV[ir.Multinomial]:
    """
    Create a `multinomial <https://en.wikipedia.org/wiki/Multinomial_distribution>`_
    distributed random variable. Call as `multinomial(n,p)` where `n` is the number of
    repetitions and `p` is a vector of probabilities that sums to one.

    Parameters
    ----------
    n
        number of repetitions (scalar)
    p
        vector of probabilities (should sum to one)

    """
    return create_rv(ir.Multinomial(), n, p)


def dirichlet(alpha: RVLike) -> RV[ir.Dirichlet]:
    """
    Create a
    `Dirichlet <https://en.wikipedia.org/wiki/Dirichlet_distribution>`_
    distributed random variable.
    Call as `dirichlet(alpha)` where `alpha` is a 1-D vector of positive reals.

    Parameters
    ----------
    alpha
        concentration (vector of positive numbers)
    """
    return create_rv(ir.Dirichlet(), alpha)
