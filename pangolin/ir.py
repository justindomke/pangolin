"""Intermediate representation for probabilistic models.

Pangolin IR is an immutable, backend-independent DAG of random variables. Each `RV` is produced
from parent random variables by either a deterministic function or a conditional distribution.
Constants are constant random variables under the same definition.

The IR represents collections of dependent random variables; execution, sampling, density
evaluation, and inference are provided by backends and inference engines. Models may be defined
directly with the IR, but most users will construct them through `pangolin.interface`.

The IR has two core abstractions: `Op` and `RV`. An `Op` describes a deterministic function or
conditional distribution and never refers to an `RV`. An `RV` is a node in a model DAG, consisting
of one `Op` and an ordered tuple of parent `RV` objects. Thus, an `Op` describes what is applied,
while an `RV` records where it is applied and which other RVs it depends on.

IR objects are immutable after construction. Normal attribute assignment and deletion cannot change
an `Op`, `Bijector`, or `RV`. Constructors copy or normalize mutable inputs where necessary; for
example, sequences are stored as tuples and constant arrays are copied and marked read-only.

Pangolin deliberately provides cooperative rather than "hard" immutability. Hard immutability is too
contrary to Python's object model: Python intentionally provides escape hatches such as
``object.__setattr__``, ``__dict__``, and mutable array flags. Attempting to prohibit every such
escape hatch would add substantial machinery while still providing no security boundary. Code that
deliberately uses those mechanisms is outside the immutability contract.

That's the whole IR. Unlike in Pyro / NumPyro / PyMC / Tensorflow Probability, Random variables do
not have names, and there is no notion of a "model" class. In Pangolin, you just work with RVs. You
can assign those RVs to named variables or put them into tuples or lists or dicts if you want. But
that's up to you.

Other notes:

* All `RV` objects have static shapes. The shape of an `RV` is recursively determined by its `Op`
  and the shapes of its parent `RV` objects.

Core types:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Purpose
   * - `Shape`
     - Static shape of an `RV`
   * - `Op`
     - Base class for deterministic functions and distributions
   * - `ScalarOp`
     - Base class for operations with scalar inputs and output
   * - `RV`
     - Node in a probabilistic model DAG
   * - `Bijector`
     - Descriptor for an invertible transformation
   * - `VMapStack`
     - Type-hint-only alias for one or more nested `VMap` operations

Types of `Op`:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Type
     - Ops
   * - Constants
     - `Constant`
   * - Identity
     - `Identity`
   * - Arithmetic
     - `Add`, `Sub`, `Mul`, `Div`
   * - Trigonometry
     - `Arccos`, `Arccosh`, `Arcsin`, `Arcsinh`, `Arctan`, `Arctanh`, `Cos`, `Cosh`,
       `Sin`, `Sinh`, `Tan`, `Tanh`
   * - Other scalar functions
     - `Pow`, `Abs`, `Minimum`, `Maximum`, `Exp`, `InvLogit`, `Log`, `Loggamma`, `Logit`, `Step`
   * - Linear algebra
     - `Matmul`, `Transpose`, `Inv`, `Cholesky`
   * - Other multivariate functions
     - `Concatenate`, `Sum`, `Softmax`, `Diag`, `DiagMatrix`
   * - Scalar distributions
     - `Normal`, `NormalPrec`, `Lognormal`, `Cauchy`, `Bernoulli`, `BernoulliLogit`,
       `Beta`, `Binomial`, `Categorical`, `Uniform`, `BetaBinomial`, `Exponential`,
       `Gamma`, `Poisson`, `StudentT`
   * - Multivariate distributions
     - `MultiNormal`, `Multinomial`, `Dirichlet`, `Wishart`
   * - Control flow
     - `VMap`, `Composite`, `Scan`
   * - Indexing
     - `Index`
   * - Transformations
     - `Bijected`, `Truncated`
   * - Distribution properties
     - `LogProb`, `CDF`, `Quantile`
   * - Differentiation
     - `VJP`

In addition, this module provides the following utilities:

* `split_shape`: Remove one axis from a shape.
* `rv_equal`: Test whether two random variables are mathematically equal.
* `print_upstream`: Print a readable table describing an entire graph.
* `graph_upstream`: Print a graph as a Unicode DAG.

---

"""

from __future__ import annotations  # so it's possible to preserve type aliases

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Callable, Final, Literal, Mapping, Sequence, cast, final, overload

import numpy as np
from numpy.typing import ArrayLike

from pangolin import dag, util
from pangolin.util import PyTree

__all__ = [
    # Core IR types
    "Shape",
    "Op",
    "ScalarOp",
    "RV",
    "RVInterface",
    "NO_INTERFACE",
    "Bijector",
    # Deterministic Ops
    "Identity",
    "Constant",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Arccos",
    "Arccosh",
    "Arcsin",
    "Arcsinh",
    "Arctan",
    "Arctanh",
    "Cos",
    "Cosh",
    "Sin",
    "Sinh",
    "Tan",
    "Tanh",
    "Abs",
    "Minimum",
    "Maximum",
    "Exp",
    "InvLogit",
    "Log",
    "Logit",
    "Step",
    "Loggamma",
    "Matmul",
    "Inv",
    "Cholesky",
    "Transpose",
    "Diag",
    "DiagMatrix",
    "Softmax",
    "Sum",
    "Concatenate",
    "Index",
    # Distribution Ops
    "Normal",
    "NormalPrec",
    "Lognormal",
    "Bernoulli",
    "BernoulliLogit",
    "Binomial",
    "Cauchy",
    "Uniform",
    "Beta",
    "Exponential",
    "Gamma",
    "Poisson",
    "BetaBinomial",
    "StudentT",
    "MultiNormal",
    "Categorical",
    "Multinomial",
    "Dirichlet",
    "Wishart",
    # Higher-order Ops
    "VMap",
    "VMapStack",
    "Composite",
    "Scan",
    "Bijected",
    "Truncated",
    "LogProb",
    "CDF",
    "Quantile",
    "VJP",
    # Shape helpers
    "split_shape",
    # Graph utilities
    "rv_equal",
    "print_upstream",
    "graph_upstream",
]


type Shape = tuple[int, ...]
"""Static shape represented by a tuple of positive integer dimensions.

The empty tuple ``()`` represents a scalar.
"""

# --------------------------------------------------------------------------------------------------
# The fundamental Op class
# --------------------------------------------------------------------------------------------------


class Op(ABC):
    """Abstract base class for operators. An `Op` represents a deterministic function or
    distribution. All functionality for sampling or evaluating densities is left to inference
    backends. An `Op` is frozen after initialization.

    Simple `Op` objects typically have no parameters. For example, ``Add()`` just represents scalar
    addition. The `Op` itself does not indicate what variables are being added together. Similarly,
    ``Normal()`` just represents a normal distribution, but it does not itself indicate what the
    mean and scale of that distribution are. More complex classes have constructors that take any
    parameters that must be fixed in order to maintain fixed shapes. For example, `Sum` takes the
    axis to sum over.

    A concrete `Op` class must provide `Op.random`, indicating if an op is a conditional
    distribution (``True``) or a deterministic function (``False``). For classes where this is
    fixed, declare a boolean attribute with a docstring stating its value. For classes where it
    depends on the particular instance (e.g. `VMap`), provide a documented property returning a
    ``bool``. Both are accessed as ``op.random``.

    A random `Op` must similarly provide `Op.discrete` as a boolean attribute or a property
    returning a boolean: ``True`` for discrete distributions and ``False`` for continuous ones.
    The base property returns ``None``, which deterministic Ops can inherit. Higher-order Ops may
    derive both properties from a contained `Op`. As with `Op.random`, concrete subclasses document
    their fixed value or derivation rule. `Op.__init__` checks that random Ops provide a boolean;
    subclasses must initialize fields used by these properties before calling ``super().__init__()``.

    Every concrete `Op` provides a public `Op.get_shape` method. It accepts the shapes of the `Op`'s
    ordered parents, validates their number and compatibility, and returns the static output
    `Shape`. Parent requirements and output rules are part of each concrete `Op`'s documentation.
    Because different Ops have different arities, the base class deliberately types `Op.get_shape`
    as a weak callable; concrete methods expose their precise signatures wherever the arity is
    fixed.

    An `Op` must provide structural equality and a compatible hash, as documented by
    `Op.__eq__`. The base implementation compares concrete types, which is sufficient for Ops with
    no constructor state. Ops with constructor state must override both `Op.__eq__` and
    ``Op.__hash__``.
    """

    _frozen = False
    get_shape: Callable[..., Shape]
    """Compute the output shape from the ordered shapes of an `Op`'s parents.

    Each concrete `Op` implements this as a method taking one `Shape` argument for each parent, in
    parent order, and returning the output `Shape`. Fixed-arity Ops should name those arguments
    explicitly. Ops whose arity depends on constructor state may accept ``*parent_shapes``, but
    must validate the number of arguments at runtime.

    The implementation must validate that the parent shapes are valid for the operation and raise
    an exception when they are not. `RV` calls this method during construction and stores its result
    unchanged, so this is the operation's validation boundary for parent shapes.

    Examples
    --------
    A fixed matrix-vector operation could implement ``get_shape`` as follows:

    >>> class MatrixVectorProduct(Op):
    ...     random = False
    ...
    ...     def get_shape(self, matrix_shape: Shape, vector_shape: Shape) -> Shape:
    ...         if len(matrix_shape) != 2:
    ...             raise ValueError("matrix parent must be two-dimensional")
    ...         if len(vector_shape) != 1:
    ...             raise ValueError("vector parent must be one-dimensional")
    ...         if matrix_shape[1] != vector_shape[0]:
    ...             raise ValueError("matrix and vector dimensions must match")
    ...         return (matrix_shape[0],)

    The arguments correspond to the two parents in order. Valid shapes produce the output shape:

    >>> op = MatrixVectorProduct()
    >>> op.get_shape((3, 4), (4,))
    (3,)

    Invalid shapes raise an exception rather than producing an invalid `RV`:

    >>> op.get_shape((3, 4), (5,))
    Traceback (most recent call last):
        ...
    ValueError: matrix and vector dimensions must match
    """

    def __init__(self):
        if self.random and not isinstance(self.discrete, bool):
            raise TypeError("Random Ops must declare discrete as True or False")
        self._frozen = True  # freeze after init

    @property
    @abstractmethod
    def random(self) -> bool:
        """Is this class a distribution (``True``) or a deterministic function (``False``)?"""
        ...

    @property
    def discrete(self) -> bool | None:
        """Return whether this distribution is discrete, or None when not applicable.

        Returns
        -------
        bool | None
            ``True`` for a discrete distribution and ``False`` for a continuous distribution.
            Deterministic Ops inherit the default ``None``. Random Ops must override this
            property with a boolean attribute or a property returning a boolean.
        """
        return None

    def __eq__(self, other) -> bool:
        """Return whether two Ops have the same IR structure.

        Ops are equal when they have the same concrete type and equal normalized constructor state.
        Constructors may normalize equivalent inputs, such as lists and tuples, to the same
        representation. This is structural equality, not algebraic equivalence. Equal Ops must have
        equal hashes.

        Examples
        --------
        Stateless Ops of the same type are equal.

        >>> Normal() == Normal()
        True

        Normalized constructor state is compared.

        >>> VMap(Normal(), [0, None]) == VMap(Normal(), (0, None))
        True

        Different constructor state is unequal.

        >>> Sum(0) == Sum(1)
        False

        Equality is structural, not algebraic.

        >>> Add() == Composite(2, (Add(),), ((0, 1),))
        False
        """
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))

    def __setattr__(self, key, value):
        if self._frozen:
            raise TypeError("Ops are immutable after init.")
        object.__setattr__(self, key, value)

    def __delattr__(self, key):
        if self._frozen:
            raise TypeError("Ops are immutable after init.")
        object.__delattr__(self, key)

    @property
    def name(self) -> str:
        "Returns the name of the op class as a string"
        return type(self).__name__

    def __repr__(self):
        """Return a structural Python representation of the operation.

        The representation is constructor-shaped and recursively uses ``repr`` for contained IR
        objects. It is intended for inspecting the structure of the IR, but is not a serialization
        format and is not guaranteed to be executable Python. In particular, bulk data may be
        abbreviated.
        """
        return self.name + "()"

    def __str__(self) -> str:
        """Return a compact, human-readable representation of the operation.

        This representation recursively uses ``str`` for contained IR objects and favors readable
        expressions, such as ``normal`` instead of ``Normal()``. It is not a serialization format or
        stable syntax.
        """
        return util.camel_case_to_snake_case(self.name)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        is_abstract = ABC in cls.__bases__

        if not is_abstract:
            if not callable(getattr(cls, "get_shape", None)):
                raise TypeError(f"Class '{cls.__name__}' must define 'get_shape'.")


# --------------------------------------------------------------------------------------------------
# The identity function
# --------------------------------------------------------------------------------------------------


class Identity(Op):
    """Return one parent unchanged."""

    random: bool = False
    """False"""

    def get_shape(self, par_shape: Shape) -> Shape:
        """Return the parent shape unchanged.

        Parameters
        ----------
        par_shape
            Shape of the single parent.

        Returns
        -------
        Shape
            ``par_shape``.
        """
        return par_shape


# --------------------------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------------------------


class Constant(Op):
    """Represent a numeric constant with no parents.

    The input is normalized to a C-contiguous NumPy array whose buffer is immutable. The public
    ``value`` remains an ordinary NumPy array, but neither it nor a view sharing its buffer can be
    made writable.

    `Constant` equality is structural. Two `Constant` objects are equal when they have the same
    concrete type, dtype, shape, and C-order data bytes. Consequently, signed zeros are distinct,
    while NaNs are equal only when their stored bit representations are equal. Hashing uses exactly
    the same state.

    Parameters
    ----------
    value
        Numeric value that NumPy can convert to an array.

    Attributes
    ----------
    value
        Immutable NumPy array containing the normalized value.

    Raises
    ------
    ValueError
        If the value is empty or does not have a numeric dtype.

    Examples
    --------
    >>> constant = Constant([[1.0, 2.0]])
    >>> constant.value.tolist()
    [[1.0, 2.0]]
    >>> constant.value.flags.writeable
    False
    """

    random: bool = False
    """False"""

    def __init__(self, value: ArrayLike):
        array = np.asarray(value)

        if array.size == 0:
            raise ValueError(
                f"Pangolin does not support empty constants (got shape {array.shape})."
            )

        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"Data {value} does not produce numeric dtype")

        data = np.ascontiguousarray(array).tobytes()
        self.value = np.frombuffer(data, dtype=array.dtype).reshape(array.shape)
        super().__init__()

    def get_shape(self) -> Shape:
        """Return the shape of the stored constant.

        `Constant` has no parents, so this method accepts no parent shapes.

        Returns
        -------
        Shape
            Shape of ``value``.
        """
        return self.value.shape

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` has the same stored constant data."""
        if not isinstance(other, Constant) or type(self) is not type(other):
            return False
        return (
            self.value.dtype == other.value.dtype
            and self.value.shape == other.value.shape
            and self.value.tobytes() == other.value.tobytes()
        )

    def __hash__(self):
        """Return a hash of the Constant's concrete type and stored data."""
        return hash((type(self), self.value.dtype, self.value.shape, self.value.tobytes()))

    def __repr__(self):
        numpy_repr = repr(np.array(self.value))
        array_contents = numpy_repr[len("array(") : -1]
        compact_contents = array_contents.replace("\n", "").replace(" ", "")
        return f"Constant({compact_contents})"

    def __str__(self):
        numpy_value = np.array(self.value)
        with np.printoptions(threshold=5, linewidth=50, edgeitems=2):
            return np.array2string(numpy_value, precision=3).replace("\n", "").replace("  ", " ")


# --------------------------------------------------------------------------------------------------
# Abstract ScalarOp class
# --------------------------------------------------------------------------------------------------


class ScalarOp(Op, ABC):
    """Base class for operations whose parents and output are all scalar.

    Scalar operations have class-fixed randomness and a class-fixed number of parents. Concrete
    subclasses declare ``random`` and ``_expected_parents``. A parent count generates generic
    parent names; a mapping supplies names and descriptions for documentation and interface
    functions.

    Random operations may also provide ``_wikipedia`` to override the Wikipedia page inferred from
    the class name, or set it to ``None`` to omit the link. Additional documentation can be
    supplied through ``_notes``.
    """

    _expected_parents: int | dict[str, str]
    _wikipedia: str | None = ""
    "Wikipedia page stem; an empty string uses the class name, and None omits the link."
    _notes: list[str] = []

    @property
    def _num_parents(self) -> int:
        if isinstance(self._expected_parents, int):
            return self._expected_parents
        else:
            return len(self._expected_parents)

    def get_shape(self, *parents_shapes: Shape) -> Shape:
        """Return the scalar output shape after validating all parent shapes.

        Parameters
        ----------
        *parents_shapes
            One shape for each parent declared by the concrete `ScalarOp`. Every parent must have
            scalar shape ``()``.

        Returns
        -------
        Shape
            Always ``()``.

        Raises
        ------
        TypeError
            If the number of parent shapes is incorrect.
        ValueError
            If any parent is nonscalar.
        """

        if len(parents_shapes) != self._num_parents:
            raise TypeError(
                f"{self.name} op got {len(parents_shapes)} parent(s) but expected"
                f" {self._num_parents}."
            )
        for shape in parents_shapes:
            if shape != ():
                raise ValueError(
                    f"{self.name} op got parent shapes {parents_shapes} not all scalar."
                )
        return ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls.__doc__ = _get_scalar_op_docstring(cls)


# --------------------------------------------------------------------------------------------------
# Create scalar Ops
# --------------------------------------------------------------------------------------------------


def _generate_expected_parents(num_parents: int) -> dict[str, str]:
    start_code = ord("a")
    expected_parents = {}
    for n in range(num_parents):
        expected_parents[str(chr(start_code + n))] = f"parent {n}"
    return expected_parents


def _get_scalar_op_parent_descriptions(cls: type[ScalarOp]) -> dict[str, str]:
    if isinstance(cls._expected_parents, int):
        expected_parents = _generate_expected_parents(cls._expected_parents)
    else:
        expected_parents = cls._expected_parents
    assert isinstance(expected_parents, dict)
    return expected_parents


def _get_scalar_op_wikipedia(cls: type[ScalarOp]) -> str | None:
    if not cls.random or cls._wikipedia is None:
        return None
    if cls._wikipedia:
        return f"https://en.wikipedia.org/wiki/{cls._wikipedia}_distribution"
    return f"https://en.wikipedia.org/wiki/{cls.__name__}_distribution"


def _get_scalar_op_docstring(cls: type[ScalarOp]):
    name = cls.__name__
    random = cls.random
    expected_parents = _get_scalar_op_parent_descriptions(cls)
    wikipedia = _get_scalar_op_wikipedia(cls)

    if random:
        summary = f"Scalar {name} distribution."
    else:
        summary = f"Scalar `{name}` operation."

    lines = [
        summary,
        "",
        "The constructor takes no arguments. When used in an `RV`, this operation",
        "expects the following scalar parents:",
        "",
    ]
    for parent_name, parent_description in expected_parents.items():
        lines.append(f"* ``{parent_name}``: {parent_description}")

    num_parents = len(expected_parents)
    parent_word = "parent" if num_parents == 1 else "parents"

    lines.extend(
        [
            "",
            f"`{name}` expects exactly {num_parents} scalar {parent_word}. Its shape",
            "behavior is inherited from `ScalarOp.get_shape`: Every parent must have scalar shape",
            "``()``, and the output shape is ``()``.",
            "",
            "Examples",
            "--------",
            f">>> op = {name}()",
            ">>> op",
            f"{name}()",
            ">>> print(op)",
            util.camel_case_to_snake_case(name),
            ">>> op.random",
            str(random),
        ]
    )

    if cls._notes:
        lines.extend(["", "Notes", "-----"])
        for note in cls._notes:
            lines.extend([note, ""])

    if wikipedia:
        title = (cls._wikipedia or name).replace("_", " ")
        lines.extend(
            ["", "See Also", "--------", f"`{title} distribution — Wikipedia <{wikipedia}>`__"]
        )

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------------------------------
# Arithmetic
# --------------------------------------------------------------------------------------------------


class Add(ScalarOp):
    _expected_parents = 2
    random: bool = False
    """False"""


class Sub(ScalarOp):
    _expected_parents = 2
    random: bool = False
    """False"""


class Mul(ScalarOp):
    _expected_parents = 2
    random: bool = False
    """False"""


class Div(ScalarOp):
    _expected_parents = 2
    random: bool = False
    """False"""


class Pow(ScalarOp):
    _expected_parents = 2
    random: bool = False
    """False"""


# --------------------------------------------------------------------------------------------------
# Trigonometry
# --------------------------------------------------------------------------------------------------


class Arccos(ScalarOp):
    _expected_parents = 1
    random: bool = False
    """False"""


class Arccosh(ScalarOp):
    _expected_parents = 1
    random: bool = False
    """False"""


class Arcsin(ScalarOp):
    _expected_parents = 1
    random: bool = False
    """False"""


class Arcsinh(ScalarOp):
    _expected_parents = 1
    random: bool = False
    """False"""


class Arctan(ScalarOp):
    _expected_parents = 1
    random: bool = False
    """False"""


class Arctanh(ScalarOp):
    _expected_parents = 1
    random: bool = False
    """False"""


class Cos(ScalarOp):
    _expected_parents = 1
    random: bool = False
    """False"""


class Cosh(ScalarOp):
    _expected_parents = 1
    random: bool = False
    """False"""


class Sin(ScalarOp):
    _expected_parents = 1
    random: bool = False
    """False"""


class Sinh(ScalarOp):
    _expected_parents = 1
    random: bool = False
    """False"""


class Tan(ScalarOp):
    _expected_parents = 1
    random: bool = False
    """False"""


class Tanh(ScalarOp):
    _expected_parents = 1
    random: bool = False
    """False"""


# --------------------------------------------------------------------------------------------------
# Other scalar functions
# --------------------------------------------------------------------------------------------------


class Abs(ScalarOp):
    random: bool = False
    """False"""
    _expected_parents = 1


class Minimum(ScalarOp):
    random: bool = False
    """False"""
    _expected_parents = {"a": "first value", "b": "second value"}
    _notes = ["Returns the smaller of the two values. If either value is NaN, the result is NaN."]


class Maximum(ScalarOp):
    random: bool = False
    """False"""
    _expected_parents = {"a": "first value", "b": "second value"}
    _notes = ["Returns the larger of the two values. If either value is NaN, the result is NaN."]


class Exp(ScalarOp):
    random: bool = False
    """False"""
    _expected_parents = 1


class InvLogit(ScalarOp):
    random: bool = False
    """False"""
    _expected_parents = 1


class Log(ScalarOp):
    random: bool = False
    """False"""
    _expected_parents = 1


class Logit(ScalarOp):
    random: bool = False
    """False"""
    _expected_parents = 1


class Step(ScalarOp):
    random: bool = False
    """False"""
    _expected_parents = 1


class Loggamma(ScalarOp):
    random: bool = False
    """False"""
    _expected_parents = 1
    _notes = ["Do we want ``scipy.special.loggamma`` or ``scipy.special.gammaln``?"]


# --------------------------------------------------------------------------------------------------
# Linear Algebra
# --------------------------------------------------------------------------------------------------


class Matmul(Op):
    """A class that does matrix multiplication, following the rules of ``numpy.matmul``. Currently
    only 1d and 2d arrays are supported.

    Examples
    --------
    >>> op = Matmul()
    >>> op
    Matmul()
    >>> print(op)
    matmul
    >>> op.random
    False
    """

    random: bool = False
    """False"""

    def get_shape(self, a_shape: Shape, b_shape: Shape) -> Shape:
        """Return the shape of a vector or matrix product.

        Parameters
        ----------
        a_shape
            One- or two-dimensional shape of the left parent.
        b_shape
            One- or two-dimensional shape of the right parent. Its contracted dimension must match
            that of ``a_shape``.

        Returns
        -------
        Shape
            Scalar for vector-vector multiplication, a vector for matrix-vector or vector-matrix
            multiplication, and a matrix for matrix-matrix multiplication.

        Raises
        ------
        ValueError
            If either parent is not one- or two-dimensional or their contracted dimensions do not
            match.

        Examples
        --------
        >>> op = Matmul()
        >>> op.get_shape((4,), (4,)) # inner product
        ()
        >>> op.get_shape((5,4), (4,)) # matrix-vector
        (5,)
        >>> op.get_shape((5,), (5,4)) # vector-matrix
        (4,)
        >>> op.get_shape((5,4), (4,3)) # matrix-matrix
        (5, 3)
        >>> op.get_shape((5,5), (10, 10)) # incoherent shapes, error expected
        Traceback (most recent call last):
        ...
        ValueError: Matmul parent shapes do not match ((5, 5) vs. (10, 10))

        """
        if len(a_shape) not in [1, 2]:
            raise ValueError(f"First parent for Matmul must have 1 or 2 dims (got {len(a_shape)}).")

        if len(b_shape) not in [1, 2]:
            raise ValueError(
                f"Second parent for Matmul must have 1 or 2 dims (got {len(b_shape)})."
            )

        a_is_vector = len(a_shape) == 1
        b_is_vector = len(b_shape) == 1

        if a_is_vector and b_is_vector:
            # inner product
            if a_shape != b_shape:
                raise ValueError(f"Matmul parent shapes do not match ({a_shape} vs. {b_shape})")
            return ()
        elif a_is_vector and not b_is_vector:
            # vector-matrix product
            if a_shape[0] != b_shape[0]:
                raise ValueError(f"Matmul parent shapes do not match ({a_shape} vs. {b_shape})")
            return (b_shape[1],)
        elif not a_is_vector and b_is_vector:
            # matrix-vector product
            if a_shape[1] != b_shape[0]:
                raise ValueError(f"Matmul parent shapes do not match ({a_shape} vs. {b_shape})")
            return (a_shape[0],)
        else:
            # matrix-matrix product
            assert len(a_shape) == 2 and len(b_shape) == 2
            if a_shape[1] != b_shape[0]:
                raise ValueError(f"Matmul parent shapes do not match ({a_shape} vs. {b_shape})")
            return (a_shape[0], b_shape[1])


class Inv(Op):
    """Take the inverse of a square matrix."""

    random: bool = False
    """False"""

    def get_shape(self, p_shape: Shape) -> Shape:
        """Return the unchanged shape of a square matrix parent.

        Parameters
        ----------
        p_shape
            Two-dimensional square shape.

        Returns
        -------
        Shape
            ``p_shape``.

        Raises
        ------
        ValueError
            If the parent is not a square matrix.
        """

        if len(p_shape) != 2:
            raise ValueError("inverse only applies to 2d arrays")
        if p_shape[0] != p_shape[1]:
            raise ValueError("inverse only for square arrays")
        return p_shape


class Cholesky(Op):
    """Compute the Cholesky decomposition of a square matrix."""

    random: bool = False
    """False"""

    def get_shape(self, p_shape: Shape) -> Shape:
        """Return the unchanged shape of a square matrix parent.

        Parameters
        ----------
        p_shape
            Two-dimensional square shape.

        Returns
        -------
        Shape
            ``p_shape``.

        Raises
        ------
        ValueError
            If the parent is not a square matrix.
        """

        if len(p_shape) != 2:
            raise ValueError("cholesky only applies to 2d arrays")
        if p_shape[0] != p_shape[1]:
            raise ValueError("cholesky only for square arrays")
        return p_shape


class Transpose(Op):
    """Take a transpose.

    Unlike in many libraries this *only* works for 2D arrays, rather than reversing all dimensions.
    The logic is that 99% of the time, calling transpose on a 1D array and getting the original
    array (like in NumPy / JAX / Pytorch) is a bug. And calling transpose on a 4d array and
    reversing the dimensions is almost never useful. So, 2D transpose only!
    """

    random: bool = False
    """False"""

    def get_shape(self, p_shape: Shape) -> Shape:
        """Reverse the dimensions of a two-dimensional parent shape.

        Parameters
        ----------
        p_shape
            Two-dimensional parent shape.

        Returns
        -------
        Shape
            ``(p_shape[1], p_shape[0])``.

        Raises
        ------
        ValueError
            If the parent is not two-dimensional.
        """

        if len(p_shape) != 2:
            raise ValueError("transpose only applies to 2d arrays")
        return (p_shape[1], p_shape[0])


class Diag(Op):
    """Extract the diagonal of a square matrix."""

    random: bool = False
    """False"""

    def get_shape(self, p_shape: Shape) -> Shape:
        """Return the vector shape of a square matrix's diagonal.

        Parameters
        ----------
        p_shape
            Two-dimensional square shape.

        Returns
        -------
        Shape
            ``(p_shape[0],)``.

        Raises
        ------
        ValueError
            If the parent is not a square matrix.
        """

        if len(p_shape) != 2:
            raise ValueError("diag only applies to 2d arrays")
        if p_shape[0] != p_shape[1]:
            raise ValueError("diag only for square arrays")
        return (p_shape[0],)


class DiagMatrix(Op):
    """Construct a square diagonal matrix from a vector."""

    random: bool = False
    """False"""

    def get_shape(self, p_shape: Shape) -> Shape:
        """Return the square matrix shape constructed from a vector parent.

        Parameters
        ----------
        p_shape
            One-dimensional vector shape.

        Returns
        -------
        Shape
            ``(p_shape[0], p_shape[0])``.

        Raises
        ------
        ValueError
            If the parent is not one-dimensional.
        """

        if len(p_shape) != 1:
            raise ValueError("DiagMatrix only applies to 1d arrays")
        return (p_shape[0], p_shape[0])


# --------------------------------------------------------------------------------------------------
# Other multivariate funs
# --------------------------------------------------------------------------------------------------


class Softmax(Op):
    """Apply softmax to a one-dimensional array.

    The output has the same shape as the input.

    Examples
    --------
    >>> Softmax().get_shape((3,))
    (3,)
    """

    random: bool = False
    """False"""

    def get_shape(self, p_shape: Shape) -> Shape:
        """Return the unchanged shape of a one-dimensional input.

        Parameters
        ----------
        p_shape
            One-dimensional parent shape.

        Returns
        -------
        Shape
            ``p_shape``.

        Raises
        ------
        ValueError
            If the input is not one-dimensional.
        """
        if len(p_shape) != 1:
            raise ValueError(f"Softmax expects a one-dimensional input, got shape {p_shape}")
        return p_shape


class Sum(Op):
    """Sum an array along one axis.

    Parameters
    ----------
    axis
        Axis to sum over. Negative axes count from the end.
    """

    random: bool = False
    """False"""

    def __init__(self, axis):
        if isinstance(axis, np.ndarray) and axis.shape == ():
            axis = int(axis)
        if isinstance(axis, np.integer):
            axis = int(axis)
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("axis for Sum must be an integer")
        self.axis: int = axis
        super().__init__()

    def get_shape(self, x_shape: Shape) -> Shape:
        """Remove the configured axis from the parent shape.

        Parameters
        ----------
        x_shape
            Shape of the array to sum. It must contain ``axis``.

        Returns
        -------
        Shape
            ``x_shape`` with the configured axis removed.

        Raises
        ------
        ValueError
            If ``axis`` is outside ``x_shape`` or ``x_shape`` is scalar.
        """
        ndim = len(x_shape)
        if not -ndim <= self.axis < ndim:
            raise ValueError(
                f"axis {self.axis} is out of bounds for an array with {ndim} dimensions"
            )
        axis = self.axis % ndim
        return x_shape[:axis] + x_shape[axis + 1 :]

    def __repr__(self):
        return f"Sum(axis={self.axis})"

    def __str__(self):
        return f"sum(axis={self.axis})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sum) or type(self) is not type(other):
            return False
        return self.axis == other.axis

    def __hash__(self) -> int:
        return hash((type(self), self.axis))


class Concatenate(Op):
    """Concatenate arrays along an existing axis.

    Every parent must have the same rank and equal dimensions except along ``axis``. The output
    retains that rank and has the sum of the parent dimensions along ``axis``. Unlike stacking,
    concatenation does not introduce a new axis.

    Parameters
    ----------
    axis
        Existing axis along which to concatenate. Negative axes count from the end.

    Examples
    --------
    Concatenate matrices by rows using the default axis:

    >>> Concatenate().get_shape((2, 3), (4, 3))
    (6, 3)

    Concatenate them by columns instead:

    >>> Concatenate(axis=1).get_shape((2, 3), (2, 4))
    (2, 7)
    """

    random: bool = False
    """False"""

    def __init__(self, axis: int = 0):
        if isinstance(axis, np.ndarray) and axis.shape == ():
            axis = int(axis)
        if isinstance(axis, np.integer):
            axis = int(axis)
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("axis for Concatenate must be an integer")
        self.axis: int = axis
        super().__init__()

    def get_shape(self, *parent_shapes: Shape) -> Shape:
        """Join the parent shapes along the configured axis.

        Parameters
        ----------
        *parent_shapes
            One or more equal-rank shapes. Dimensions other than ``axis`` must match.

        Returns
        -------
        Shape
            Common parent shape with the dimensions along ``axis`` summed.

        Raises
        ------
        ValueError
            If there are no parents, ``axis`` is out of bounds, parent ranks differ, or a
            non-concatenated dimension differs.
        """
        if not parent_shapes:
            raise ValueError("Concatenate requires at least one parent")

        ndim = len(parent_shapes[0])
        if not -ndim <= self.axis < ndim:
            raise ValueError(f"axis {self.axis} is out of bounds for arrays with {ndim} dimensions")
        axis = self.axis % ndim

        first_shape = parent_shapes[0]
        for shape in parent_shapes[1:]:
            if len(shape) != ndim:
                raise ValueError("Concatenate parents must have the same number of dimensions")
            if shape[:axis] + shape[axis + 1 :] != first_shape[:axis] + first_shape[axis + 1 :]:
                raise ValueError(
                    f"Concatenate parent shapes must match except along axis {self.axis} "
                    f"(got {first_shape} and {shape})"
                )

        output_shape = list(first_shape)
        output_shape[axis] = sum(shape[axis] for shape in parent_shapes)
        return tuple(output_shape)

    def __repr__(self):
        return f"Concatenate(axis={self.axis})"

    def __str__(self):
        return f"concatenate(axis={self.axis})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Concatenate) or type(self) is not type(other):
            return False
        return self.axis == other.axis

    def __hash__(self) -> int:
        return hash((type(self), self.axis))


# --------------------------------------------------------------------------------------------------
# Scalar distributions
# --------------------------------------------------------------------------------------------------


class Normal(ScalarOp):
    random: bool = True
    """True"""
    discrete: bool = False
    """False"""
    _expected_parents = {
        "mu": "location / mean",
        "sigma": "scale / standard deviation",
    }


class NormalPrec(ScalarOp):
    random: bool = True
    """True"""
    discrete: bool = False
    """False"""
    _expected_parents = {
        "mu": "location / mean",
        "tau": "precision / inverse variance",
    }
    _wikipedia = "Normal"


class Lognormal(ScalarOp):
    random: bool = True
    """True"""
    discrete: bool = False
    """False"""
    _expected_parents = {
        "mu": "logarithm of location",
        "sigma": "logarithm of scale (not sigma squared!)",
    }
    _wikipedia = "Log-normal"


class Bernoulli(ScalarOp):
    random: bool = True
    """True"""
    discrete: bool = True
    """True"""
    _expected_parents = {"theta": "probability (between 0 and 1)"}


class BernoulliLogit(ScalarOp):
    random: bool = True
    """True"""
    discrete: bool = True
    """True"""
    _expected_parents = {"theta": "logit of probability (unbounded)"}
    _wikipedia = None


class Binomial(ScalarOp):
    random: bool = True
    """True"""
    discrete: bool = True
    """True"""
    _expected_parents = {
        "N": "number of trials",
        "theta": "probability of success for each trial",
    }


class Cauchy(ScalarOp):
    random: bool = True
    """True"""
    discrete: bool = False
    """False"""
    _expected_parents = {"mu": "location", "sigma": "scale"}


class Uniform(ScalarOp):
    random: bool = True
    """True"""
    discrete: bool = False
    """False"""
    _expected_parents = {"alpha": "lower bound", "beta": "upper bound"}
    _wikipedia = "Continuous_uniform"


class Beta(ScalarOp):
    random: bool = True
    """True"""
    discrete: bool = False
    """False"""
    _expected_parents = {"alpha": "shape", "beta": "shape"}


class Exponential(ScalarOp):
    random: bool = True
    """True"""
    discrete: bool = False
    """False"""
    _expected_parents = {"beta": "rate / inverse scale"}


class Gamma(ScalarOp):
    random: bool = True
    """True"""
    discrete: bool = False
    """False"""
    _expected_parents = {"alpha": "shape", "beta": "rate / inverse scale"}
    _notes = [
        "This follows `Stan "
        "<https://mc-stan.org/docs/functions-reference/positive_continuous_distributions.html"
        "#gamma-distribution>`__ "
        'in using the "shape/rate" parameterization, *not* the "shape/scale" parameterization.'
    ]


class Poisson(ScalarOp):
    random: bool = True
    """True"""
    discrete: bool = True
    """True"""
    _expected_parents = {"lambd": "lambda"}


class BetaBinomial(ScalarOp):
    random: bool = True
    """True"""
    discrete: bool = True
    """True"""
    _expected_parents = {
        "N": "as in binomial dist",
        "alpha": "as in beta dist",
        "beta": "as in beta dist",
    }
    _wikipedia = "Beta-binomial"
    _notes = [
        "This follows the (N,alpha,beta) convention of `Stan "
        "<https://mc-stan.org/docs/2_19/functions-reference/beta-binomial-distribution.html>`__ "
        "(and Wikipedia). Some other systems (e.g. `Numpyro "
        "<https://num.pyro.ai/en/stable/distributions.html#betabinomial>`__) use alternate "
        "variable orderings. This is no problem for you as a user, since pangolin does the "
        "re-ordering for you based on the backend. But keep it in mind if translating a model "
        "from one system to another."
    ]


class StudentT(ScalarOp):
    random: bool = True
    """True"""
    discrete: bool = False
    """False"""
    _expected_parents = {
        "nu": "degress of freedom",
        "mu": "location (often 0)",
        "sigma": "scale (often 1)",
    }
    _wikipedia = "Student's_t"


# --------------------------------------------------------------------------------------------------
# Multivariate dists
# --------------------------------------------------------------------------------------------------


class MultiNormal(Op):
    """Represent a multivariate normal distribution.

    When used in an `RV`, the parents are a mean vector followed by a covariance matrix.
    """

    random: bool = True
    """True"""
    discrete: bool = False
    """False"""

    def get_shape(self, mean_shape: Shape, covariance_shape: Shape) -> Shape:
        """Return the event-vector shape after validating mean and covariance.

        Parameters
        ----------
        mean_shape
            One-dimensional mean-vector shape ``(n,)``.
        covariance_shape
            Square covariance-matrix shape ``(n, n)``.

        Returns
        -------
        Shape
            ``mean_shape``.

        Raises
        ------
        ValueError
            If the mean is not a vector or the covariance is not a square matrix matching its
            length.
        """
        if len(mean_shape) != 1:
            raise ValueError("first parameter must be a vector.")
        if len(covariance_shape) != 2:
            raise ValueError("second parameter must be a matrix.")
        n = mean_shape[0]
        if covariance_shape != (n, n):
            raise ValueError("second parameter must be matrix with size matching first parameter")
        return (n,)


class Categorical(Op):
    """Represent a categorical distribution.

    A `Categorical` `RV` has one parent: a one-dimensional vector of weights. Its output is scalar.

    Examples
    --------
    >>> Categorical().get_shape((3,))
    ()
    """

    random: bool = True
    """True"""
    discrete: bool = True
    """True"""

    def get_shape(self, weights_shape: Shape) -> Shape:
        """Return the scalar output shape for a vector of weights.

        Parameters
        ----------
        weights_shape
            One-dimensional shape of the categorical weights.

        Returns
        -------
        Shape
            ``()``.

        Raises
        ------
        ValueError
            If the weights are not one-dimensional.
        """
        if len(weights_shape) != 1:
            raise ValueError(
                f"Categorical expects one-dimensional weights, got shape {weights_shape}"
            )
        return ()


class Multinomial(Op):
    """Create a `Multinomial` `Op`. Takes no parameters.

    When used in an `RV`, a multinomial op expects a first parent ``n`` (a scalar) indicating the
    number of observations and a second parent ``p`` indicating a vector of probabilities (a 1D
    array). Note that this is different from Stan (which doesn't need ``n`` to be passed).
    """

    random: bool = True
    """True"""
    discrete: bool = True
    """True"""

    def get_shape(self, n_shape: Shape, p_shape: Shape) -> Shape:
        """Return the count-vector shape after validating both parents.

        Parameters
        ----------
        n_shape
            Scalar shape ``()`` for the number of trials.
        p_shape
            One-dimensional probability-vector shape.

        Returns
        -------
        Shape
            ``p_shape``.

        Raises
        ------
        ValueError
            If the number of trials is nonscalar or the probabilities are not one-dimensional.
        """

        if n_shape != ():
            raise ValueError("First input to Multinomial op must be scalar")
        if len(p_shape) != 1:
            raise ValueError("Second input to Multinomial op must be a 1-d vector")
        return p_shape


class Dirichlet(Op):
    """Create a `Dirichlet` `Op`. Takes no parameters.

    When used in an `RV`, a Dirichlet op expects one parent, namely the concentration
    """

    random: bool = True
    """True"""
    discrete: bool = False
    """False"""

    def get_shape(self, concentration_shape: Shape) -> Shape:
        """Return the concentration-vector shape unchanged.

        Parameters
        ----------
        concentration_shape
            One-dimensional concentration-vector shape.

        Returns
        -------
        Shape
            ``concentration_shape``.

        Raises
        ------
        ValueError
            If the concentration is not one-dimensional.
        """
        if len(concentration_shape) != 1:
            raise ValueError("Dirichlet op must have a single 1-d vector input")
        return concentration_shape


class Wishart(Op):
    """Create a `Wishart` `Op`. Takes no parameters.

    When used in an `RV`, expects two parameters: ``nu`` (degrees of freedom, scalar) and ``S``
    (symmetric positive-definite scale matrix).
    """

    random: bool = True
    """True"""
    discrete: bool = False
    """False"""

    def get_shape(self, nu_shape: Shape, scale_shape: Shape) -> Shape:
        """Return the scale-matrix shape after validating both parents.

        Parameters
        ----------
        nu_shape
            Scalar shape ``()`` for the degrees of freedom.
        scale_shape
            Two-dimensional square scale-matrix shape.

        Returns
        -------
        Shape
            ``scale_shape``.

        Raises
        ------
        ValueError
            If the degrees of freedom are nonscalar or the scale is not a square matrix.
        """
        if nu_shape != ():
            raise ValueError("degrees of freedom for Wishart must be scalar.")

        if len(scale_shape) != 2 or scale_shape[0] != scale_shape[1]:
            raise ValueError("scale for Wishart must be a square matrix.")

        return scale_shape


# --------------------------------------------------------------------------------------------------
# VMap
# --------------------------------------------------------------------------------------------------


def split_shape(shape: Shape, axis: int | None) -> tuple[Shape, int | None]:
    """Remove one axis from a shape.

    Parameters
    ----------
    shape
        Shape from which to remove an axis.
    axis
        Axis to remove, including negative axes, or ``None`` to leave the shape unchanged.

    Returns
    -------
    remaining_shape
        Shape with the selected axis removed.
    axis_size
        Size of the removed axis, or ``None`` when ``axis`` is ``None``.

    Raises
    ------
    TypeError
        If ``axis`` is not an integer or ``None``.
    ValueError
        If ``axis`` is outside the shape.

    Examples
    --------
    >>> split_shape((2, 3, 4), -1)
    ((2, 3), 4)
    >>> split_shape((2, 3, 4), None)
    ((2, 3, 4), None)
    """
    if axis is None:
        return shape, None
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer or None")
    axis = int(axis)

    ndim = len(shape)
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of bounds for an input with {ndim} dimensions")
    normalized_axis = axis % ndim
    new_shape = shape[:normalized_axis] + shape[normalized_axis + 1 :]
    return new_shape, shape[normalized_axis]


def _get_sliced_shapes(
    shapes: Sequence[Shape], in_axes: tuple[int | None, ...], axis_size: int | None
) -> tuple[list[Shape], int]:
    if len(shapes) != len(in_axes):
        raise ValueError(
            f"VMap has {len(in_axes)} in_axes entries but received {len(shapes)} input shapes"
        )

    remaining_shapes = []
    for input_number, (axis, shape) in enumerate(zip(in_axes, shapes, strict=True)):
        new_shape, new_axis_size = split_shape(shape, axis)
        remaining_shapes.append(new_shape)
        if axis_size is None and new_axis_size is not None:
            axis_size = new_axis_size
        elif new_axis_size is not None and axis_size != new_axis_size:
            raise ValueError(
                f"input {input_number} has mapped axis size {new_axis_size}, expected {axis_size}"
            )
    if axis_size is None:
        raise ValueError("axis_size is required when no inputs are mapped")
    if axis_size <= 0:
        raise ValueError("mapped axis size must be positive")
    return remaining_shapes, axis_size


class VMap[BaseOp: Op](Op):
    """Vectorize one operation over axes of its inputs.

    All arguments here are heavily inspired by `jax.lax.vmap
    <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__ although note that `VMap`
    only maps a single `Op`. (The ``vmap`` function in the interface elsewhere takes an arbitrary
    function and transforms it into a graph of `RV` objects with `VMap` operations.)

    Parameters
    ----------
    base_op
        Operation to vectorize.
    in_axes : Sequence[int | None]
        Axis to map for each input, or ``None`` for an unmapped input. Negative axes count from the
        end.
    axis_size
        Positive size of the mapped axis. It is inferred from mapped inputs when omitted and
        required when every input is unmapped.

    Raises
    ------
    TypeError
        If the constructor arguments have invalid types.
    ValueError
        If ``axis_size`` is not positive or cannot be inferred from ``in_axes``.

    Examples
    --------
    >>> # diagonal normal
    >>> op = VMap(Normal(), in_axes=[0, 0])
    >>> op.get_shape((5,), (5,))
    (5,)

    >>> # diagonal normal with shared scale
    >>> op = VMap(Normal(), in_axes=[0, None])
    >>> op.get_shape((5,), ())
    (5,)

    >>> # 2D diagonal normal with shared scale
    >>> op = VMap(VMap(Normal(), in_axes=[0, None]), in_axes=[0, None])
    >>> op.get_shape((6,4), ())
    (6, 4)

    >>> # 2D diagonal normal with "outer product" of location and scale
    >>> op = VMap(VMap(Normal(), in_axes=[None, 0]), in_axes=[0, None])
    >>> op.get_shape((6,), (4,))
    (6, 4)

    >>> # 2D diagonal normal with "outer product" in the other order
    >>> op = VMap(VMap(Normal(), in_axes=[0, None]), in_axes=[None, 0])
    >>> op.get_shape((6,), (4,))
    (4, 6)
    """

    @property
    def random(self) -> bool:
        """Equal to ``base_op.random``"""

        return self.base_op.random

    @property
    def discrete(self) -> bool | None:
        """Equal to ``base_op.discrete``"""

        return self.base_op.discrete

    def __init__(
        self,
        base_op: BaseOp,
        in_axes: Sequence[int | None],
        axis_size: int | None = None,
    ):
        if not isinstance(base_op, Op):
            raise TypeError("base_op must be an Op")
        try:
            in_axes = tuple(in_axes)
        except TypeError as error:
            raise TypeError("in_axes must be a sequence") from error
        for axis in in_axes:
            if axis is not None and (
                isinstance(axis, bool) or not isinstance(axis, (int, np.integer))
            ):
                raise TypeError("each in_axes entry must be an integer or None")

        if axis_size is None:
            if not any(axis is not None for axis in in_axes):
                raise ValueError("axis_size is required when no inputs are mapped")
        else:
            if isinstance(axis_size, bool) or not isinstance(axis_size, (int, np.integer)):
                raise TypeError("axis_size must be an integer or None")
            if axis_size <= 0:
                raise ValueError("axis_size must be positive")

        self.base_op: BaseOp = base_op
        self.in_axes: tuple[int | None, ...] = tuple(None if axis is None else int(axis) for axis in in_axes)
        self.axis_size: int | None = None if axis_size is None else int(axis_size)
        super().__init__()

    def get_shape(self, *parents_shapes: Shape) -> Shape:
        """Remove mapped axes, apply the base shape rule, and prepend the mapped size.

        Parameters
        ----------
        *parents_shapes
            One shape for each entry of ``in_axes``. An entry of ``None`` passes its shape unchanged
            to ``base_op``. An integer entry removes that axis before calling ``base_op.get_shape``.
            All removed axes must have the same size, equal to ``axis_size`` when it was provided.

        Returns
        -------
        Shape
            The mapped axis size prepended to the output shape of ``base_op``.

        Raises
        ------
        ValueError
            If the number of shapes differs from ``in_axes``, a mapped axis is invalid, mapped sizes
            disagree, or the remaining shapes are invalid for ``base_op``.
        """

        if len(parents_shapes) != len(self.in_axes):
            raise ValueError(
                f"VMap has {len(self.in_axes)} in_axes entries but received "
                f"{len(parents_shapes)} input shapes"
            )

        remaining_shapes, axis_size = _get_sliced_shapes(
            parents_shapes, self.in_axes, self.axis_size
        )
        dummy_shape: Shape = self.base_op.get_shape(*remaining_shapes)
        return (axis_size,) + dummy_shape

    def __repr__(self):
        out = f"VMap({repr(self.base_op)}, {repr(list(self.in_axes))}"
        if self.axis_size is not None:
            out += f", {repr(self.axis_size)}"
        out += ")"
        return out

    def __str__(self):
        """Return a string representation of the `VMap` operation.

        Just like ``VMap.__repr__`` except uses ``str`` for calling the recursive distribution.
        """
        out = f"vmap({str(self.base_op)}, {str(list(self.in_axes))}"
        if self.axis_size is not None:
            out += f", {repr(self.axis_size)}"
        out += ")"
        return out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VMap) or type(self) is not type(other):
            return False
        return (
            self.base_op == other.base_op
            and self.in_axes == other.in_axes
            and self.axis_size == other.axis_size
        )

    def __hash__(self) -> int:
        return hash((type(self), self.base_op, self.in_axes, self.axis_size))


type VMapStack[BaseOp: Op] = VMap[BaseOp] | VMap[VMapStack[BaseOp]]
"""Type-hint-only alias for one or more nested `VMap` operations around ``BaseOp``.

This alias has no runtime behavior.

For example, ``VMapStack[Normal]`` includes ``VMap[Normal]``,
``VMap[VMap[Normal]]``, and any deeper nesting, but not ``Normal`` itself.
"""


# --------------------------------------------------------------------------------------------------
# Composite
# --------------------------------------------------------------------------------------------------


class Composite[LastOp: Op](Op):
    """Compose a sequence of operations into one operation.

    Values are numbered in sequential order. The first ``num_inputs`` values are the inputs to the
    composite. Each operation appends one value, so operation ``i`` may refer only to values
    numbered from ``0`` through ``num_inputs + i - 1``. Repeated and reordered references are
    allowed.

    Parameters
    ----------
    num_inputs
        Number of inputs to the composite.
    ops
        Nonempty sequence of operations to apply. Every operation except the last must be
        deterministic.
    par_nums
        One sequence of parent value numbers for each operation.

    Raises
    ------
    TypeError
        If an argument has an invalid type.
    ValueError
        If the structure is empty or inconsistent, an intermediate operation is random, or a parent
        refers to a value that is not yet available.

    Examples
    --------
    Add two inputs and exponentiate the result.

    >>> op = Composite(2, (Add(), Exp()), ((0, 1), (2,)))
    >>> op.get_shape((), ())
    ()
    """

    @overload
    def __init__(
        self,
        num_inputs: int,
        ops: tuple[*tuple[Op, ...], LastOp],
        par_nums: Sequence[Sequence[int]],
    ) -> None: ...

    @overload
    def __init__(
        self: Composite[Op],
        num_inputs: int,
        ops: Sequence[Op],
        par_nums: Sequence[Sequence[int]],
    ) -> None: ...

    def __init__(
        self,
        num_inputs: int,
        ops: Sequence[Op],
        par_nums: Sequence[Sequence[int]],
    ) -> None:
        if isinstance(num_inputs, bool) or not isinstance(num_inputs, (int, np.integer)):
            raise TypeError("num_inputs must be an integer")
        if num_inputs < 0:
            raise ValueError("num_inputs must be non-negative")
        num_inputs = int(num_inputs)

        ops = tuple(ops)
        if not ops:
            raise ValueError("ops must contain at least one Op")
        for op in ops:
            if not isinstance(op, Op):
                raise TypeError(f"Composite argument {op!r} is not an Op")

        try:
            par_nums = tuple(tuple(my_par_nums) for my_par_nums in par_nums)
        except TypeError as error:
            raise TypeError("par_nums must be a sequence of sequences") from error
        if len(par_nums) != len(ops):
            raise ValueError(f"received {len(ops)} Ops but {len(par_nums)} par_nums entries")

        normalized_par_nums = []
        for op_number, my_par_nums in enumerate(par_nums):
            available_values = num_inputs + op_number
            normalized_my_par_nums = []
            for parent_number in my_par_nums:
                if isinstance(parent_number, bool) or not isinstance(
                    parent_number, (int, np.integer)
                ):
                    raise TypeError("each parent number must be an integer")
                parent_number = int(parent_number)
                if not 0 <= parent_number < available_values:
                    available_description = (
                        "no values are available"
                        if available_values == 0
                        else f"only values 0 through {available_values - 1} are available"
                    )
                    raise ValueError(
                        f"operation {op_number} refers to value {parent_number}, "
                        f"but {available_description}"
                    )
                normalized_my_par_nums.append(parent_number)
            normalized_par_nums.append(tuple(normalized_my_par_nums))

        for op in ops[:-1]:
            if op.random:
                raise ValueError(f"all but the last Op must be deterministic (got {op})")
        self.num_inputs: int = num_inputs
        # The tuple overload determines LastOp; other sequences use Op. Validation ensures
        # this tuple is nonempty, but normalization erases its final element's static type.
        self.ops = cast(tuple[*tuple[Op, ...], LastOp], ops)
        self.par_nums: tuple[tuple[int, ...], ...] = tuple(normalized_par_nums)
        super().__init__()

    @property
    def random(self) -> bool:
        """Equal to ``ops[-1].random``"""
        return self.ops[-1].random

    @property
    def discrete(self) -> bool | None:
        """Equal to ``ops[-1].discrete``"""
        return self.ops[-1].discrete

    def get_shape(self, *parents_shapes: Shape) -> Shape:
        """Apply the component shape rules and return the final output shape.

        Parameters
        ----------
        *parents_shapes
            Exactly ``num_inputs`` shapes. Component Ops receive these input shapes or earlier
            intermediate shapes according to ``par_nums``.

        Returns
        -------
        Shape
            Output shape produced by the final component `Op`.

        Raises
        ------
        ValueError
            If the number of supplied shapes differs from ``num_inputs`` or a component `Op`
            rejects its selected parent shapes.
        """
        if len(parents_shapes) != self.num_inputs:
            raise ValueError(
                f"Composite expects {self.num_inputs} input shapes but received "
                f"{len(parents_shapes)}"
            )

        all_shapes = list(parents_shapes)
        for my_op, my_par_nums in zip(self.ops, self.par_nums, strict=True):
            my_parents_shapes = [all_shapes[i] for i in my_par_nums]
            my_shape = my_op.get_shape(*my_parents_shapes)
            all_shapes.append(my_shape)
        return all_shapes[-1]

    def __str__(self):
        str_ops = f"[{', '.join(map(str, self.ops))}]"

        par_nums_list = list(list(a) for a in self.par_nums)
        return f"composite({self.num_inputs}, {str_ops}, {par_nums_list})"

    def __repr__(self):
        par_nums_list = list(list(a) for a in self.par_nums)
        return f"Composite({self.num_inputs}, {list(self.ops)}, {par_nums_list})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Composite) or type(self) is not type(other):
            return False
        return (
            self.num_inputs == other.num_inputs
            and self.ops == other.ops
            and self.par_nums == other.par_nums
        )

    def __hash__(self) -> int:
        return hash((type(self), self.num_inputs, self.ops, self.par_nums))


# --------------------------------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------------------------------


class Scan[O: Op](Op):
    """Repeat an operation while feeding each output into the next iteration.

    An `RV` using `Scan` has parents ordered as ``start, *inputs``. Conceptually, it represents the
    following recurrence:

    .. code-block:: python

        carry = start
        outputs = []
        for t in range(length):
            args = []
            for x, axis in zip(inputs, in_axes):
                if axis is None:
                    args.append(x)
                else:
                    args.append(take(x, t, axis))
            carry = base_op(carry, *args)
            outputs.append(carry)
        result = stack(outputs)

    Here, ``take(x, t, axis)`` selects the ``t``-th slice of ``x`` along the specified axis,
    analogous to `numpy.take <https://numpy.org/doc/stable/reference/generated/numpy.take.html>`__.
    Thus, ``in_axes`` describes only ``inputs``, not ``start``. A mapped input supplies one slice
    per iteration, while an unmapped input is reused at every iteration. The carry is always the
    first argument to ``base_op``.

    If ``base_op`` is random, each iteration is a conditional draw; otherwise, the scan is a
    deterministic recurrence. The result contains every newly produced carry, but not the initial
    carry. Because each output becomes the next input carry, ``base_op`` must return
    ``start_shape``. The result shape is therefore ``(length,) + start_shape``.

    Parameters
    ----------
    base_op
        Operation to repeat.
    length
        Positive number of repetitions.
    in_axes : Sequence[int | None]
        Axis to scan for each non-carry input, or ``None`` for an unmapped input. Every mapped axis
        must have size ``length``.

    Raises
    ------
    TypeError
        If the constructor arguments have invalid types.
    ValueError
        If ``length`` is not positive, or mapped input shapes disagree with ``length`` or
        ``in_axes``.

    Examples
    --------
    Compute a scalar cumulative sum. The carry is the first argument to
    `Add`, and the mapped input contributes one scalar per iteration.

    >>> op = Scan(Add(), length=3, in_axes=(0,))
    >>> op.get_shape((), (3,))
    (3,)

    Repeatedly multiply a vector carry by the same matrix. The carry is the first argument to
    `Matmul`, and the unmapped matrix is the second.

    >>> op = Scan(Matmul(), length=4, in_axes=(None,))
    >>> op.get_shape((3,), (3, 3))
    (4, 3)
    """

    def __init__(
        self,
        base_op: O,
        length: int,
        in_axes: Sequence[int | None],
    ):
        if not isinstance(base_op, Op):
            raise TypeError("base_op must be an Op")
        if isinstance(length, bool) or not isinstance(length, (int, np.integer)):
            raise TypeError("length must be an integer")
        if length <= 0:
            raise ValueError("length must be positive")

        try:
            in_axes = tuple(in_axes)
        except TypeError as error:
            raise TypeError("in_axes must be a sequence") from error
        for axis in in_axes:
            if axis is not None and (
                isinstance(axis, bool) or not isinstance(axis, (int, np.integer))
            ):
                raise TypeError("each in_axes entry must be an integer or None")

        self.base_op: O = base_op
        self.length: int = int(length)
        self.in_axes: tuple[int | None, ...] = tuple(None if axis is None else int(axis) for axis in in_axes)
        super().__init__()

    @property
    def random(self) -> bool:
        """Equal to ``self.base_op.random``"""

        return self.base_op.random

    @property
    def discrete(self) -> bool | None:
        """Equal to ``self.base_op.discrete``"""
        return self.base_op.discrete

    def get_shape(self, start_shape: Shape, *other_shapes: Shape) -> Shape:
        """Return the stacked shapes produced by repeatedly applying ``base_op``.

        Parameters
        ----------
        start_shape
            Shape of the initial carried value. It is the first ``base_op`` input, and the output of
            ``base_op`` must have this same shape.
        *other_shapes
            One shape for each entry of ``in_axes``. Unmapped shapes are passed unchanged to
            ``base_op``; each mapped shape loses its selected axis, whose size must equal
            ``length``.

        Returns
        -------
        Shape
            ``(length,)`` followed by the output shape of ``base_op`` for one iteration.

        Raises
        ------
        ValueError
            If the number of other shapes differs from ``in_axes``, a mapped axis is invalid or has
            the wrong length, ``base_op`` rejects the per-iteration shapes, or its output shape
            differs from the carry shape.
        """
        if len(other_shapes) != len(self.in_axes):
            raise ValueError(
                f"Scan has {len(self.in_axes)} in_axes entries but received "
                f"{len(other_shapes)} scanned input shapes"
            )

        base_input_shapes = []

        for input_number, (shape, axis) in enumerate(zip(other_shapes, self.in_axes, strict=True)):
            if axis is None:
                base_input_shapes.append(shape)
            else:
                ndim = len(shape)
                if not -ndim <= axis < ndim:
                    raise ValueError(
                        f"in_axes[{input_number}]={axis} is out of bounds for an input with "
                        f"{ndim} dimensions"
                    )
                normalized_axis = axis % ndim
                axis_length = shape[normalized_axis]
                if axis_length != self.length:
                    raise ValueError(
                        f"scanned input {input_number} has mapped axis length {axis_length}, "
                        f"expected {self.length}"
                    )
                base_input_shapes.append(shape[:normalized_axis] + shape[normalized_axis + 1 :])

        base_output_shape = self.base_op.get_shape(start_shape, *base_input_shapes)
        if base_output_shape != start_shape:
            raise ValueError(
                f"Scan base_op output shape {base_output_shape} must match carry shape "
                f"{start_shape}"
            )
        output_shape = (self.length,) + base_output_shape
        return output_shape

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Scan) or type(self) is not type(other):
            return False
        return (
            self.base_op == other.base_op
            and self.length == other.length
            and self.in_axes == other.in_axes
        )

    def __hash__(self) -> int:
        return hash((type(self), self.base_op, self.length, self.in_axes))

    def __str__(self):
        return f"scan({self.base_op}, {self.length}, {list(self.in_axes)})"

    def __repr__(self):
        return f"Scan({repr(self.base_op)}, {self.length}, {list(self.in_axes)})"


# --------------------------------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------------------------------


class Index(Op):
    """Select a scalar entry from an array.

    An `Index` `RV` has parents ``array, *indices``, with exactly one scalar integer-valued index
    for each dimension of ``array``. The output shape is always ``()``.

    `Index` itself does not represent slices or array indexing. These can be constructed using
    `VMap` around `Index`. Index parents may be random variables, although a backend need not
    support inference with random indices.

    Examples
    --------
    >>> op = Index()
    >>> op.get_shape((5, 5, 5), (), (), ())
    ()
    """

    random: bool = False
    """False"""

    def get_shape(self, var_shape: Shape, *indices_shapes: Shape) -> Shape:
        """Validate scalar indices and return the scalar output shape.

        Parameters
        ----------
        var_shape
            Shape of the array being indexed.
        *indices_shapes
            Exactly one scalar shape ``()`` for each dimension of ``var_shape``.

        Returns
        -------
        Shape
            Always ``()``.

        Raises
        ------
        ValueError
            If the number of index shapes differs from the rank of ``var_shape``, or any index
            is nonscalar.
        """
        num_indexed = len(indices_shapes)
        num_dims = len(var_shape)
        if num_indexed != num_dims:
            raise ValueError(f"Indexed RV with {num_dims} dims with {num_indexed} indices.")

        if any(shape != () for shape in indices_shapes):
            raise ValueError("Index requires scalar indices")
        return ()


# --------------------------------------------------------------------------------------------------
# Bijectors
# --------------------------------------------------------------------------------------------------


class Bijector:
    """Describe an invertible transformation using deterministic IR operations.

    A `Bijector` is not an `Op` and cannot itself be attached to an `RV`. It is an immutable
    descriptor containing three deterministic operations. `Bijected` embeds that descriptor in a
    random `Op` so transformed distributions remain explicit in the IR and independent of any
    backend.

    For a transformation ``y = T(x, *params)``, ``forward`` computes ``y`` and ``inverse`` computes
    ``x``. ``log_det_jac`` receives both ``x`` and ``y``, followed by the same parameters, and
    computes the scalar forward log absolute Jacobian determinant ``log |det ∇T(x)|``. Supplying
    both values lets the operation use whichever representation is more convenient.

    Optional ``sign`` is a deterministic Op returning scalar ``+1`` for increasing or ``-1``
    for decreasing transformations. It receives every bijector parameter in order, without the
    forward input or output. By supplying it, the author guarantees a continuous, strictly
    monotone scalar bijection for every valid parameter setting.

    In general if ``Y=T(X)`` then

    ``P_X(x) = P_Y(T(x)) × |det ∇T(x)|``

    and

    ``P_Y(y) = P_X(T⁻¹(y)) × |det ∇T⁻¹(y)|``.

    That is, ``P_Y`` is the pushforward of ``P_X`` and ``P_X`` is the pullback of ``P_Y``.

    Thus a transformed distribution samples ``x`` from its base distribution and applies
    ``forward``. Its log density applies ``inverse`` and subtracts the forward log determinant.

    Parameters
    ----------
    forward
        Deterministic operation implementing ``T(x, *params)``.
    inverse
        Deterministic operation implementing ``T⁻¹(y, *params)``.
    log_det_jac
        Deterministic operation implementing ``log |det ∇T(x)|`` from ``(x, y, *params)``.
    n_biject_params
        Number of parameter RVs accepted by every supplied operation.
    sign
        Optional deterministic Op returning scalar ``-1`` or ``+1`` from all ordered bijector
        parameters. Describes scalar applications only.

    Examples
    --------
    An implementation of the ``exp`` transformation.

    >>> forward = Exp()
    >>> inverse = Log()
    >>> log_jac_det = Composite(
    ...     2,
    ...     (Identity(),),
    ...     [[0]])
    >>> bijector = Bijector(forward, inverse, log_jac_det)
    >>> isinstance(bijector, Op)
    False
    >>> print(bijector)
    bijector(exp, log, composite(2, [identity], [[0]]))

    The exponential bijection is increasing and has no parameters:

    >>> increasing = Bijector(Exp(), Log(), log_jac_det, sign=Constant(1))
    >>> increasing.sign
    Constant(1)
    >>> increasing.get_shape(())
    ()

    Scaled logit maps ``(lo, hi)`` to the reals, for finite parameters ``lo < hi``. Its forward
    function is ``logit((x - lo) / (hi - lo))`` and its inverse is
    ``lo + (hi - lo) * inv_logit(y)``:

    >>> forward = Composite(
    ...     3, (Sub(), Sub(), Div(), Logit()),
    ...     ((0, 1), (2, 1), (3, 4), (5,)))
    >>> inverse = Composite(
    ...     3, (Sub(), InvLogit(), Mul(), Add()),
    ...     ((2, 1), (0,), (3, 4), (1, 5)))

    The forward log-Jacobian is ``log((hi - lo) / ((x - lo) * (hi - x)))``. Its inputs are
    ``(x, y, lo, hi)``:

    >>> log_jac_det = Composite(
    ...     4, (Sub(), Sub(), Sub(), Mul(), Div(), Log()),
    ...     ((3, 2), (0, 2), (3, 0), (5, 6), (4, 7), (8,)))

    The sign Op receives ``(lo, hi)``, even though the sign is constant:

    >>> sign = Composite(2, (Constant(1),), ((),))
    >>> scaled_logit = Bijector(forward, inverse, log_jac_det, 2, sign=sign)
    >>> scaled_logit.get_shape((), (), ())
    ()
    >>> scaled_logit.sign
    Composite(2, [Constant(1)], [[]])

    Raises
    ------
    TypeError
        If an operation or parameter count has an invalid type.
    ValueError
        If any operation is random or ``n_biject_params`` is negative. Values produced by the
        sign Op are an author guarantee, not evaluated during construction.
    """

    _frozen = False

    def __init__(
        self,
        forward: Op,
        inverse: Op,
        log_det_jac: Op,
        n_biject_params: int = 0,
        *,
        sign: Op | None = None,
    ):
        for name, op in (("forward", forward), ("inverse", inverse), ("log_det_jac", log_det_jac)):
            if not isinstance(op, Op):
                raise TypeError(f"{name} must be an Op")
            if op.random:
                raise ValueError(f"{name} op cannot be random")

        if sign is not None:
            if not isinstance(sign, Op):
                raise TypeError("sign must be an Op or None")
            if sign.random:
                raise ValueError("sign Op must be deterministic")

        if isinstance(n_biject_params, bool) or not isinstance(n_biject_params, int):
            raise TypeError("n_biject_params must be an integer")
        if n_biject_params < 0:
            raise ValueError("n_biject_params must be non-negative")

        self.forward: Op = forward
        self.inverse: Op = inverse
        self.log_det_jac: Op = log_det_jac
        self.n_biject_params: int = n_biject_params
        self.sign: Op | None = sign
        self._frozen = True

    def get_shape(self, x_shape: Shape, *param_shapes: Shape) -> Shape:
        """Validate operation shapes and return the transformed shape."""
        if len(param_shapes) != self.n_biject_params:
            raise ValueError(
                f"Number of bijection parameters {len(param_shapes)} not as expected "
                f"({self.n_biject_params})"
            )

        y_shape = self.forward.get_shape(x_shape, *param_shapes)
        x_shape_pred = self.inverse.get_shape(y_shape, *param_shapes)
        if x_shape_pred != x_shape:
            raise ValueError(
                f"x_shape {x_shape} did not match predicted x_shape {x_shape_pred}, indicating "
                "forward and inverse ops have incompatible shapes"
            )

        log_det_shape = self.log_det_jac.get_shape(x_shape, y_shape, *param_shapes)
        if log_det_shape != ():
            raise ValueError(f"log_det_jac must return a scalar, got shape {log_det_shape}")
        if self.sign is not None:
            sign_shape = self.sign.get_shape(*param_shapes)
            if sign_shape != ():
                raise ValueError(f"sign must return a scalar, got shape {sign_shape}")
        return y_shape

    def __eq__(self, other: object) -> bool:
        """Return whether two Bijectors have the same concrete type and structural state."""
        if not isinstance(other, Bijector) or type(self) is not type(other):
            return False
        return (
            self.forward == other.forward
            and self.inverse == other.inverse
            and self.log_det_jac == other.log_det_jac
            and self.n_biject_params == other.n_biject_params
            and self.sign == other.sign
        )

    def __hash__(self) -> int:
        return hash(
            (
                type(self),
                self.forward,
                self.inverse,
                self.log_det_jac,
                self.n_biject_params,
                self.sign,
            )
        )

    def __setattr__(self, key, value):
        if self._frozen:
            raise TypeError("Bijectors are immutable after init.")
        object.__setattr__(self, key, value)

    def __delattr__(self, key):
        if self._frozen:
            raise TypeError("Bijectors are immutable after init.")
        object.__delattr__(self, key)

    def __repr__(self):
        """Return a structural Python representation of the bijector.

        This follows the same constructor-shaped convention as `Op.__repr__` and recursively
        uses ``repr`` for its operations.
        """
        args = [repr(self.forward), repr(self.inverse), repr(self.log_det_jac)]
        if self.n_biject_params:
            args.append(str(self.n_biject_params))
        if self.sign is not None:
            args.append(f"sign={self.sign!r}")
        return f"Bijector({', '.join(args)})"

    def __str__(self):
        """Return a compact, human-readable representation of the bijector.

        This follows the convention of `Op.__str__` and recursively uses ``str`` for its
        operations.
        """
        args = [str(self.forward), str(self.inverse), str(self.log_det_jac)]
        if self.n_biject_params:
            args.append(str(self.n_biject_params))
        if self.sign is not None:
            args.append(f"sign={self.sign}")
        return f"bijector({', '.join(args)})"


# --------------------------------------------------------------------------------------------------
# Bijected distributions
# --------------------------------------------------------------------------------------------------


class Bijected[O: Op, B: Bijector](Op):
    """A continuous distribution obtained by applying a bijector to another distribution.

    `Bijected` is restricted to continuous base distributions because its density uses the
    Jacobian change-of-variables formula. A transformation of a discrete distribution instead
    preserves probability masses and must not apply a Jacobian correction.

    The parent order is part of the IR contract. The first ``bijector.n_biject_params`` parents are
    parameters of the bijection; all remaining parents are parameters of ``base_op``.

    Given an ordered parent sequence ``parents``, the transformed random variable is equal in
    distribution to:

    .. code-block:: python

        n = op.bijector.n_biject_params
        x = RV(op.base_op, *parents[n:])
        y = RV(op.bijector.forward, x, *parents[:n])

    In the actual IR, ``RV(op, *parents)`` represents this distribution with a single random node
    rather than explicitly constructing ``x`` and the deterministic ``y``.

    Examples
    --------
    Create a new op that's equivalent to `Lognormal`:

    >>> exp_bijector = Bijector(
    ...     Exp(),
    ...     Log(),
    ...     Composite(2, (Identity(),), [[0]]),
    ... )
    >>> op = Bijected(Normal(), exp_bijector)
    >>> print(op)
    bijected(normal, bijector(exp, log, composite(2, [identity], [[0]])))
    >>> op.get_shape((),())
    ()

    Use that new op to create a random variable:

    >>> y = RV(op, RV(Constant(0)), RV(Constant(1)))
    >>> y.shape
    ()

    A parameterized bijection receives its parameters before those of the base distribution. This
    example describes ``Y = X + offset``.

    >>> shift = Bijector(
    ...     Add(),
    ...     Sub(),
    ...     Composite(3, (Constant(0),), [[]]),
    ...     n_biject_params=1,
    ... )
    >>> shifted_normal = Bijected(Normal(), shift)
    >>> shifted_normal.get_shape((), (), ())
    ()

    Parameters
    ----------
    base_op
        Random, continuous `Op` describing the base distribution.
    bijector
        Bijection applied to samples from the base distribution.

    Raises
    ------
    TypeError
        If ``base_op`` is not an `Op` or ``bijector`` is not a `Bijector`.
    ValueError
        If ``base_op`` is deterministic or discrete.
    """

    random: bool = True
    """True"""
    discrete: bool = False
    """False"""

    def __init__(
        self,
        base_op: O,
        bijector: B,
    ):
        if not isinstance(base_op, Op):
            raise TypeError("base_op must be an Op")
        if not isinstance(bijector, Bijector):
            raise TypeError("bijector must be a Bijector")
        if not base_op.random:
            raise ValueError("base_op must be random")
        if base_op.discrete:
            raise ValueError("base_op must be continuous")

        self.base_op: O = base_op
        self.bijector: B = bijector
        super().__init__()

    def get_shape(self, *args: Shape) -> Shape:
        """Apply the base-distribution and bijector shape rules in parent order.

        Parameters
        ----------
        *args
            First, exactly ``bijector.n_biject_params`` shapes for the bijector parameters; then the
            parent shapes expected by ``base_op``.

        Returns
        -------
        Shape
            Shape returned by the bijector for a value having the output shape of ``base_op``.

        Raises
        ------
        TypeError
            If the total number of shapes is incompatible with ``base_op``.
        ValueError
            If ``base_op`` or the bijector rejects the supplied shapes.
        """
        n = self.bijector.n_biject_params
        biject_args_shapes = args[:n]
        dist_shapes = args[n:]

        original_shape = self.base_op.get_shape(*dist_shapes)
        new_shape = self.bijector.get_shape(original_shape, *biject_args_shapes)
        return new_shape

    def __repr__(self):
        return f"Bijected({repr(self.base_op)}, {repr(self.bijector)})"

    def __str__(self):
        """Return a compact representation of the transformed distribution."""
        return f"bijected({str(self.base_op)}, {str(self.bijector)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bijected) or type(self) is not type(other):
            return False
        return self.base_op == other.base_op and self.bijector == other.bijector

    def __hash__(self) -> int:
        return hash((type(self), self.base_op, self.bijector))


# --------------------------------------------------------------------------------------------------
# Truncation
# --------------------------------------------------------------------------------------------------


class Truncated[BaseOp: Op](Op):
    """Restrict a continuous scalar distribution to an interval.

    Parents are the lower bound if ``lower``, then the upper bound if ``upper``, then
    the parameters of ``base_op`` in their original order. Bounds and the base output must be
    scalar. The result is the normalized distribution restricted to the supplied bounds.
    With neither bound, it is the base distribution unchanged.

    Each supplied bound must be finite and strictly inside the base distribution's support;
    when both are supplied, the lower bound must be less than the upper bound. Backends must
    enforce these value-dependent requirements whenever the distribution is evaluated, including
    sampling, density, CDF, and quantile calculations. The IR checks only structural requirements
    and does not require CDF or inverse-CDF capability at construction.

    Parameters
    ----------
    base_op
        Random, continuous `Op` whose output must be scalar for the supplied parameter shapes.
    lower
        Whether a lower-bound parent is supplied. Default ``True``.
    upper
        Whether an upper-bound parent is supplied. Default ``True``.

    Raises
    ------
    TypeError
        If ``base_op`` is not an `Op` or either bound flag is not a boolean.
    ValueError
        If ``base_op`` is deterministic or discrete.

    Examples
    --------
    A truncated Normal takes lower bound, upper bound, location, and scale:

    >>> op = Truncated(Normal())
    >>> op.get_shape((), (), (), ())
    ()
    >>> op.random, op.discrete
    (True, False)

    One-sided truncations take just one bound before the base parameters:

    >>> Truncated(Normal(), upper=False).get_shape((), (), ())
    ()
    >>> Truncated(Normal(), lower=False).get_shape((), (), ())
    ()

    With neither bound, only the base parameters are supplied:

    >>> Truncated(Normal(), lower=False, upper=False).get_shape((), ())
    ()

    Truncation can be nested. Each level supplies its own bounds:

    >>> Truncated(op).get_shape((), (), (), (), (), ())
    ()

    Vectorize outside truncation to produce independent draws with shared bounds and scale:

    >>> VMap(op, (None, None, 0, None)).get_shape((), (), (3,), ())
    (3,)
    """

    random: bool = True
    """True"""
    discrete: bool = False
    """False"""

    def __init__(
        self, base_op: BaseOp, *, lower: bool = True, upper: bool = True
    ) -> None:
        if not isinstance(base_op, Op):
            raise TypeError("base_op must be an Op")
        if not base_op.random:
            raise ValueError("base_op must be random")
        if base_op.discrete:
            raise ValueError("base_op must be continuous")
        if not isinstance(lower, bool) or not isinstance(upper, bool):
            raise TypeError("lower and upper must be booleans")
        self.base_op: BaseOp = base_op
        self.lower: bool = lower
        self.upper: bool = upper
        super().__init__()

    def get_shape(self, *parent_shapes: Shape) -> Shape:
        """Validate scalar bounds and base output, and return scalar shape.

        Parameters
        ----------
        *parent_shapes
            Lower-bound shape if ``lower``, then upper-bound shape if ``upper``;
            each must be ``()``. Remaining shapes are passed to ``base_op.get_shape`` in
            their original order.

        Raises
        ------
        TypeError
            If required shapes are missing or the base Op rejects the number of parameters.
        ValueError
            If the bounds or base output are not scalar, or the base Op rejects its parameters.
        """
        num_bounds = int(self.lower) + int(self.upper)
        if len(parent_shapes) < num_bounds:
            raise TypeError(f"expected {num_bounds} bound shapes before base parameter shapes")
        if any(shape != () for shape in parent_shapes[:num_bounds]):
            raise ValueError("truncation bounds must be scalar")
        if self.base_op.get_shape(*parent_shapes[num_bounds:]) != ():
            raise ValueError("base_op must have scalar output")
        return ()

    def __repr__(self) -> str:
        args = [repr(self.base_op)]
        if not self.lower:
            args.append("lower=False")
        if not self.upper:
            args.append("upper=False")
        return f"Truncated({', '.join(args)})"

    def __str__(self) -> str:
        args = [str(self.base_op)]
        if not self.lower:
            args.append("lower=False")
        if not self.upper:
            args.append("upper=False")
        return f"truncated({', '.join(args)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Truncated) or type(self) is not type(other):
            return False
        return (
            self.base_op == other.base_op
            and self.lower == other.lower
            and self.upper == other.upper
        )

    def __hash__(self) -> int:
        return hash((type(self), self.base_op, self.lower, self.upper))


# --------------------------------------------------------------------------------------------------
# Log probability
# --------------------------------------------------------------------------------------------------


class LogProb[BaseOp: Op](Op):
    """Evaluate a distribution's log density or log probability mass.

    The first parent is the evaluation value; all remaining parents are the parameters of
    ``base_op``, in its original parent order. The evaluation value must have exactly the shape
    produced by ``base_op``. The output is always scalar: for an array-valued distribution it is
    the joint log density or mass of the whole array, not an array of componentwise values.

    This is a deterministic operation, even when its parents are random RVs. It evaluates the
    local distribution with the supplied parameter values fixed; it does not integrate over
    their upstream distributions. Constructing this Op neither samples nor evaluates anything.
    Numerical evaluation and support for particular base Ops are left to backends.

    Parameters
    ----------
    base_op
        Random `Op` describing a discrete or continuous distribution.

    Raises
    ------
    TypeError
        If ``base_op`` is not an `Op`.
    ValueError
        If ``base_op`` is deterministic.

    Examples
    --------
    A Normal log density takes a scalar value, location, and scale:

    >>> op = LogProb(Normal())
    >>> op.get_shape((), (), ())
    ()
    >>> op.random
    False

    Discrete distributions are supported as well:

    >>> LogProb(Bernoulli()).get_shape((), ())
    ()

    A vector-valued distribution still has a scalar log density:

    >>> LogProb(VMap(Normal(), (0, None))).get_shape((3,), (3,), ())
    ()
    """

    random: bool = False
    """False"""

    def __init__(self, base_op: BaseOp) -> None:
        if not isinstance(base_op, Op):
            raise TypeError("base_op must be an Op")
        if not base_op.random:
            raise ValueError("base_op must be random")
        self.base_op: BaseOp = base_op
        super().__init__()

    def get_shape(self, value_shape: Shape, *parent_shapes: Shape) -> Shape:
        """Validate the evaluation value and distribution parameters, and return scalar shape.

        Parameters
        ----------
        value_shape
            Shape of the value at which to evaluate the log density or mass.
        *parent_shapes
            Parameter shapes passed to ``base_op.get_shape`` in their original order.

        Raises
        ------
        TypeError
            If a required shape is missing or the base Op rejects the number of parameter shapes.
        ValueError
            If the base Op rejects the parameter shapes or its output shape differs from
            ``value_shape``.
        """
        expected_shape = self.base_op.get_shape(*parent_shapes)
        if value_shape != expected_shape:
            raise ValueError(
                f"value shape {value_shape} does not match base_op output shape {expected_shape}"
            )
        return ()

    def __repr__(self) -> str:
        return f"LogProb({self.base_op!r})"

    def __str__(self) -> str:
        return f"log_prob({self.base_op})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogProb) or type(self) is not type(other):
            return False
        return self.base_op == other.base_op

    def __hash__(self) -> int:
        return hash((type(self), self.base_op))


# --------------------------------------------------------------------------------------------------
# Cumulative distribution functions
# --------------------------------------------------------------------------------------------------


class CDF[BaseOp: Op](Op):
    """Evaluate a distribution's cumulative distribution function.

    The first parent is the evaluation value; all remaining parents are the parameters of
    ``base_op``, in its original parent order. The value must have exactly the shape produced by
    ``base_op``. The output is always scalar: the probability that every component of a draw is
    less than or equal to the corresponding component of the value. This is a joint CDF, not
    an array of marginal CDFs. For discrete distributions, equality includes the mass at the
    value; the evaluation value itself need not be an integer.

    This is a deterministic operation, even when its parents are random RVs. It evaluates the
    local distribution with the supplied parameter values fixed; it does not integrate over
    their upstream distributions. Constructing this Op neither samples nor evaluates anything.
    Numerical evaluation and support for particular base Ops are left to backends.

    Parameters
    ----------
    base_op
        Random `Op` describing a discrete or continuous distribution.

    Raises
    ------
    TypeError
        If ``base_op`` is not an `Op`.
    ValueError
        If ``base_op`` is deterministic.

    Examples
    --------
    A Normal CDF takes a scalar value, location, and scale:

    >>> op = CDF(Normal())
    >>> op.get_shape((), (), ())
    ()
    >>> op.random
    False

    Discrete distributions have CDFs as well:

    >>> CDF(Poisson()).get_shape((), ())
    ()

    The joint CDF of independent draws is scalar, whereas mapping a scalar CDF gives a vector:

    >>> CDF(VMap(Normal(), (0, None))).get_shape((3,), (3,), ())
    ()
    >>> VMap(CDF(Normal()), (0, 0, None)).get_shape((3,), (3,), ())
    (3,)
    """

    random: bool = False
    """False"""

    def __init__(self, base_op: BaseOp) -> None:
        if not isinstance(base_op, Op):
            raise TypeError("base_op must be an Op")
        if not base_op.random:
            raise ValueError("base_op must be random")
        self.base_op: BaseOp = base_op
        super().__init__()

    def get_shape(self, value_shape: Shape, *parent_shapes: Shape) -> Shape:
        """Validate the evaluation value and distribution parameters, and return scalar shape.

        Parameters
        ----------
        value_shape
            Shape of the value at which to evaluate the CDF.
        *parent_shapes
            Parameter shapes passed to ``base_op.get_shape`` in their original order.

        Raises
        ------
        TypeError
            If a required shape is missing or the base Op rejects the number of parameter shapes.
        ValueError
            If the base Op rejects the parameter shapes or its output shape differs from
            ``value_shape``.
        """
        expected_shape = self.base_op.get_shape(*parent_shapes)
        if value_shape != expected_shape:
            raise ValueError(
                f"value shape {value_shape} does not match base_op output shape {expected_shape}"
            )
        return ()

    def __repr__(self) -> str:
        return f"CDF({self.base_op!r})"

    def __str__(self) -> str:
        return f"cdf({self.base_op})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CDF) or type(self) is not type(other):
            return False
        return self.base_op == other.base_op

    def __hash__(self) -> int:
        return hash((type(self), self.base_op))


# --------------------------------------------------------------------------------------------------
# Quantiles
# --------------------------------------------------------------------------------------------------


class Quantile[BaseOp: Op](Op):
    """Evaluate a scalar distribution's quantile function (inverse CDF).

    The first parent is a scalar probability; all remaining parents are the parameters of
    ``base_op``, in its original parent order. The base distribution and the result must both
    be scalar. For a probability strictly between zero and one, the quantile is the smallest
    value at which the CDF reaches that probability. This definition also applies to discrete
    distributions. At zero and one, use the corresponding one-sided limits of the quantile
    function, which may be infinite.

    This is a deterministic operation, even when its parents are random RVs. It evaluates the
    local distribution with the supplied parameter values fixed; it does not integrate over
    their upstream distributions. Constructing this Op neither samples nor evaluates anything.
    Numerical evaluation, validation that the probability lies in ``[0, 1]``, and support for
    particular base Ops are left to backends.

    Parameters
    ----------
    base_op
        Random `Op` describing a discrete or continuous scalar distribution.

    Raises
    ------
    TypeError
        If ``base_op`` is not an `Op`.
    ValueError
        If ``base_op`` is deterministic. Scalar applicability is checked by ``get_shape``.

    Examples
    --------
    A Normal quantile takes a scalar probability, location, and scale:

    >>> op = Quantile(Normal())
    >>> op.get_shape((), (), ())
    ()
    >>> op.random
    False

    Discrete distributions have quantiles as well:

    >>> Quantile(Poisson()).get_shape((), ())
    ()

    The distribution may itself be wrapped, as long as its output is scalar:

    >>> Quantile(Truncated(Normal())).get_shape((), (), (), (), ())
    ()

    Map a scalar quantile to evaluate several probabilities or parameter settings:

    >>> VMap(Quantile(Normal()), (0, 0, None)).get_shape((3,), (3,), ())
    (3,)

    There is no joint quantile for an array-valued distribution:

    >>> Quantile(VMap(Normal(), (0, None))).get_shape((), (3,), ())
    Traceback (most recent call last):
        ...
    ValueError: base_op must have scalar output
    """

    random: bool = False
    """False"""

    def __init__(self, base_op: BaseOp) -> None:
        if not isinstance(base_op, Op):
            raise TypeError("base_op must be an Op")
        if not base_op.random:
            raise ValueError("base_op must be random")
        self.base_op: BaseOp = base_op
        super().__init__()

    def get_shape(self, probability_shape: Shape, *parent_shapes: Shape) -> Shape:
        """Validate the scalar probability and distribution output, and return scalar shape.

        Parameters
        ----------
        probability_shape
            Shape of the probability at which to evaluate the quantile.
        *parent_shapes
            Parameter shapes passed to ``base_op.get_shape`` in their original order.

        Raises
        ------
        TypeError
            If a required shape is missing or the base Op rejects the number of parameter shapes.
        ValueError
            If the probability or base output is not scalar, or the base Op rejects its
            parameter shapes.
        """
        if probability_shape != ():
            raise ValueError("probability must be scalar")
        if self.base_op.get_shape(*parent_shapes) != ():
            raise ValueError("base_op must have scalar output")
        return ()

    def __repr__(self) -> str:
        return f"Quantile({self.base_op!r})"

    def __str__(self) -> str:
        return f"quantile({self.base_op})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantile) or type(self) is not type(other):
            return False
        return self.base_op == other.base_op

    def __hash__(self) -> int:
        return hash((type(self), self.base_op))


# --------------------------------------------------------------------------------------------------
# Vector-Jacobian products
# --------------------------------------------------------------------------------------------------


class VJP[BaseOp: Op](Op):
    """Apply a vector-Jacobian product to one input of a deterministic Op.

    The first parent is a cotangent with exactly the shape of the output of ``base_op``;
    all remaining parents are its inputs in their original order. The result has the shape
    of base input ``argnum``.
    For real-valued inputs and output, this is the gradient of the sum of the elementwise
    product of the output and cotangent, with respect to that input. All other inputs and
    the cotangent are held fixed for this differentiation.

    This is a deterministic operation, even when its parents are random RVs. It differentiates
    the local function described by ``base_op``, not its parents' upstream computations.
    Constructing this Op neither evaluates nor differentiates anything. Numerical evaluation,
    supported dtypes, and differentiability of particular base Ops are left to backends.

    Parameters
    ----------
    base_op
        Deterministic `Op` to differentiate, including deterministic container or wrapper Ops.
    argnum
        Zero-based, nonnegative index of the input to differentiate with respect to.
        The cotangent is not counted as an input of ``base_op``.

    Raises
    ------
    TypeError
        If ``base_op`` is not an `Op`, or ``argnum`` is not an integer (excluding booleans).
    ValueError
        If ``base_op`` is random or ``argnum`` is negative.

    Examples
    --------
    A scalar multiplication takes a scalar cotangent followed by two inputs:

    >>> op = VJP(Mul(), 0)
    >>> op.get_shape((), (), ())
    ()
    >>> op.random
    False

    A matrix product returns a gradient shaped like the selected matrix:

    >>> VJP(Matmul(), 0).get_shape((2, 4), (2, 3), (3, 4))
    (2, 3)
    >>> VJP(Matmul(), 1).get_shape((2, 4), (2, 3), (3, 4))
    (3, 4)

    A log density is deterministic, so it can be differentiated as well:

    >>> VJP(LogProb(Normal()), 1).get_shape((), (), (), ())
    ()
    """

    random: bool = False
    """False"""

    def __init__(self, base_op: BaseOp, argnum: int) -> None:
        if not isinstance(base_op, Op):
            raise TypeError("base_op must be an Op")
        if base_op.random:
            raise ValueError("base_op must be deterministic")
        if isinstance(argnum, bool) or not isinstance(argnum, (int, np.integer)):
            raise TypeError("argnum must be an integer")
        if argnum < 0:
            raise ValueError("argnum must be nonnegative")
        self.base_op: BaseOp = base_op
        self.argnum: int = int(argnum)
        super().__init__()

    def get_shape(self, cotangent_shape: Shape, *input_shapes: Shape) -> Shape:
        """Validate the base inputs and cotangent, and return the selected input's shape.

        Parameters
        ----------
        cotangent_shape
            Shape of the cotangent, which must match the base Op's output shape.
        *input_shapes
            Input shapes passed to ``base_op.get_shape`` in their original order.

        Raises
        ------
        TypeError
            If the cotangent is missing or the base Op rejects the number of input shapes.
        ValueError
            If ``argnum`` is out of range, the base Op rejects its input shapes, or the
            cotangent shape differs from the base Op's output shape.
        """
        output_shape = self.base_op.get_shape(*input_shapes)
        if self.argnum >= len(input_shapes):
            raise ValueError("argnum must index an input of base_op")
        if cotangent_shape != output_shape:
            raise ValueError(
                f"cotangent shape {cotangent_shape} does not match base_op output shape {output_shape}"
            )
        return input_shapes[self.argnum]

    def __repr__(self) -> str:
        return f"VJP({self.base_op!r}, {self.argnum})"

    def __str__(self) -> str:
        return f"vjp({self.base_op}, {self.argnum})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VJP) or type(self) is not type(other):
            return False
        return self.base_op == other.base_op and self.argnum == other.argnum

    def __hash__(self) -> int:
        return hash((type(self), self.base_op, self.argnum))


# --------------------------------------------------------------------------------------------------
# RVs
# --------------------------------------------------------------------------------------------------


class RVInterface:
    """Provide typed targets for RV syntax, with rejecting defaults.

    RVs forward standard syntax directly to these methods: for example, ``x + y`` calls
    ``x.interface.add(x, y)``. Subclasses implement whichever operations they support;
    unsupported operations raise ``NotImplementedError``. Additional public attributes
    are provided explicitly through ``get_attribute``, which otherwise raises ``AttributeError``.

    Notes
    -----
    ``x.T`` calls ``x.interface.transpose(x)``. Bracket indexing calls
    ``x.interface.getitem(x, *indices)``; each interface chooses its indexing semantics.
    For example, `pangolin.interface.Interface` uses orthogonal indexing.

    Additional public attributes call ``x.interface.get_attribute(x, name)``. Interface methods
    do not automatically become RV attributes. Core graph attributes such as ``parents``, ``op``,
    and ``shape`` always belong to RV, as do identity equality and immutability. Unknown private
    and dunder attributes are never automatically forwarded.

    Standard RV operations, including ``x.T``, have return type ``RV``. Additional extensible
    attributes are statically typed as ``Any``; explicit interface calls retain their method
    signatures. Implementing only the supported operations is enough to define an interface.

    Examples
    --------
    Pangolin's standard interface exposes ``sum`` and ``vindex`` through ``get_attribute``,
    but does not expose arbitrary interface methods as RV attributes:

    >>> from pangolin import interface as pi
    >>> x = pi.constant([[1, 2], [3, 4]])
    >>> x.sum(axis=0).shape
    (2,)
    >>> x.T.shape
    (2, 2)
    >>> hasattr(x, "overload")
    False
    """

    def neg(self, x: RV, /) -> RV:
        raise NotImplementedError("RV interface does not implement 'neg'")

    def abs(self, x: RV, /) -> RV:
        raise NotImplementedError("RV interface does not implement 'abs'")

    def add(self, x: Any, y: Any, /) -> RV:
        raise NotImplementedError("RV interface does not implement 'add'")

    def sub(self, x: Any, y: Any, /) -> RV:
        raise NotImplementedError("RV interface does not implement 'sub'")

    def mul(self, x: Any, y: Any, /) -> RV:
        raise NotImplementedError("RV interface does not implement 'mul'")

    def div(self, x: Any, y: Any, /) -> RV:
        raise NotImplementedError("RV interface does not implement 'div'")

    def pow(self, x: Any, y: Any, /) -> RV:
        raise NotImplementedError("RV interface does not implement 'pow'")

    def matmul(self, x: Any, y: Any, /) -> RV:
        raise NotImplementedError("RV interface does not implement 'matmul'")

    def transpose(self, x: RV, /) -> RV:
        raise NotImplementedError("RV interface does not implement 'transpose'")

    def getitem(self, x: RV, /, *indices: Any) -> RV:
        raise NotImplementedError("RV interface does not implement 'getitem'")

    def get_attribute(self, rv: RV, name: str, /) -> Any:
        """Provide an additional RV attribute, or raise if it is unsupported."""
        raise AttributeError(f"RV interface does not expose {name!r}")

    def __repr__(self) -> str:
        return "NO_INTERFACE" if self is NO_INTERFACE else object.__repr__(self)


NO_INTERFACE: Final[RVInterface] = RVInterface()
"""The default interface, which rejects Python conveniences."""


@final
class RV(dag.Node):
    """An `RV` is essentially just an `Op` and a tuple of parent `RV` objects.

    Each construction creates a distinct graph node. Distinct `RV` nodes with random `Op` objects
    represent separate conditionally independent draws given their parents; reusing an `RV` object
    reuses the same draw.

    The only purpose of an RV's ``interface`` is to delegate Python syntax. For example,
    ``x + y`` calls ``x.interface.add(x, y)``, and ``x.T`` calls
    ``x.interface.transpose(x)``. Unsupported standard operations raise ``NotImplementedError``.
    Additional public attributes call ``x.interface.get_attribute(x, name)``, which raises
    ``AttributeError`` unless the interface explicitly supports that name.
    RV-owned attributes and graph behavior always take precedence; unknown private and
    dunder attributes are not forwarded. The interface has no effect on the probabilistic
    interpretation of the DAG, and backend code should never need to inspect it.

    Parameters
    ----------
    op
        The `Op` defining the `RV`.
    *parents : RV
        The parents of the `RV`.
    interface
        Immutable target for Python-syntax delegation. The default `NO_INTERFACE` rejects all such
        syntax.

    Examples
    --------
    >>> constant_op = Constant(3)
    >>> x = RV(constant_op)
    >>> x
    RV(Constant(3))
    >>> x.op
    Constant(3)
    >>> x.parents
    ()
    >>> y = RV(constant_op)
    >>> y
    RV(Constant(3))
    >>> normal_op = Normal()
    >>> z = RV(normal_op, x, y)
    >>> z
    RV(Normal(), RV(Constant(3)), RV(Constant(3)))
    >>> z.parents[0] == x
    True
    >>> z.parents[1] == y
    True

    """

    _frozen = False
    _n = 1

    __array_priority__ = 1000

    def __init__(
        self,
        op: Op,
        *parents: RV,
        interface: RVInterface = NO_INTERFACE,
    ):
        self._n = RV._n
        RV._n += 1
        dag.Node.__init__(self, *parents)
        parents_shapes = tuple(p.shape for p in parents)
        self._shape: Shape = op.get_shape(*parents_shapes)

        self.op: Op = op
        self.interface: RVInterface = interface
        self._frozen = True

    def __getattr__(self, name: str) -> Any:
        """Delegate missing public attributes to the interface extension hook."""
        if name.startswith("_") or any(name in cls.__dict__ for cls in type(self).__mro__):
            raise AttributeError(f"RV has no attribute {name!r}")
        interface: RVInterface = object.__getattribute__(self, "interface")
        return interface.get_attribute(self, name)

    def __bool__(self) -> bool:
        raise TypeError("RVs do not have a truth value")

    def __neg__(self) -> RV:
        return self.interface.neg(self)

    def __abs__(self) -> RV:
        return self.interface.abs(self)

    def __add__(self, other: Any) -> RV:
        return self.interface.add(self, other)

    def __radd__(self, other: Any) -> RV:
        return self.interface.add(other, self)

    def __sub__(self, other: Any) -> RV:
        return self.interface.sub(self, other)

    def __rsub__(self, other: Any) -> RV:
        return self.interface.sub(other, self)

    def __mul__(self, other: Any) -> RV:
        return self.interface.mul(self, other)

    def __rmul__(self, other: Any) -> RV:
        return self.interface.mul(other, self)

    def __truediv__(self, other: Any) -> RV:
        return self.interface.div(self, other)

    def __rtruediv__(self, other: Any) -> RV:
        return self.interface.div(other, self)

    def __pow__(self, other: Any) -> RV:
        return self.interface.pow(self, other)

    def __rpow__(self, other: Any) -> RV:
        return self.interface.pow(other, self)

    def __matmul__(self, other: Any) -> RV:
        return self.interface.matmul(self, other)

    def __rmatmul__(self, other: Any) -> RV:
        return self.interface.matmul(other, self)

    def __getitem__(self, index: Any, /) -> RV:
        indices = index if isinstance(index, tuple) else (index,)
        return self.interface.getitem(self, *indices)

    @property
    def T(self) -> RV:
        """Return the transpose selected by the interface."""
        return self.interface.transpose(self)

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        raise TypeError(f"Cannot convert {self!r} to a numpy array")

    @property
    def shape(self) -> Shape:
        """The shape of the `RV`. (A tuple of ints.)"""
        return self._shape

    @property
    def ndim(self) -> int:
        """The number of dimensions of the `RV`. Equal to the length of `RV.shape`."""
        return len(self._shape)

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is the same `RV` node.

        `RV` equality and hashing are identity-based, regardless of whether two `RV` objects have
        equal `Op` objects and parents. Use `rv_equal` to compare the values represented by `RV`
        graphs.

        Examples
        --------
        >>> x = RV(Constant(1))
        >>> y = RV(Constant(1))
        >>> x == y
        False
        >>> x == x
        True
        >>> rv_equal(x, y)
        True
        """
        return self is other

    def __hash__(self) -> int:
        """Return the identity-based hash of this `RV` node."""
        return object.__hash__(self)

    def __repr__(self) -> str:
        """Return a structural Python representation of this `RV`.

        The representation contains the `RV`'s operation followed by recursive representations of
        its ordered parents, extending the convention of `Op.__repr__` to the graph.
        """
        ret = "RV(" + repr(self.op)
        if self.parents:
            for p in self.parents:
                ret += ", " + repr(p)
        ret += ")"
        return ret

    def __str__(self) -> str:
        """Return a compact, human-readable expression for this `RV`.

        The expression contains the compact representation of the operation followed by recursive
        representations of its ordered parents, extending the convention of `Op.__str__` to the
        graph.
        """
        ret = str(self.op)
        if self.parents:
            ret += util.comma_separated(self.parents, fun=str, parens=True, spaces=True)
        return ret

    def __setattr__(self, key, value):
        if self._frozen:
            raise TypeError("RVs are immutable after init.")
        object.__setattr__(self, key, value)

    def __delattr__(self, key):
        if self._frozen:
            raise TypeError("RVs are immutable after init.")
        object.__delattr__(self, key)


# --------------------------------------------------------------------------------------------------
# equality
# --------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def rv_equal(A: RV, B: RV) -> bool:
    """Are ``A`` and ``B`` equal in a mathematical sense?

    That is, are they guaranteed to always have the same value? This is defined by the following
    rules:

    1. If ``A.op`` is random, then ``A`` is is `rv_equal` to ``B`` if and only if they refer to the
       same object in memory.
    2. If ``A.op`` is non-random, then ``A`` is `rv_equal` to ``B`` if and only if they have the
       same `Op` and their parents are `rv_equal` (defined recursively using this function).

    This function is implemented using caching to make it efficient even with large graphs. This
    doesn't change the results since `RV` and `Op` objects are immutable.

    Args:
        A: the first `RV` to be compared
        B: the second `RV` to be compared

    Examples
    --------

    Constants behave as you'd expect.

    >>> a = RV(Constant(0.5))
    >>> b = RV(Constant(0.5))
    >>> c = RV(Constant(0.7))
    >>> # same object always equal
    >>> rv_equal(a, a)
    True
    >>> # equivalent non-random ops, all (nonexistent) parents equal
    >>> rv_equal(a, b)
    True
    >>> # non-equivalent ops always non-equal
    >>> rv_equal(a, c)
    False

    Random `Op` objects are always non-equal unless they are the same object.

    >>> rv_equal(RV(Bernoulli(), a), RV(Bernoulli(), a))
    False
    >>> rv_equal(RV(Bernoulli(), a), RV(Bernoulli(), b))
    False
    >>> rv_equal(RV(Bernoulli(), a), RV(Bernoulli(), c))
    False

    Non-random op are if ops are equal and parents are recursively equal.

    >>> rv_equal(RV(Exp(), a), RV(Exp(), a))
    True
    >>> rv_equal(RV(Exp(), a), RV(Exp(), b))
    True
    >>> rv_equal(RV(Exp(), a), RV(Exp(), c))
    False
    """
    if A.op.random:
        return A is B
    else:
        return (
            A.op == B.op
            and len(A.parents) == len(B.parents)
            and all(rv_equal(a, b) for a, b in zip(A.parents, B.parents))
        )


# --------------------------------------------------------------------------------------------------
# Inspection and display
# --------------------------------------------------------------------------------------------------


def _upstream_named(
    vars: PyTree[RV], namespace: Mapping[str, object] | None = None
) -> dict[RV, str]:
    """Return upstream RVs in topological order with unique display names."""
    import jax.tree_util

    all_vars = jax.tree_util.tree_leaves(vars)
    if not all(isinstance(node, RV) for node in all_vars):
        raise TypeError("Expected a PyTree of RVs")
    nodes = dag.upstream_nodes(all_vars)

    aliases = util.node_names(nodes, namespace if namespace is not None else {})

    count = 0
    node_to_name: dict[RV, str] = {}
    used_names = {name for names in aliases.values() for name in names}
    for node in nodes:
        assert isinstance(node, RV)

        if node in aliases:
            name = aliases[node][0]
        else:
            name = util.num2str(count)
            while name in used_names:
                count += 1
                name = util.num2str(count)
            count += 1

        node_to_name[node] = name
        used_names.add(name)

    return node_to_name


def print_upstream(vars: PyTree[RV], namespace: Mapping[str, object] | None = None) -> None:
    """Print all upstream RVs as a readable table.

    Parameters
    ----------
    vars
        An RV or a PyTree of RVs whose upstream graph is printed. Each distinct RV appears once.
    namespace
        Optional mapping supplying names, such as ``locals()``. Lists, tuples, and dictionary
        values are searched recursively; the first name found for each upstream RV is used.
        The namespace supplies labels only and does not add roots to the printed graph.
        Unnamed RVs receive names ``a``, ``b``, ... in topological order, skipping all discovered
        names, including unused aliases. See `pangolin.util.node_names` for path naming rules.

    Raises
    ------
    TypeError
        If an input leaf is not an RV.
    ValueError
        If namespace traversal encounters a cyclic container or a name shared by distinct
        upstream RVs.

    Examples
    --------
    >>> r = RV(Constant(0.5))
    >>> s = RV(Bernoulli(), r)
    >>> t = RV(Constant(2))
    >>> u = RV(Normal(), s, t)
    >>> v = RV(Constant([75, 50, 99]))
    >>> print_upstream([u, v]) # use autogenerated names
    shape | statement
    ----- | ---------
    ()    | a = 0.5
    ()    | b ~ bernoulli(a)
    ()    | c = 2
    ()    | d ~ normal(b, c)
    (3,)  | e = [75 50 99]
    >>> print_upstream({'dog':[u], 'kitty':(v,)}) # any pytree is OK
    shape | statement
    ----- | ---------
    ()    | a = 0.5
    ()    | b ~ bernoulli(a)
    ()    | c = 2
    ()    | d ~ normal(b, c)
    (3,)  | e = [75 50 99]
    >>> print_upstream([u, v], locals())
    shape | statement
    ----- | ---------
    ()    | r = 0.5
    ()    | s ~ bernoulli(r)
    ()    | t = 2
    ()    | u ~ normal(s, t)
    (3,)  | v = [75 50 99]
    >>> print_upstream([u, v], {"cool_r": r, "dog": u, "kitty": v})
    shape | statement
    ----- | ---------
    ()    | cool_r = 0.5
    ()    | a ~ bernoulli(cool_r)
    ()    | b = 2
    ()    | dog ~ normal(a, b)
    (3,)  | kitty = [75 50 99]

    RVs stored in nested containers receive names using dictionary keys and sequence indices.
    Shared parents appear only once:

    >>> from pangolin import interface as pi
    >>> params = {
    ...     "location": pi.normal(0, 1),
    ...     "scale": pi.exponential(1),
    ... }
    >>> observations = [
    ...     pi.normal(params["location"], params["scale"])
    ...     for _ in range(2)
    ... ]
    >>> print_upstream(observations, locals())
    shape | statement
    ----- | ---------
    ()    | a = 0
    ()    | b = 1
    ()    | params['location'] ~ normal(a, b)
    ()    | c = 1
    ()    | params['scale'] ~ exponential(c)
    ()    | observations[0] ~ normal(params['location'], params['scale'])
    ()    | observations[1] ~ normal(params['location'], params['scale'])

    See Also
    --------
    graph_upstream : Print upstream RVs as a Unicode DAG.
    """

    node_to_name = _upstream_named(vars, namespace)

    if not node_to_name:
        print("[empty vars, nothing to print]")
        return

    max_shape = 5
    for node in node_to_name:
        max_shape = max(max_shape, len(str(node.shape)))

    print(f"shape{' ' * (max_shape - 5)} | statement")
    print(f"{'-' * max_shape} | ---------")
    for node, name in node_to_name.items():
        parent_names = []
        for parent in node.parents:
            assert isinstance(parent, RV)
            parent_names.append(node_to_name[parent])
        parent_names_str = util.comma_separated(parent_names, parens=False, spaces=True)

        shape_str = str(node.shape)
        shape_str += " " * (max_shape - len(shape_str))

        relation = "~" if node.op.random else "="

        line = f"{shape_str} | {name} {relation} {node.op}"
        if node.parents:
            line += f"({parent_names_str})"

        print(line)


def graph_upstream(
    vars: PyTree[RV],
    namespace: Mapping[str, object] | None = None,
    *,
    number_parents: Literal["off", "on", "auto"] = "auto",
) -> None:
    """Print all upstream RVs and shapes as a Unicode DAG.

    Circles are RVs. Edges always travel from top to bottom and from left to right; they never
    travel upward or leftward.

    ``┼`` is always a pure crossing: the horizontal and vertical lines do not connect there.

    A node's distinct parents are read from left to right by default. When that visual order differs
    from the operation's parent order, automatic subscripts (``₀``, ``₁``, ``₂``, ...) record
    the position of each parent. When the same parent appears in more than one position, ``⁎``
    is used.

    This gives a compact overview of chains and simple branches; for dense graphs or an explicit
    statement of every parent relationship, use `print_upstream`. Detailed rendering conventions
    are documented by `pangolin.util.render_dag`.

    Parameters
    ----------
    vars
        An RV or a PyTree of RVs whose upstream graph is printed. Each distinct RV appears once.
    namespace
        Optional mapping supplying names, such as ``locals()``. Lists, tuples, and dictionary
        values are searched recursively; the first name found for each upstream RV is appended
        to its label. The namespace supplies labels only and does not add roots to the graph.
        Unnamed RVs have no name annotation. See `pangolin.util.node_names` for path naming rules.
    number_parents
        Whether to number no edges, all edges, or only edges whose parent order or multiplicity
        would otherwise be ambiguous. A repeated parent is marked with ``⁎`` when annotations are
        enabled.

    Raises
    ------
    TypeError
        If an input leaf is not an RV.
    ValueError
        If ``number_parents`` is not ``"off"``, ``"on"``, or ``"auto"``; or if parent annotations
        would be printed for a node with more than ten parent positions. Also raised if namespace
        traversal encounters a cyclic container or a name shared by distinct upstream RVs.

    Examples
    --------
    A simple example:

    >>> from pangolin import interface as pi
    >>> x = pi.constant(3.3)
    >>> y = pi.exp(x)
    >>> z = pi.normal(x, y)
    >>> graph_upstream(z, {"z": z})
    ○   () 3.3
    │
    ├─○ () exp
    │ │
    └─○ () normal ← z

    A more complex example:

    >>> x = pi.constant(2)
    >>> y = pi.exp(x)
    >>> z = pi.log(x)
    >>> u = pi.beta_binomial(x, y, z)
    >>> v = y + u

    These RVs form a small DAG that can be drawn spaciously as follows::

        x
        │
        ├─────────┬────┐
        │         │    │
        └───▶y    │    │
             │    ↓    │
             │    z    │
             │    │    │
             │    ↓    │
             ├───▶u ◀──┘
             │    │
             │    ↓
             └───▶v

    `graph_upstream` draws the same DAG compactly, assigning each node its own row:

    >>> graph_upstream(v, locals())
    ○     () 2 ← x
    │
    ├─○   () exp ← y
    │ │
    ├─┼─○ () log ← z
    │ ├┐│
    └─┼┴○ () beta_binomial ← u
      │ │
      └─○ () add ← v

    >>> x = pi.constant([-3.0, 0.0, 2.5])
    >>> y = pi.constant(3.1)
    >>> z = pi.vfor(lambda i: pi.normal(pi.exp(x[i]), y), i=3)
    >>> graph_upstream(z, {"z": z})
    ○   (3,) [-3.  0.  2.5]
    │
    ○   (3,) vmap(exp, [0], 3)
    │
    │ ○ ()   3.1
    │ │
    └─○ (3,) vmap(normal, [0, None], 3) ← z

    See Also
    --------
    print_upstream : Print upstream RVs as a table.
    pangolin.util.render_dag : Generic renderer with a complete glyph reference.
    """

    node_to_name = _upstream_named(vars)
    aliases = util.node_names(node_to_name, namespace if namespace is not None else {})

    if not node_to_name:
        print("[empty vars, nothing to print]")
        return

    nodes = list(node_to_name)
    parents = [node.parents for node in nodes]
    max_shape = max(len(str(node.shape)) for node in nodes)
    max_shape = max(2, max_shape)

    def label_fn(node: RV) -> str:
        shape_str = str(node.shape)
        shape_str += " " * (max_shape - len(shape_str))

        label = f"{shape_str} {node.op}"

        if node in aliases:
            label += f" ← {aliases[node][0]}"
        return label

    util.render_dag(nodes, parents, label_fn=label_fn, number_parents=number_parents)
