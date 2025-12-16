"""
The pangolin IR for defining joint distributions over collections of random variables.
Note that while it is certainly possible to define probabilistic model directly in terms
of this IR, it is quite a "low-level" notation, and **end-users of Pangolin are not
typically expected to manipulate this IR directly**. Instead, users would typically
use an `pangolin.interface` that provides a friendlier way of creating these models.

This IR only *represents* groups of dependent random variables. All actual functionality
is left to inference engines.

The two core abstractions in the IR are `Op` and `RV`. An `Op` represents a simple
function or conditional distribution, while an `RV` consists of a single `Op` and
a (possibly empty) tuple of parent `Op`. An `RV` represents a random variable.
Each `RV` contains a single `Op` and a (possibly empty) tuple of parent `RV`. Thus,
the IR is essentially just a directed graph where each node is a RV with a single `Op`.

That's the whole IR. Unlike in Pyro / NumPyro / PyMC / Tensorflow Probability,
Random variables do not have names, and there is no notion of a "model" class.
In Pangolin you just work with RVs. You can assign those RVs to named variables
or put them into tuples or lists or dicts if you want. But that's up to you.

Other notes:

* All RVs and Ops are immutable. Nothing is allowed to change after initialization.
* All RVs have static shapes. The shape of an RV is recursively determined by its Op and
  the shapes of its parent RVs.    

  
Types of `Op`:

============================ ============
Type                         Ops
============================ ============
Constants                    :class:`Constant`
Arithmetic                   :class:`Add` :class:`Sub` :class:`Mul` :class:`Div`
Trigonometry                 :class:`Arccos` :class:`Arccosh` :class:`Arcsin` :class:`Arcsinh` :class:`Arctan` :class:`Arctanh` :class:`Cos` :class:`Cosh` :class:`Sin` :class:`Sinh` :class:`Tan` :class:`Tanh` 
Other scalar functions       :class:`Pow` :class:`Abs` :class:`Exp` :class:`InvLogit` :class:`Log` :class:`Loggamma` :class:`Logit` :class:`Step` 
Linear algebra               :class:`Matmul` :class:`Inv` 
Other multivariate functions :class:`Sum` :class:`Softmax`
Scalar distributions         :class:`Normal` :class:`NormalPrec` :class:`Lognormal` :class:`Cauchy` :class:`Bernoulli` :class:`BernoulliLogit` :class:`Beta` :class:`Binomial` :class:`Categorical` :class:`Uniform` :class:`BetaBinomial` :class:`Exponential` :class:`Gamma` :class:`Poisson` :class:`StudentT`
Multivariate distributions   :class:`MultiNormal` :class:`Multinomial` :class:`Dirichlet`
Control flow                 :class:`VMap` :class:`Composite` :class:`Autoregressive`
Indexing                     `Index` :class:`SimpleIndex`
============================ ============

In addition, this module provides `print_upstream`, which provides a nice human-readable description of an entire graph.
"""

from __future__ import annotations  # so it's possible to preserve type aliases
from abc import ABC, abstractmethod

from typing import Type, Sequence, Self, Literal, Tuple, Any, cast
from collections.abc import Callable
from pangolin import util, dag
import numpy as np
from numpy.typing import ArrayLike
from jaxtyping import PyTree

Shape = tuple[int, ...]
"""
A `Shape` is just a tuple of ints
"""

########################################################################################
# The fundamental Op class
########################################################################################


class Op(ABC):
    """
    Abstract base class for operators. An `Op` represents a deterministic function or
    conditional distribution. All functionality for sampling or evaluating densities is
    left to inference backends. This is frozen after initialization.

    Simple `Op` typically have no parameters. For example, `Add()` just represents
    scalar addition. The `Op` itself does not indicate what variables are being
    added together. Similarly, ``Normal()`` just represents a normal distribution, but
    it does not itself indicate what the mean and scale of that distribution are.

    An `Op` must provide an `__eq__` method that indicates *mathematical* equality.
    Thus, if ``op1 = Normal()`` and ``op2 = Normal()`` then, ``op1 == op2``, even though
    ``op1`` and ``op2`` are distinct objects. For simple `Op` that take no parameters,
    this is done just by comparing types. For complex `Op` that take parameters, this
    should be overriden.

    An `Op` must also provide a boolean `random` attribute indicating if it is a
    conditional distribution (``True``) or a deterministic function (``False``).

    An `Op` must also a ``get_parents`` method to determine the shape of the output of the
    Op given the shapes of the parents. This method is needed because some types of `Op`
    (e.g. multivariate normal distributions) can have different shapes depending on the
    shapes of the parents. It is also expected that the `get_shape` method will do error
    checking—e.g. verify that the correct number of parents are provided and the shapes of
    the parents are coherent with each other.
    """

    _frozen = False

    def __init__(self):
        self._frozen = True  # freeze after init

    @property
    @abstractmethod
    def random(self) -> bool:
        """
        Is this class a distribution (``True``) or a deterministic function (``False``)
        """
        pass

    @abstractmethod
    def get_shape(self, *parents_shapes: Shape) -> Shape:
        """
        Given the shapes of parents, what is the shape of the output of this Op?

        Args:
            parents_shapes: Shape of each parent

        Returns:
            shape of output of this op

        Raises:
            TypeError: If an incorrect number of parents are provided.
            ValueError: If the shapes of the parents are incoherent.
        """
        pass

    def __eq__(self, other: Op) -> bool:
        """
        Are ``self`` and ``other`` *mathematically* equal?

        Args:
            other: op to compare to.
        """
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))

    def __setattr__(self, key, value):
        if self._frozen:
            raise TypeError("CondDists are immutable after init.")
        else:
            self.__dict__[key] = value

    @property
    def name(self) -> str:
        "Returns the name of the op class as a string"
        return type(self).__name__

    def __repr__(self):
        return self.name + "()"

    def __str__(self):
        """
        Provides a more compact representation, e.g. `normal` instead of `Normal()`
        """
        return util.camel_case_to_snake_case(self.name)


class OpRandom(Op):
    """
    Just an `Op` where ``op.random==True``.
    """

    @property
    def random(self) -> bool:
        """
        ``True``.
        """
        return True


class OpNonrandom(Op):
    """
    Just an `Op` where ``op.random==False``.
    """

    @property
    def random(self) -> bool:
        """
        ``False``.
        """
        return False


################################################################################
# Constant
################################################################################


class Constant(OpNonrandom):
    """
    Represents a "constant" distribution. Has no parents. Data is always stored
    as a numpy array. If you want to live dangerously, you may be able to switch to
    jax's version of numpy by setting ``ir.np = jax.numpy``.

    Parameters
    ----------
    value
        Some constant value that is either a numpy array or something that can be cast
        to a numpy array.
    """

    def __init__(self, value: ArrayLike):
        self.value = np.array(value)
        """The actual stored data, stored as an immutable numpy array"""
        self.value.flags.writeable = False  # make value immutable
        super().__init__()

    def get_shape(self, *parents_shapes: Shape) -> Shape:
        """
        If ``len(parents_shapes)>0``, raises ``ValueError``. Otherwise, returns the
        shape of ``value``.
        """
        if len(parents_shapes) != 0:
            raise ValueError(
                f"Constant got {len(parents_shapes)} arguments but expected 0."
            )
        return self.value.shape

    def __eq__(self, other: Op) -> bool:
        """
        Returns ``True`` if ``other`` is of type ``Constant`` and ``other.value`` is
        exactly the same as ``self.value``.
        """
        if isinstance(other, Constant):
            if (
                self.value.shape == other.value.shape
                and np.all(self.value == other.value)
                and self.value.dtype == other.value.dtype
            ):
                assert hash(self) == hash(
                    other
                ), "bug: hashes don't match for equal Constant"
                return True
        return False

    def __hash__(self):
        return hash(self.value.tobytes())

    def __repr__(self):
        # assure regular old numpy in case jax being used
        # if self.value.ndim > 0 and np.max(self.value.shape) > 5:
        #     ret = "Constant("
        #     with np.printoptions(threshold=5, linewidth=50, edgeitems=2):
        #         ret += np.array2string(self.value)
        #     ret += ")"
        #     print("path 1")
        #     return ret

        numpy_value = np.array(self.value)
        array_str = repr(numpy_value)  # get base string
        array_str = array_str[6:-1]  # cut off "array(" and ")"
        array_str = array_str.replace("\n", "")  # remove newlines
        array_str = array_str.replace(" ", "")  # remove specs
        return "Constant(" + array_str + ")"

    def __str__(self):
        # return str(self.value).replace("\n", "").replace("  ", " ")
        # assure regular old numpy in case jax being used
        numpy_value = np.array(self.value)
        with np.printoptions(threshold=5, linewidth=50, edgeitems=2):
            return (
                np.array2string(numpy_value, precision=3)
                .replace("\n", "")
                .replace("  ", " ")
            )


########################################################################################
# Abstract ScalarOp class
########################################################################################


class ScalarOp(Op):
    """
    An `Op` expecting scalar inputs and producing a single scalar output.
    """

    @property
    @abstractmethod
    def _num_parents(self) -> int:
        pass

    def get_shape(self, *parents_shapes: Shape) -> Shape:
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

    # TODO: simplify?
    def __hash__(self):
        return hash((self.name, self.random, self._num_parents))


################################################################################
# Create scalar Ops
################################################################################


def _generate_expected_parents(num_parents: int) -> dict[str, str]:
    start_code = ord("a")
    expected_parents = {}
    for n in range(num_parents):
        expected_parents[str(chr(start_code + n))] = f"parent {n}"
    return expected_parents


class _OpInfo:
    def __init__(
        self,
        name,
        random,
        expected_parents: int | dict[str, str],
        wikipedia=None,
        notes=[],
    ):
        self.name = name
        self.random = random
        self.notes = notes

        if isinstance(expected_parents, int):
            self.expected_parents = _generate_expected_parents(expected_parents)
        else:
            self.expected_parents = expected_parents

        if random:
            if wikipedia:
                self.wikipedia = (
                    f"https://en.wikipedia.org/wiki/{wikipedia}_distribution"
                )
            else:
                self.wikipedia = f"https://en.wikipedia.org/wiki/{name}_distribution"
        else:
            self.wikipedia = None


def _get_scalar_op_docstring(op_info):
    name = op_info.name
    random = op_info.random
    expected_parents = op_info.expected_parents

    s = f"Represents a {name} `Op`. Takes no parameters. When used in an RV, expects\
        {len(expected_parents)} scalar parent(s):"
    if expected_parents:
        s += "\n\n"
    for parent_name in expected_parents:
        parent_description = expected_parents[parent_name]
        s += f"* **{parent_name}**: {parent_description}"
        s += "\n"

    parent_shapes = ["()"] * len(expected_parents)

    s += f"""

Examples
--------
>>> op = {name}()
>>> op
{name}()
>>> print(op)
{util.camel_case_to_snake_case(name)}
>>> op.random
{random}
>>> op.get_shape{util.comma_separated(parent_shapes, spaces=True)}
()
"""

    if op_info.wikipedia or op_info.notes:
        s += "\nNotes\n---\n"
        if op_info.notes:
            for note in op_info.notes:
                s += note + "\n\n"
        if op_info.wikipedia:
            s += f"`wikipedia definition <{op_info.wikipedia}>`__\n"

    return s


def _scalar_op_factory(name, random, expected_parents, **kwargs) -> Type[ScalarOp]:
    op_info = _OpInfo(
        name=name, random=random, expected_parents=expected_parents, **kwargs
    )
    name = op_info.name
    random = op_info.random
    notes = op_info.notes
    expected_parents = op_info.expected_parents
    num_parents = len(expected_parents)

    def __init__(self):
        ScalarOp.__init__(self)

    @property
    def prop_num_parents(self) -> int:
        return num_parents

    prop_num_parents.__doc__ = f"{num_parents}"

    @property
    def prop_random(self) -> bool:
        return random

    prop_random.__doc__ = f"{random}"

    # __init__.__doc__ = f"""
    # Create a {name} op. Takes no arguments.
    # """

    __doc__ = _get_scalar_op_docstring(op_info)

    MyClass = type(
        name,
        (ScalarOp,),
        {
            "__init__": __init__,
            "__doc__": __doc__,
            "_expected_parents": expected_parents,
            "random": prop_random,
            "_num_parents": prop_num_parents,
            # "_num_parents": num_parents,
            "_notes": notes,
            # "__qualname__": f"ScalarOp.{name}",  # crucial for docs?
            "__module__": "pangolin.ir",
        },
    )

    # # this also seems to be OK
    # class MyClass(ScalarOp):
    #     def __init__(self):
    #         self._expected_parents = expected_parents
    #         ScalarOp.__init__(self, random, num_parents)

    # MyClass.__init__.__doc__ = f"Create a {name} op. Takes no arguments."
    # MyClass.__name__ = name
    # MyClass.__qualname__ = f"ScalarOp.{name}" # crucial for docs!
    # MyClass.__doc__ = _get_scalar_op_docstring(op_info)

    return MyClass


################################################################################
# Arithmetic
################################################################################

Add = _scalar_op_factory(name="Add", random=False, expected_parents=2)
Sub = _scalar_op_factory(name="Sub", random=False, expected_parents=2)
Mul = _scalar_op_factory(name="Mul", random=False, expected_parents=2)
Div = _scalar_op_factory(name="Div", random=False, expected_parents=2)
Pow = _scalar_op_factory(name="Pow", random=False, expected_parents=2)

################################################################################
# Trigonometry
################################################################################

Arccos = _scalar_op_factory(name="Arccos", random=False, expected_parents=1)
Arccosh = _scalar_op_factory(name="Arccosh", random=False, expected_parents=1)
Arcsin = _scalar_op_factory(name="Arcsin", random=False, expected_parents=1)
Arcsinh = _scalar_op_factory(name="Arcsinh", random=False, expected_parents=1)
Arctan = _scalar_op_factory(name="Arctan", random=False, expected_parents=1)
Arctanh = _scalar_op_factory(name="Arctanh", random=False, expected_parents=1)
Cos = _scalar_op_factory(name="Cos", random=False, expected_parents=1)
Cosh = _scalar_op_factory(name="Cosh", random=False, expected_parents=1)
Sin = _scalar_op_factory(name="Sin", random=False, expected_parents=1)
Sinh = _scalar_op_factory(name="Sinh", random=False, expected_parents=1)
Tan = _scalar_op_factory(name="Tan", random=False, expected_parents=1)
Tanh = _scalar_op_factory(name="Tanh", random=False, expected_parents=1)


################################################################################
# Other scalar functions
################################################################################

Abs = _scalar_op_factory(name="Abs", random=False, expected_parents=1)
Exp = _scalar_op_factory(name="Exp", random=False, expected_parents=1)
InvLogit = _scalar_op_factory(name="InvLogit", random=False, expected_parents=1)
Log = _scalar_op_factory(name="Log", random=False, expected_parents=1)
Loggamma = _scalar_op_factory(
    name="Loggamma",
    random=False,
    expected_parents=1,
    notes=[
        "Do we want `scipy.special.loggamma <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.loggamma.html>`__ or `scipy.special.gammaln <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaln.html>`__? These are different!"
    ],
)
Logit = _scalar_op_factory(name="Logit", random=False, expected_parents=1)
Step = _scalar_op_factory(name="Step", random=False, expected_parents=1)

################################################################################
# Linear Algebra
################################################################################


class Matmul(OpNonrandom):
    """
    A class that does matrix multiplication, following the rules of ``numpy.matmul``.
    Currently only 1d and 2d arrays are supported.

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

    def __init__(self):
        super().__init__()

    def get_shape(self, a_shape: Shape, b_shape: Shape) -> Shape:
        """
        Get the shape of applying a matmul to given shapes.

        Parameters
        ----------
        a_shape:
            shape of first argument
        b_shape:
            shape of second argument

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
            raise ValueError(
                f"First parent for Matmul must have 1 or 2 dims (got {len(a_shape)})."
            )

        if len(b_shape) not in [1, 2]:
            raise ValueError(
                f"Second parent for Matmul must have 1 or 2 dims (got {len(a_shape)})."
            )

        if len(a_shape) == 1 and len(b_shape) == 1:
            # inner product
            if a_shape != b_shape:
                raise ValueError(
                    f"Matmul parent shapes do not match ({a_shape} vs. {b_shape})"
                )
            return ()
        elif len(a_shape) == 1 and len(b_shape) == 2:
            # vector-matrix product
            if a_shape[0] != b_shape[0]:
                raise ValueError(
                    f"Matmul parent shapes do not match ({a_shape} vs. {b_shape})"
                )
            return (b_shape[1],)
        elif len(a_shape) == 2 and len(b_shape) == 1:
            # matrix-vector product
            if a_shape[1] != b_shape[0]:
                raise ValueError(
                    f"Matmul parent shapes do not match ({a_shape} vs. {b_shape})"
                )
            return (a_shape[0],)
        elif len(a_shape) == 2 and len(b_shape) == 2:
            # matrix-matrix product
            if a_shape[1] != b_shape[0]:
                raise ValueError(
                    f"Matmul parent shapes do not match ({a_shape} vs. {b_shape})"
                )
            return (a_shape[0], b_shape[1])
        else:
            raise Exception("bug: should be impossible")


class Inv(OpNonrandom):
    """
    Take the inverse of a square matrix
    """

    def __init__(self):
        """"""
        super().__init__()

    def get_shape(self, *parents):
        assert len(parents) == 1
        p_shape = parents[0]
        assert len(p_shape) == 2, "inverse only applies to 2d arrays"
        assert p_shape[0] == p_shape[1], "inverse only for square 2d arrays"
        return p_shape


################################################################################
# Other multivariate funs
################################################################################


class Softmax(OpNonrandom):
    """
    Softmax
    """

    def __init__(self):
        super().__init__()

    def get_shape(self, *parents):
        assert len(parents) == 1
        p_shape = parents[0]
        assert len(p_shape) == 1, "input to softmax would be 1d"
        return p_shape


class Sum(OpNonrandom):
    """Take the sum of an array over some axis"""

    def __init__(self, axis):
        """
        Create a Sum instance
        Parameters
        ----------
        axis: int
            What axis to sum over.
        """
        if isinstance(axis, np.ndarray) and axis.shape == ():
            axis = int(axis)
        if not isinstance(axis, int):
            raise ValueError("axis argument for Sum must be a fixed integer")
        self.axis = axis
        super().__init__()

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


################################################################################
# Scalar distributions
################################################################################


Normal = _scalar_op_factory(
    name="Normal",
    random=True,
    expected_parents={
        "mu": "location / mean",
        "sigma": "scale / standard deviation",
    },
)

NormalPrec = _scalar_op_factory(
    name="NormalPrec",
    random=True,
    expected_parents={
        "mu": "location / mean",
        "tau": "precision / inverse variance",
    },
    wikipedia="Normal",
)

Lognormal = _scalar_op_factory(
    name="Lognormal",
    random=True,
    expected_parents={
        "mu": "logarithm of location",
        "sigma": "logarithm of scale (not sigma squared!)",
    },
    wikipedia="Log-Normal",
)

Bernoulli = _scalar_op_factory(
    name="Bernoulli",
    random=True,
    expected_parents={"theta": "probability (between 0 and 1)"},
)

BernoulliLogit = _scalar_op_factory(
    name="BernoulliLogit",
    random=True,
    expected_parents={"theta": "logit of probability (unbounded)"},
)

Binomial = _scalar_op_factory(
    name="Binomial",
    random=True,
    expected_parents={
        "N": "number of trials",
        "theta": "probability of success for each trial",
    },
)

Cauchy = _scalar_op_factory(
    name="Cauchy",
    random=True,
    expected_parents={"mu": "location", "sigma": "scale"},
)

Uniform = _scalar_op_factory(
    name="Uniform",
    random=True,
    expected_parents={"alpha": "lower bound", "beta": "upper bound"},
    wikipedia="Continuous_uniform",
)

Beta = _scalar_op_factory(
    name="Beta",
    random=True,
    expected_parents={"alpha": "shape", "beta": "shape"},
)

Exponential = _scalar_op_factory(
    name="Exponential",
    random=True,
    expected_parents={"beta": "rate / inverse scale"},
)
Gamma = _scalar_op_factory(
    name="Gamma",
    random=True,
    expected_parents={"alpha": "shape", "beta": "rate / inverse scale"},
    notes=[
        'This follows [stan](https://mc-stan.org/docs/functions-reference/positive_continuous_distributions.html#gamma-distribution)) in using the "shape/rate" parameterization, *not* the "shape/scale" parameterization.'
    ],
)

Poisson = _scalar_op_factory(
    name="Poisson",
    random=True,
    expected_parents={"lambd": "lambda"},
    # TODO: using lambda causes problems with makefun
)


BetaBinomial = _scalar_op_factory(
    name="BetaBinomial",
    random=True,
    expected_parents={
        "N": "as in binomial dist",
        "alpha": "as in beta dist",
        "beta": "as in beta dist",
    },
    wikipedia="Beta-binomial",
    notes=[
        "This follows the (N,alpha,beta) convention of [Stan](https://mc-stan.org/docs/2_19/functions-reference/beta-binomial-distribution.html) (and Wikipedia). Some other systems (e.g. [Numpyro](https://num.pyro.ai/en/stable/distributions.html#betabinomial)) use alternate variable orderings. This is no problem for you as a user, since pangolin does the re-ordering for you based on the backend. But keep it in mind if translating a model from one system to another."
    ],
)

StudentT = _scalar_op_factory(
    name="StudentT",
    random=True,
    expected_parents={
        "nu": "degress of freedom",
        "mu": "location (often 0)",
        "sigma": "scale (often 1)",
    },
    wikipedia="Student's_t",
)


################################################################################
# Multivariate dists
################################################################################


class VecMatOp(OpRandom):
    """
    Convenience class to create "vec mat" distributions that take as input a vector of
    length N, a matrix of size NxN and is a vector of length N. Takes no parameters.
    """

    def __init__(self):
        super().__init__()

    # def get_shape(self, vec_shape: Shape, mat_shape: Shape) -> Shape:
    def get_shape(self, *parents_shapes: Shape) -> Shape:
        if len(parents_shapes) != 2:
            raise TypeError("VecMatOp expected 2 parents")
        vec_shape, mat_shape = parents_shapes

        if len(vec_shape) != 1:
            raise ValueError("first parameter must be a vector.")
        if len(mat_shape) != 2:
            raise ValueError("second parameter must be a matrix.")
        N = vec_shape[0]
        if mat_shape != (N, N):
            raise ValueError(
                "second parameter must be matrix with size matching first parameter"
            )
        return (N,)


class MultiNormal(VecMatOp):
    """
    Create a MultiNormal distribution. Takes no parameters.

    When used in an RV, expects two parents: The mean and covariance.
    """

    def __init__(self):
        """ """
        super().__init__()


class Categorical(OpRandom):
    """
    Create a Categorical distribution. Takes no parameters.

    When used in an RV, expects one parents: A 1-D vector of weights.
    """

    def __init__(self):
        super().__init__()

    def get_shape(self, weights_shape: Shape) -> Shape:
        assert isinstance(weights_shape, tuple)
        if len(weights_shape) != 1:
            raise ValueError(
                f"Categorical op got input with {len(weights_shape)} dims but "
                f"expected 1."
            )
        return ()


class Multinomial(OpRandom):
    """
    Create a Multinomial Op. Takes no parameters.

    When used in an RV, a multinomial op expects a first parent ``n`` (a scalar)
    indicating the number of observations and a second parent ``p`` indicating a vector
    of probabilities (a 1D array). Note that this is different from Stan (which doesn't
    need ``n`` to be passed).
    """

    def __init__(self):
        super().__init__()

    def get_shape(self, n_shape: Shape, p_shape: Shape) -> Shape:

        if n_shape != ():
            raise ValueError("First input to Multinomial op must be scalar")
        if len(p_shape) != 1:
            raise ValueError("Second input to Multinomial op must be a 1-d vector")
        return p_shape


class Dirichlet(OpRandom):
    """Create a Dirichlet Op. Takes no parameters.

    When used in an RV, a Dirichlet op expects one parent, namely the concentration
    """

    def __init__(self):
        """ """
        super().__init__()

    def get_shape(self, concentration_shape: Shape) -> Shape:
        """ """
        if len(concentration_shape) != 1:
            raise ValueError("Dirichlet op must have a single 1-d vector input")
        return concentration_shape


################################################################################
# VMap
################################################################################


def split_shape(shape: Shape, i: int | None) -> tuple[Shape, int | None]:
    if i is None:
        new_shape = shape
        new_axis_size = None
    else:
        lo, mid, hi = (shape[:i], shape[i], shape[i + 1 :])
        new_shape = lo + hi
        new_axis_size = shape[i]
    return new_shape, new_axis_size


def get_sliced_shapes(
    shapes: Sequence[Shape], in_axes: tuple[int | None, ...], axis_size: int | None
) -> tuple[list[Shape], int]:
    axis_size = axis_size
    remaining_shapes = []
    for i, shape in zip(in_axes, shapes, strict=True):
        new_shape, new_axis_size = split_shape(shape, i)
        remaining_shapes.append(new_shape)
        if axis_size is None:
            axis_size = new_axis_size
        elif new_axis_size is not None:
            assert axis_size == new_axis_size, "incoherent axis size"
    assert isinstance(axis_size, int), "couldn't identify axis size"
    return remaining_shapes, axis_size


class VMap(Op):
    """
    Create a ``VMap`` Op. That's *one specific* op vectorized over some number of arguments.

    All arguments here are heavily inspired by
    `jax.lax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`__
    although note that ``VMap`` only maps a single `Op`. (The ``vmap`` function in the
    interface elsewhere takes an arbitrary function and transforms it into a graph of
    RVs with `VMap` `Op` .)

    Args:
        base_op: The `Op` to be mapped
        in_axes: What axis to map for each argument of the op (each can be a
            non-negative int or ``None``)
        axis_size: The size of the mapped axis/axes. Optional unless all elements of
            ``in_axes`` are ``None``.

    Examples
    --------
    >>> # diagonal normal
    >>> op = VMap(Normal(), in_axes=(0, 0))
    >>> op.get_shape((5,), (5,))
    (5,)

    >>> # diagonal normal with shared scale
    >>> op = VMap(Normal(), in_axes=(0, None))
    >>> op.get_shape((5,), ())
    (5,)

    >>> # 2D diagonal normal with shared scale
    >>> op = VMap(VMap(Normal(), in_axes=(0, None)), in_axes=(0, None))
    >>> op.get_shape((6,4), ())
    (6, 4)

    >>> # 2D diagonal normal with "outer product" of location and scale
    >>> op = VMap(VMap(Normal(), in_axes=(None, 0)), in_axes=(0, None))
    >>> op.get_shape((6,), (4,))
    (6, 4)

    >>> # 2D diagonal normal with "outer product" in the other order
    >>> op = VMap(VMap(Normal(), in_axes=(0, None)), in_axes=(None, 0))
    >>> op.get_shape((6,), (4,))
    (4, 6)
    """

    # tuple[int | None, ...] means a tuple of int or None of any length
    # list[int | None] means a list of any length (... not appropriate)
    def __init__(
        self,
        base_op: Op,
        in_axes: tuple[int | None, ...],
        axis_size: int | None = None,
    ):
        """ """

        assert isinstance(base_op, Op)
        if isinstance(in_axes, list):
            in_axes = tuple(in_axes)
        assert isinstance(in_axes, tuple), "in_axes must be tuple"
        if axis_size is None:
            assert any(
                axis is not None for axis in in_axes
            ), "if axis_size=None, at least one axis must be mapped"
        else:
            if not isinstance(axis_size, (int, np.integer)):
                raise Exception(f"axis_size must be None or int was {type(axis_size)}")

        self.base_op = base_op
        self.in_axes = in_axes
        self.axis_size = axis_size
        self._random = base_op.random
        super().__init__()

    @property
    def random(self) -> bool:
        """
        Equal to ``base_op.random``
        """

        return self._random

    def get_shape(self, *parents_shapes: Shape) -> Shape:
        """
        Expects shapes corresponding to the shapes expected by ``base_op`` but with
        extra axes as dictated by ``axis_size``.
        """

        if len(parents_shapes) != len(self.in_axes):
            raise ValueError(
                f"len(in_axes) {len(self.in_axes)} does not match number of parents {len(parents_shapes)}"
            )

        remaining_shapes, axis_size = get_sliced_shapes(
            parents_shapes, self.in_axes, self.axis_size
        )
        dummy_shape: Shape = self.base_op.get_shape(*remaining_shapes)
        return (axis_size,) + dummy_shape

    def __repr__(self):
        out = f"VMap({repr(self.base_op)}, {repr(self.in_axes)}"
        if self.axis_size:
            out += f", {repr(self.axis_size)}"
        out += ")"
        return out

    def __str__(self):
        """
        Return a string representation of the VMap op. Just like ``__repr__`` except
        uses str for calling the recursive distribution.
        """
        # this is kind of overkill but whatever...
        # new_in_axes = jax.tree_util.tree_map(
        #     lambda x: "∅" if x is None else x,
        #     self.in_axes,
        #     is_leaf=util.is_leaf_with_none,
        # )
        # out = f"vmap({str(self.base_op)}, {str(new_in_axes)}"
        out = f"vmap({str(self.base_op)}, {str(self.in_axes)}"
        if self.axis_size:
            out += f", {repr(self.axis_size)}"
        out += ")"
        return out

    def __eq__(self, other):
        if isinstance(other, VMap):
            return (
                self.base_op == other.base_op
                and self.in_axes == other.in_axes
                and self.axis_size == other.axis_size
            )
        return False

    def __hash__(self):
        return hash((self.base_op, self.in_axes, self.axis_size))


################################################################################
# Composite
################################################################################


class Composite(Op):
    """
    A composite distribution can have any number of inputs. par_nums counts all
    the inputs first (in order) followed by the variables generated by the
    composite distribution itself.
    """

    def __init__(
        self,
        num_inputs: int,
        ops: tuple[Op, ...] | list[Op],
        par_nums: tuple[tuple[int, ...], ...] | list[list[int]],
    ):
        assert isinstance(num_inputs, int)
        assert all(isinstance(d, Op) for d in ops)
        for my_par_nums in par_nums:
            assert all(isinstance(i, int) for i in my_par_nums)
        for d in ops[:-1]:
            if d.random:
                raise ValueError(
                    f"all but last op for Composite must be non-random (got {d})"
                )
        self.num_inputs = num_inputs
        self.ops = tuple(ops)
        # self.par_nums = tuple(par_nums)
        self.par_nums = tuple(tuple(pp) for pp in par_nums)
        self._random = ops[-1].random
        super().__init__()

    @property
    def random(self):
        return self._random

    def get_shape(self, *parents_shapes):
        all_shapes = list(parents_shapes)
        for my_op, my_par_nums in zip(self.ops, self.par_nums):
            my_parents_shapes = [all_shapes[i] for i in my_par_nums]
            my_shape = my_op.get_shape(*my_parents_shapes)
            all_shapes.append(my_shape)
        return all_shapes[-1]

    # TODO: str and repr should be much more descriptive
    # should this print f"Composite({ops[-1].name})"?
    def __str__(self):
        if len(self.ops) == 1:
            str_ops = f"({self.ops[0]},)"
        else:
            str_ops = f"({', '.join(str(op) for op in self.ops)})"
        return f"composite({self.num_inputs}, {str_ops}, {self.par_nums})"

    def __repr__(self):
        return f"Composite({self.num_inputs}, {self.ops}, {self.par_nums})"

    def __eq__(self, other):
        if isinstance(other, Composite):
            return (
                self.num_inputs == other.num_inputs
                and self.ops == other.ops
                and self.par_nums == other.par_nums
            )
        return False

    def __hash__(self):
        return hash((self.num_inputs, self.ops, self.par_nums))


################################################################################
# Autoregressive
################################################################################

# TODO:
# maybe remove convenient form for in_axes—make everything explicit
# interface can simplify


class Autoregressive(Op):
    """Represents an autoregressive distribution"""

    def __init__(
        self,
        base_op: Op,
        length: int,
        # in_axes: tuple[int | None, ...] | list[int | None],
        in_axes: Sequence[int | None],
        where_self: int = 0,
    ):
        """
        Parameters
        ----------
        base_cond_dist
            what distribution to repeat on
        num_constants
            number of constant arguments (default 0)
        length
            the number of times to repeat (optional if there are )
        """
        self.base_op = base_op
        # self.num_constants = num_constants
        self.length = length
        self.in_axes = tuple(in_axes)
        self.where_self = where_self
        self._random = base_op.random
        super().__init__()

    @property
    def random(self):
        return self._random

    def get_shape(self, start_shape, *other_shapes):
        # const_shapes = other_shapes[: self.num_constants]
        # other_shapes = other_shapes[self.num_constants :]

        # if self.length is None and other_shapes == ():
        #     raise ValueError("Can't create Autoregressive with length=None and no mapped arguments")
        #
        # if self.length is None:
        #     my_length = other_shapes[0][0]
        # else:
        #     my_length = self.length

        base_input_shapes = []

        for n, (s, ax) in enumerate(zip(other_shapes, self.in_axes, strict=True)):
            if ax is None:
                base_input_shapes.append(s)
            else:
                assert isinstance(ax, int)
                base_input_shapes.append(s[:ax] + s[ax + 1 :])

        # insert self
        # if n == self.where_self:
        #    base_input_shapes.append(start_shape)
        base_input_shapes = (
            base_input_shapes[: self.where_self]
            + [start_shape]
            + base_input_shapes[self.where_self :]
        )

        # for s in other_shapes:
        #    assert s[0] == my_length
        # base_other_shapes = tuple(s[1:] for s in other_shapes)

        # base_input_shapes = (start_shape,) + const_shapes + base_other_shapes
        base_output_shape = self.base_op.get_shape(*base_input_shapes)
        output_shape = (self.length,) + base_output_shape
        return output_shape

    def __eq__(self, other):
        if isinstance(other, Autoregressive):
            return (
                self.base_op == other.base_op
                and self.length == other.length
                and self.in_axes == other.in_axes
                and self.where_self == other.where_self
            )
        return False

    def __hash__(self):
        return hash((self.base_op, self.length, self.in_axes, self.where_self))

    def __str__(self):
        return f"autoregressive({self.base_op}, {self.length}, {self.in_axes}, {self.where_self})"

    def __repr__(self):
        return f"Autoregressive({repr(self.base_op)}, {self.length}, {self.in_axes}, {self.where_self})"


################################################################################
# Indexing
################################################################################

"""
An Index is a deterministic `Op` that takes (1) an RV to be indexed and (2) a set of (
integer-valued) indexing RVs and returns the result of indexing the first RV with the second.

Conceptually, say the user does something like this:

```python
x = constant([1,2,3])
i = categorical([.1,.2,.7])
y = x[i]
```

Then we would like this to lead to an IR something like:

```python
x = RV(Constant([1,2,3]))
p = RV(Constant([.1,.2,.7])
y = RV(Index(),x,i)
```

Seems simple, right? But there are two major subtleties:

**First subtlety.** Given the above description, you might think that the Index Op itself would
need no parameters—there aren't "different kinds" of indexing, after all. And we want to allow
indices to be random variables.

But there is a problem: We also want to allow indexing with *slices* and if *slices* are defined
with random variables, then the *size* of the output would also be random, which would break our
abstraction where all RVs have fixed shapes.

To deal with this, there are different instances of `Index` Ops, depending on what dimensions
will be sliced. All slices must be baked into the Index `Op` with *fixed* integer values. Then,
all the non-sliced arguments are still free to be `RV`s. So an `Index` Op is created by givin a
list of slices, each can either be a fixed slice, or `None`, indicating that the dimension is
unsliced and will come from a `RV`.

So, for example, if you do this:

```python
x = constant([[1,2,3],[4,5,6])
i = categorical([.1,.2,.7])
y = x[:,i]
```

then under the hood you will get a representation like

```python
x = RV(Constant([1,2,3],[4,5,6]])
p = RV(Constant([.1,.2,.7])
i = RV(Categorical,p)
y = RV(Index(slice(None),None),x,i)
```

**Second subtlety.** We would like to support substantially all of Numpy's indexing features. But
Numpy's indexing is much weirder than most people realize. For example, consider this code:

```python
A = np.ones([2,3,4,5])
i = [0,1,0,1,0,1]
A[i,i,:,:].shape # (6,4,5) # ok
A[:,:,i,i].shape # (2,3,6) # ok
A[:,i,i,:].shape # (2,6,5) # fine
A[i,:,:,i].shape # (6,3,4) # ok, I guess
A[:,i,:,i].shape # (6,2,4) # what!?
```

Yes, that is really what happens, try it! What's happening here is that when you have multiple
"advanced indices" (like `i`) above, numpy has very complicated [advanced indexing rules](
https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing). These basically say that:

1. If all the advanced indices are next to each other, then the corresponding dimension in the
output goes at the location of the *first* advanced index.

2. If the advanced indices are separated by any slices, then the corresponding dimension goes at
the *start* in the output.

We just have to live with this, but it complicates things quite a bit.

**Note:** There is one indexing feature from numpy that is *not* supported at the moment,
namely broadcasting of indices. For example, in Pangolin, you cannot write `x[3,[0,1,2]]` as a
shorthand for `x[[3,3,3],[0,1,2]]`. We might change this at some point if there is some need.

**Note** As illustrated above, non-sliced indices are free to be random variables. Although,
currently, JAGS is the only backend that can actually do inference in these settings.
"""


def _slice_length(size, slice):
    return len(np.ones(size)[slice])


class Index(OpNonrandom):
    """
    Represents an `Op` to index into a `RV`. Slices for all sliced dimensions
    must be baked into the Index op when created. Non sliced dimensions are not
    part of the `Op` (they can be random variables).
    """

    def __init__(self, *slices: slice | None):
        """
        Create an Index op given a set of slices.

        Parameters
        ----------
        slices
            A list with length equal to the number of dimensions of the RV that
            is to be indexed. Each element must either be (a) a slice with fixed
            integer values or (b) None, indicating that the dimension will be
            indexed with a RV instead.
        """
        self.slices = slices
        super().__init__()

    @property
    def advanced_at_start(self) -> bool:
        """

        Most people don't realize that this is how numpy works:

        Numpy rules say that if you have "advanced indices" (like ``i``) above
        and they are *separated by a slice* then the dimension for the advanced
        index goes at the start. (This is what happens in the second case above.)
        Otherwise, all the indices go to the location of the first advanced index.
        """
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
                output_shape += (_slice_length(var_shape[n], my_slice),)
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


################################################################################
# Simple Indexing
################################################################################

"""
A SimpleIndex is a deterministic `Op` that takes (1) an RV to be indexed and (2) a set of (
integer-valued) indexing RVs and returns the result of indexing the first RV with the second.

Conceptually, say the user does something like this:

```python
x = constant([1,2,3])
i = categorical([.1,.2,.7])
y = x[i]
```

Then we would like this to lead to an IR something like:

```python
x = RV(Constant([1,2,3]))
p = RV(Constant([.1,.2,.7])
y = RV(Index(),x,i)
```

Seems simple, right? But there are two major subtleties:

**First subtlety.** Given the above description, you might think that the Index Op itself would
need no parameters—there aren't "different kinds" of indexing, after all. And we want to allow
indices to be random variables.

But there is a problem: We also want to allow indexing with *slices* and if *slices* are defined
with random variables, then the *size* of the output would also be random, which would break our
abstraction that all RVs have fixed shapes.

To deal with this, there are different instances of `Index` Ops, depending on what dimensions
will be sliced. All slices must be baked into the Index `Op` with *fixed* integer values. Then,
all the non-sliced arguments are still free to be `RV`s. So an `Index` Op is created by givin a
list of slices, each can either be a fixed slice, or `None`, indicating that the dimension is
unsliced and will come from a `RV`.

So, for example, if you do this:

```python
x = constant([[1,2,3],[4,5,6])
i = categorical([.1,.2,.7])
y = x[:,i]
```

then under the hood you will get a representation like

```python
x = RV(Constant([1,2,3],[4,5,6]])
p = RV(Constant([.1,.2,.7])
i = RV(Categorical,p)
y = RV(Index(slice(None),None),x,i)
```

**Second subtlety.** Numpy's indexing features are way too complicated. For example, consider this code:

```python
A = np.ones([2,3,4,5])
i = [0,1,0,1,0,1]
A[i,i,:,:].shape # (6,4,5) # ok
A[:,:,i,i].shape # (2,3,6) # ok
A[:,i,i,:].shape # (2,6,5) # fine
A[i,:,:,i].shape # (6,3,4) # ok, I guess
A[:,i,:,i].shape # (6,2,4) # what!?
```

Yes, that is really what happens, try it! What's happening here is that when you have multiple
"advanced indices" (like `i`) above, numpy has very complicated [advanced indexing rules](
https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing).

We don't want to impose a heavy burden on users of the IR. Thus, the rules in Pangolin are:

1. Every dimension must be indexed (either with a slice or RV)
2. All indexing is orthogonal.

This means that the output shape is the shape of all the indices, in order.

**Note** As illustrated above, non-sliced indices are free to be random variables. Although,
currently, JAGS is the only backend that can actually do inference in these settings.
"""


class SimpleIndex(OpNonrandom):
    """
    Represents an `Op` to index into a `RV`. Does not deal with slices. (That's the interface's problem)
    """

    def __init__(self):
        """
        Create an Index op
        """
        super().__init__()

    def get_shape(self, *shapes):
        var_shape, *indices_shapes = shapes

        num_indexed = len(indices_shapes)
        num_dims = len(var_shape)
        if num_indexed != num_dims:
            raise ValueError(
                f"Indexed RV with {num_dims} dims with {num_indexed} indices."
            )

        # num_non_scalar = len([idx_shape for idx_shape in indices_shapes if idx_shape is not ()])
        # if num_non_scalar > 1:
        #    raise ValueError(f"Only one RV index can be non-scalar, got {num_non_scalar}")

        output_shape = ()
        for i in indices_shapes:
            output_shape += i

        return output_shape


# function to do full orthogonal indexing on regular numpy arrays (for testing)
def index_orthogonal(array, *index_arrays):
    """
    Create orthogonal index arrays for advanced indexing.
    """
    # Calculate final shape

    assert array.ndim == len(index_arrays)

    index_arrays = [
        np.arange(array.shape[i])[arr] if isinstance(arr, slice) else np.array(arr)
        for i, arr in enumerate(index_arrays)
    ]

    index_shapes = [arr.shape for arr in index_arrays]
    total_dims = sum(len(shape) for shape in index_shapes)

    result_arrays = []
    current_dim = 0

    for arr in index_arrays:
        # Create shape with 1s everywhere except for this array's dimensions
        new_shape = [1] * total_dims

        # Place this array's dimensions in the correct position
        for j, dim_size in enumerate(arr.shape):
            new_shape[current_dim + j] = dim_size

        # Reshape and add to result
        reshaped = arr.reshape(new_shape)
        result_arrays.append(reshaped)

        current_dim += len(arr.shape)

    return array[*result_arrays]


def index_orthogonal_no_slices(A, *index_arrays):
    """
    Create orthogonal index arrays for advanced indexing.
    """
    # Calculate final shape

    assert A.ndim == len(index_arrays)

    index_shapes = [arr.shape for arr in index_arrays]
    total_dims = sum(len(shape) for shape in index_shapes)

    result_arrays = []
    current_dim = 0

    for arr in index_arrays:
        # Create shape with 1s everywhere except for this array's dimensions
        new_shape = [1] * total_dims

        # Place this array's dimensions in the correct position
        for j, dim_size in enumerate(arr.shape):
            new_shape[current_dim + j] = dim_size

        # Reshape and add to result
        reshaped = arr.reshape(new_shape)
        result_arrays.append(reshaped)

        current_dim += len(arr.shape)

    return A[*result_arrays]


########################################################################################
# RVs
########################################################################################

from typing import TypeVar, Generic

OpT = TypeVar("OpT", bound=Op, covariant=True)  # TODO: covariant=True? does it matter?


class RV(dag.Node, Generic[OpT]):
    """
    A `RV` is essentially just an `Op` and a tuple of parent `RV`.

    Parameters
    ----------
    op: `Op`
        The Op defining the RV
    *parents
        The parents of the RV

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

    # _parents: RV # should add this?

    _frozen = False
    _n = 1  # convenient to store order all RVs were created

    def __init__(self, op: OpT, *parents: "RV"):
        parents_shapes = tuple(p.shape for p in parents)
        self._shape = op.get_shape(*parents_shapes)
        self._n = RV._n
        RV._n += 1
        self.op = op
        super().__init__(*parents)
        self._frozen = True

    @property
    def shape(self) -> Shape:
        """
        The shape of the RV. (A tuple of ints.)
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """
        The number of dimensions of the RV. Equal to the length of `shape`.
        """
        return len(self._shape)

    def __len__(self) -> int:
        return self._shape[0]

    def __repr__(self) -> str:
        ret = "RV(" + repr(self.op)
        # if self.parents:
        #    ret += ", parents=[" + util.comma_separated(self.parents, repr, False) + "]"
        if self.parents:
            for p in self.parents:
                ret += ", " + repr(p)
        ret += ")"
        return ret

    def __str__(self) -> str:
        ret = str(self.op)
        if self.parents:
            ret += util.comma_separated(self.parents, fun=str, parens=True, spaces=True)
        return ret

    def __setattr__(self, key, value):
        """
        Set attribute. Special case to freeze after init.
        """
        if self._frozen:
            raise Exception("RVs are immutable after init.")
        else:
            self.__dict__[key] = value

    # def __eq__(self, other):

    # def __hash__(self):
    #     if self.op.random:
    #         return object.__hash__(self)
    #     else:
    #         return hash((self.op, self.parents))


########################################################################################
# equality
########################################################################################

from functools import lru_cache


@lru_cache(maxsize=None)
def rv_equal(A: RV, B: RV) -> bool:
    """
    Are ``A`` and ``B`` *distributionally* equal? That is, are they guaranteed to always have the same value? This is defined by the following rules:

    1. If ``A.op`` is random, then ``A`` is is equal to ``B`` if and only if they refer to the same object in memory.
    2. If ``A.op`` is non-random, then ``A`` is equal to ``B`` if and only if they have the same `Op` and their parents (defined recursively using this function).

    This function is implemented using caching. This doesn't change the results since :class:`RV` s and :class:`Op` s are immutable.

    Args:
        A: the first RV to be compared
        B: the second RV to be compared

    Examples
    --------
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

    >>> # with random op, always non-equal unless same object
    >>> rv_equal(RV(Bernoulli(), a), RV(Bernoulli(), a))
    False
    >>> rv_equal(RV(Bernoulli(), a), RV(Bernoulli(), b))
    False
    >>> rv_equal(RV(Bernoulli(), a), RV(Bernoulli(), c))
    False

    >>> # with non-random op, equal if ops equal and parents recursively equal
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

    # >>> equal(RV(Normal(), a, c), RV(Normal(), a, c)) # random op, same objects
    # False
    # >>> equal(RV(Normal(), a, c), RV(Normal(), b, c)) # random op, equivalent objects
    # False
    # >>> equal(RV(Normal(), a, c), RV(Normal(), b, c)) # random op, equivalent objects
    # False
    # >>> equal(RV(Add(), a, b), RV(Add(), a, b))
    # True
    # >>> equal(RV(Add(), c, d), RV(Add(), c, d))
    # True


################################################################################
# printing
################################################################################


def print_upstream(*vars: PyTree[RV], **named_vars: RV):
    """Prints all upstream variables in a friendly readable format.

    Parameters
    ----------
    vars
        any number of pytrees containing `RV`
    named_vars
        single `RV` s as keyword arguments, will be printed with those names

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
    ()    | d ~ normal(b,c)
    (3,)  | e = [75 50 99]
    >>> print_upstream({'dog':[u], 'kitty':(v,)}) # any pytree is OK
    shape | statement
    ----- | ---------
    ()    | a = 0.5
    ()    | b ~ bernoulli(a)
    ()    | c = 2
    ()    | d ~ normal(b,c)
    (3,)  | e = [75 50 99]
    >>> print_upstream(dog=u, kitty=v)
    shape | statement
    ----- | ---------
    ()    | a = 0.5
    ()    | b ~ bernoulli(a)
    ()    | c = 2
    ()    | dog ~ normal(b,c)
    (3,)  | kitty = [75 50 99]
    >>> print_upstream(r=r, s=s, t=t, u=u, v=v) # control all names
    shape | statement
    ----- | ---------
    ()    | r = 0.5
    ()    | s ~ bernoulli(r)
    ()    | t = 2
    ()    | u ~ normal(s,t)
    (3,)  | v = [75 50 99]
    """

    import jax.tree_util

    all_vars = jax.tree_util.tree_leaves([vars, named_vars])
    nodes = dag.upstream_nodes(all_vars)
    # nodes = cast(list[RV], nodes)  # list[Node] -> list[RV]

    if all_vars == []:
        print("[empty vars, nothing to print]")
        return

    # get maximum # parents
    max_pars = 0
    max_shape = 5
    for node in nodes:
        max_pars = max(max_pars, len(node.parents))
        max_shape = max(max_shape, len(str(node.shape)))

    # if len(nodes) > 1:
    #     digits = 1 + int(np.log10(len(nodes) - 1))
    #     par_str_len = (digits + 1) * max_pars - 1
    # else:
    #     par_str_len = 0

    vars_named = util.reverse_dict(named_vars)

    count = 0
    node_to_id = {}  # type: ignore
    id_to_node = {}
    print(f"shape{' ' * (max_shape - 5)} | statement")
    print(f"{'-' * max_shape} | ---------")
    for node in nodes:
        assert isinstance(node, RV)

        par_ids = [node_to_id[p] for p in node.parents]

        par_id_str = util.comma_separated(par_ids, parens=False)
        # par_id_str = par_id_str + " " * (par_str_len - len(par_id_str))

        shape_str = str(node.shape)
        shape_str += " " * (max_shape - len(shape_str))

        op = "~" if node.op.random else "="

        if node in vars_named:
            id = vars_named[node]
        else:
            # find a unique id
            while (
                util.num2str(count) in id_to_node or util.num2str(count) in named_vars
            ):
                count += 1
            id = util.num2str(count)

        line = f"{shape_str} | {id} {op} {str(node.op)}"
        if node.parents:
            line += "(" + par_id_str + ")"

        print(line)

        node_to_id[node] = id
        id_to_node[id] = node
