from abc import ABC, abstractmethod
#from .rv import RV
from typing import Type
from collections.abc import Callable
from .types import Shape
from pangolin import util

class Op(ABC):
    """
    Abstract base class for operators. An `Op` represents a deterministic function or conditional
    distribution.

    Notes:
    * An `Op` only *represents* an operator—all functionality for sampling or density evaluation,
    etc. is left to inference engines.
    * `Op`s must provide an `__eq__` method such that *mathematically equivalent* `Op`s are
    equal, regardless of if they occupy the same place in memory. For example, `d1 = Normal()`
    and `d2 = Normal()` then `d1 == d2`. This base class provides a default implementation that
    simply tests if the types are the same. If an Op takes parameters (e.g. `VMap`), this should be
    overridden.
    * `Op`s are programmatically enforced to be frozen after initialization.
    """

    _frozen = False

    def __init__(
        self, random: bool):
        """
        Create a new op

        Parameters
        ----------
        random: bool
            is this a conditional distribution? (`random==True`) or a deterministic function (
            `random==False`)
        """
        assert isinstance(random, bool)
        self.random: bool = random
        "True for conditional distributions, False for deterministic functions"
        self._frozen = True  # freeze after init

    def get_shape(self, *parents_shapes: Shape) -> Shape:
        """
        Given the shapes of parents, return the shape of the output of this `Op`. Subclasses
        must provide a `_get_shape(*parents_shapes)` function. This is needed because some `Op`s
        (e.g. multivariate normal distributions) can have different shapes depending on the
        shapes of the parents.

        It is also expected that `Op`s define a `_get_shape` method that does error checking—e.g.
        verifies that the correct number of parents are provided and the shapes of the parents
        are coherent with each other.
        """
        return self._get_shape(*parents_shapes)

    @abstractmethod
    def _get_shape(self, *parents_shapes: Shape) -> Shape:
        pass

    def __eq__(self, other):
        "Returns true if `self` and `other` have the same type. If subtypes have more structure, should override."
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

