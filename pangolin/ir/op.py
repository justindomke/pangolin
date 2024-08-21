from abc import ABC, abstractmethod
from .rv import RV
from typing import Type
from collections.abc import Callable


class Op(ABC):
    """
    Abstract base class for operators. An `Op` represents a deterministic function or conditional
    distribution.

    Notes:
    * An `Op` *only* represents an operator—all functionality for sampling or density evaluation,
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
        self, name: str, random: bool):
        """
        Create a new op
        Parameters
        ----------
        name: str
            name for the operator (used only for printing, not functionality)
        random: bool
            is this a conditional distribution? (`random=True`) or a deterministic function (
            `random=False`)?
        """
        assert isinstance(name, str)
        assert isinstance(random, bool)
        self.name: str = name
        "The name of the Op"
        self.random: bool = random
        "True for conditional distributions, False for deterministic functions"
        self._frozen = True  # freeze after init

    def get_shape(self, *parents_shapes):
        """
        Given the shapes of parents, return the shape of the output of this `Op`. Subclasses
        must provide a `_get_shape(*parents_shapes)` function. This is needed because some `Op`s
        (e.g. multivariate normal distributions) can have different shapes depending on the
        shapes of the parents.

        It is also expected that `Op`s define a `_get_shape` method that does error checking—e.g.
        verifies that the correct number of arguments are provided and the shapes of the parents
        are coherent with each other.
        """
        return self._get_shape(*parents_shapes)

    @abstractmethod
    def _get_shape(self, *parents_shapes):
        pass

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __setattr__(self, key, value):
        if self._frozen:
            raise TypeError("CondDists are immutable after init.")
        else:
            self.__dict__[key] = value

    def __repr__(self):
        return self.name + "()"
