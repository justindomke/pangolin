from abc import ABC, abstractmethod
from .rv import RV

current_rv = [RV]


class SetAutoRV:
    def __init__(self, rv_class):
        self.rv_class = rv_class

    def __enter__(self):
        current_rv.append(self.rv_class)

    def __exit__(self, exc_type, exc_value, exc_tb):
        assert current_rv.pop(1) == self.rv_class


class Op(ABC):
    """
    An `Op` represents a deterministic function or conditional distribution.
    Note that it *only* represents it—all functionality for sampling or density
    evaluation, etc. is left to inference engines.
    * Frozen after creation.
    * `__eq__` should be defined so that *mathematically equivalent*
    `CondDist`s are equal, regardless of if they occupy the same place in memory.
    Unnecessary for pre-defined `CondDist` objects like `normal_scale` or `add`,
    but needed for `CondDist`s that are constructed with parameters, like `VMapDist`.
    * Some `CondDist`s, e.g. multivariate normal distributions, can have different
    shapes depending on the shapes of the parents. So a concrete `CondDist` must
    provide a `_get_shape(*parents_shapes)` method to resolve this.
    """

    _frozen = False

    def __init__(self, name: str, random: bool):
        assert isinstance(name, str)
        assert isinstance(random, bool)
        self.name: str = name
        "The name of the Op"
        self.random: bool = random
        "True for conditional distributions, False for deterministic functions"
        self._frozen = True  # freeze after init

    def get_shape(self, *parents_shapes):
        """
        Given the shapes of parents, return the shape of the output of this `Op`.
        """
        return self._get_shape(*parents_shapes)

    @abstractmethod
    def _get_shape(self, *parents_shapes):
        pass

    def __setattr__(self, key, value):
        if self._frozen:
            raise TypeError("CondDists are immutable after init.")
        else:
            self.__dict__[key] = value

    def __call__(self, *parents):
        """when you call a conditional distribution you get a RV"""
        return current_rv[-1](self, *parents)

    def __repr__(self):
        return self.name + "()"
