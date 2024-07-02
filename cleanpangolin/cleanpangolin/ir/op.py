from abc import ABC, abstractmethod
from .rv import RV
from typing import Type
from collections.abc import Callable


class Op(ABC):
    """
    Abstract base class for operators.
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

    def __init__(
        self, name: str, random: bool, default_rv_lookup: Callable[[], Type[RV]] = lambda: RV
    ):
        """
        Create a new op
        Parameters
        ----------
        name: str
            name for the operator (used only for printing, not functionality)
        random: bool
            is this a conditional distribution? (`random=True`) or a deterministic function (
            `random=False`)?
        default_rv_lookup: Callable[[],Type[RV]]
            `op(*args)` can be used as a convenient shorthand for `RV(op,*args)`. If this is
            done, `default_rv_lookup()` is first called to choose the type of `RV`. (Default is
            to always return `RV`). The purpose of this is to allow the same code to be used to
            create, e.g., `RV`s that support operator overloading or abstract RVs depending on
            the context.
        """
        assert isinstance(name, str)
        assert isinstance(random, bool)
        self.name: str = name
        "The name of the Op"
        self.random: bool = random
        "True for conditional distributions, False for deterministic functions"
        self.default_rv_lookup = default_rv_lookup
        "When new RVs are created with `__call__` use this function to get RV type for created RV"
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
        """@public
        when you call a conditional distribution you get a new RV with type determined by
        `default_rv_lookup()`"""
        rv_class = self.default_rv_lookup()  # might be RV or a subclass
        return rv_class(self, *parents)

    def __repr__(self):
        return self.name + "()"
