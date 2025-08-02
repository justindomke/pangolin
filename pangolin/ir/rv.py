from pangolin import dag, util
from typing import Self
from .types import Shape
from .op import Op

class RV(dag.Node):    
    """
    A RV is essentially just a tuple of an Op and a set of parent RVs.
    """
    _frozen = False
    __array_priority__ = 1000  # so x @ y works when x numpy.ndarray and y RV

    def __init__(self, op: Op, *parents: Self):
        """
        Initialize an RV with Op `op` and parents `*parents`.
        """

        parents_shapes = tuple(p.shape for p in parents)
        self._shape = op.get_shape(*parents_shapes)
        self.op = op
        "The Op corresponding to this RV."
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
        #if self.parents:
        #    ret += ", parents=[" + util.comma_separated(self.parents, repr, False) + "]"
        if self.parents:
            for p in self.parents:
                ret += ', ' + repr(p)
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

    __all__ = ['op','shape','parents']