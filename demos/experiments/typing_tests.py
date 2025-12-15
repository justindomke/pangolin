from typing import Sequence, Callable, Optional, TypeVar, Self, Generic, Any


class Node:
    def __init__(self, *parents: Self):
        self._parents: tuple[Self, ...] = parents

    @property
    def parents(self) -> tuple[Self, ...]:
        return self._parents


class Op:
    pass


class SubOp(Op):
    pass


OpU = TypeVar("OpU", bound=Op, covariant=True)
NodeT = TypeVar("NodeT", bound=Node)


class SubNode(Node, Generic[OpU]):
    _parents: tuple["SubNode[Any]"]

    def __init__(self, op: OpU, *parents: "SubNode[Any]"):
        self.op = op
        super().__init__(*parents)

    @property
    def parents(self) -> tuple["SubNode[Any]", ...]:
        return self._parents


class SubSubNode(SubNode[OpU], Generic[OpU]): ...


def get_first(a: NodeT, b: NodeT) -> NodeT:
    return a


op1 = Op()
op2 = Op()
a = SubNode(op1)
a_parents: tuple[SubNode[Any], ...] = a.parents

b = SubNode(op2)
c: SubNode[Op] = get_first(a, b)
print(c.op)

d = SubSubNode(op1)
e = SubSubNode(op2)
f: SubSubNode[Op] = get_first(d, e)

op3 = SubOp()
op4 = SubOp()
g: SubSubNode[SubOp] = get_first(SubSubNode(op3), SubSubNode(op4))

h: SubSubNode[Op] = get_first(SubSubNode(op1), SubSubNode(op4))
