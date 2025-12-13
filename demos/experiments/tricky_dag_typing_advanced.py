from typing import TypeVar, Protocol

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class ClosedNode(Protocol):
    @property
    def parents(self) -> tuple[Self, ...]: ...


T = TypeVar("T", bound=ClosedNode, covariant=True)


class OpenNode(Protocol[T]):
    @property
    def parents(self) -> tuple[T, ...]: ...


def traceback(start: OpenNode[T]) -> T:
    curr: T = start.parents[0]

    while len(curr.parents) > 0:
        curr = curr.parents[0]

    return curr


class NodeY:
    def __init__(self, y: int, *parents: "NodeY"):
        self._parents = parents
        self.y = y

    @property
    def parents(self):
        return self._parents


class SubNodeY(NodeY):
    def __init__(self, other: int, *parents: NodeY):
        super().__init__(0, *parents)
        self.other = other


root = NodeY(y=10)
entry: OpenNode[NodeY] = SubNodeY(99, root)

result = traceback(entry)

print(result.y)
