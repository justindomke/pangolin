from typing import TypeVar, Protocol

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class ClosedNode(Protocol):
    @property
    def prev(self) -> Self: ...


T = TypeVar("T", bound=ClosedNode, covariant=True)


class OpenNode(Protocol[T]):
    @property
    def prev(self) -> T: ...


def traceback(start: OpenNode[T], steps: int) -> T:
    curr: T = start.prev

    for _ in range(steps - 1):
        curr = curr.prev

    return curr


class NodeY:
    def __init__(self, prev: "NodeY", y: int):
        self._prev = prev
        self.y = y

    @property
    def prev(self):
        return self._prev


class SubNodeY(NodeY):
    def __init__(self, prev: NodeY, other: int):
        super().__init__(prev, 0)
        self.other = other


root = NodeY(None, y=10)  # type: ignore
entry: OpenNode[NodeY] = SubNodeY(root, other=99)

result = traceback(entry, 1)

print(result.y)
