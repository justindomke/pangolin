from typing import TypeVar, Protocol

P = TypeVar("P", bound="Node", covariant=True)


class Node(Protocol[P]):
    @property
    def parents(self) -> tuple["Node[P]", ...]: ...


def traceback(start: Node[P]):
    curr = start

    while len(curr.parents) > 0:
        curr = curr.parents[0]

    return curr


class NodeY:
    def __init__(self, y: int, *parents: "NodeY"):
        self._parents = parents
        self.y = y

    @property
    def parents(self) -> tuple["NodeY", ...]:
        return self._parents


class SubNodeY(NodeY):
    def __init__(self, other: int, *parents: NodeY):
        super().__init__(0, *parents)
        self.other = other


root = NodeY(y=10)

entry = SubNodeY(99, root)

result = traceback(root)

print(result.y)
