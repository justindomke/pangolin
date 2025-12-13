from typing import TypeVar, Protocol, cast

# would LIKE to declare bound="RecursiveNode[T]" but Python can't handle this
T = TypeVar("T", bound="RecursiveNode", covariant=True)


class RecursiveNode(Protocol[T]):
    @property
    def parents(self) -> tuple[T, ...]: ...


def traceback(start: RecursiveNode[T]) -> T:
    curr: T = start.parents[0]

    while len(curr.parents) > 0:
        # need cast() because we couldn't declare T to be bounded to RecursiveNode[T]
        curr = cast(T, curr.parents[0])

    return curr


class NodeY(RecursiveNode["NodeY"]):
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


# this is valid
root1: NodeY = NodeY(y=10)

root2: RecursiveNode = NodeY(y=10)

root3: RecursiveNode[NodeY] = NodeY(y=10)

# this would NOT be valid (yay)
# entry: RecursiveNode[SubNodeY] = SubNodeY(99, NodeY(y=10))

# this is also vaid
entry1: RecursiveNode[NodeY] = SubNodeY(99, root1)
entry2: RecursiveNode = SubNodeY(99, root1)
entry3: NodeY = SubNodeY(99, root1)

entry4: RecursiveNode[NodeY] = SubNodeY(99, root2)
entry5: RecursiveNode = SubNodeY(99, root2)
entry6: NodeY = SubNodeY(99, root2)

entry7: RecursiveNode[NodeY] = SubNodeY(99, root3)

result1: NodeY = traceback(entry1)
result2: NodeY = traceback(entry2)
result3: NodeY = traceback(entry3)
result4: NodeY = traceback(entry4)
result5: NodeY = traceback(entry5)
result6: NodeY = traceback(entry6)
result7: NodeY = traceback(entry7)

print(result1.y)

result8: RecursiveNode[NodeY] = traceback(entry1)

print(result2.y)
