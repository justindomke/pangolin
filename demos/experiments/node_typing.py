from typing import Self, TypeVar


class Node:
    def __init__(self, *parents: Self):
        self.parents = parents


class MyNode(Node):
    def __init__(self, val: int, *parents: Self):
        self.val = val
        super().__init__(*parents)

    def __add__(self, other: Self) -> "MyNode":
        return MyNode(self.val + other.val, self, other)


class OtherNode(Node):
    pass


def get_first_neighbor_312[N: Node](node: N) -> N:
    if not node.parents:
        return node
    return node.parents[0]


N = TypeVar("N", bound="Node")


def get_first_neighbor(node: N) -> N:
    if not node.parents:
        return node
    return node.parents[0]


n1 = MyNode(10)
o1 = OtherNode()

n2 = MyNode(20, n1)

# n3 = MyNode(20, o1)  # error!

result = get_first_neighbor(n1)
added = result + result
