from typing import Self, Protocol, Any


class SupportsAdd(Protocol):
    def __add__(self, other: Self, /) -> Any: ...


class Node:
    def __init__(self, *parents: Self):
        self.parents = parents


class MyNode[V: SupportsAdd](Node):

    def __init__(self, val: V, *parents: Self):
        self.val = val
        super().__init__(*parents)

    def __add__(self, other: Self) -> "MyNode[V]":
        return MyNode(self.val + other.val, self, other)

    def __repr__(self):
        return f"MyNode({self.val})"


def get_first_neighbor[N: Node](node: N) -> N:
    if not node.parents:
        return node
    return node.parents[0]


n_int_1 = MyNode(10)
n_int_2 = MyNode(20, n_int_1)
res_int = n_int_1 + n_int_2

n_str_1 = MyNode("Hello")
n_str_2 = MyNode(" World", n_str_1)
res_str = n_str_1 + n_str_2

# n_mix = n_int_1 + n_str_1  # error!

x = get_first_neighbor(n_int_2)
print(x.val + 5)
