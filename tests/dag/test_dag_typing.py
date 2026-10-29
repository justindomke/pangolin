from typing import assert_type

from pangolin.dag import Node, graph_fold, graph_scan
from pangolin.util import PyTree


def test_graph_scan_typing():
    root = Node()
    leaf = Node(root)
    nodes: dict[str, tuple[Node, list[Node]]] = {"nodes": (root, [leaf])}

    def step(node: Node, parent_carries: tuple[int, ...]) -> tuple[int, str]:
        return 1 + sum(parent_carries), str(len(node.parents))

    carries, emissions = graph_scan(nodes, step, boundary={root: 10})

    assert_type(carries, PyTree[int])
    assert_type(emissions, dict[Node, str])


def test_graph_scan_same_node_carry_typing():
    root = Node()
    leaf = Node(root)
    nodes: dict[str, tuple[Node, list[Node]]] = {"nodes": (root, [leaf])}

    def rebuild(node: Node, rebuilt_parents: tuple[Node, ...]) -> tuple[Node, None]:
        return Node(*rebuilt_parents), None

    rebuilt, emissions = graph_scan(nodes, rebuild)

    assert_type(rebuilt, PyTree[Node])
    assert_type(emissions, dict[Node, None])


def test_graph_fold_typing():
    root = Node()
    leaf = Node(root)

    def step(node: Node, parent_carries: tuple[int, ...]) -> tuple[int, float]:
        return 1 + sum(parent_carries), float(len(node.parents))

    folded = graph_fold(leaf, step)

    assert_type(folded, float)


def test_graph_fold_custom_emission_typing():
    root = Node()
    leaf = Node(root)

    def step(node: Node, parent_carries: tuple[int, ...]) -> tuple[int, frozenset[str]]:
        return 1 + sum(parent_carries), frozenset({str(len(node.parents))})

    folded = graph_fold(leaf, step, lambda left, right: left | right)

    assert_type(folded, frozenset[str])
