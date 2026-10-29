import pytest

import pangolin.dag as dag

from pangolin.dag import (
    Node,
    graph_fold,
    graph_scan,
    upstream_nodes_flat,
    upstream_nodes,
    _get_children,
    _has_second_path,
)


def test_public_api():
    assert dag.__all__ == [
        "Node",
        "upstream_nodes_flat",
        "upstream_nodes",
        "graph_scan",
        "graph_fold",
    ]


def same(l1, l2):
    assert {a for a in l1} == {a for a in l2}  # equality of frozen sets


def test_upstream_single():
    a = Node()
    assert upstream_nodes([a]) == [a]


def test_upstream_empty():
    a = Node()
    assert upstream_nodes([]) == []


def test_upstream_pair():
    a = Node()
    b = Node(a)
    assert upstream_nodes([a]) == [a]
    assert upstream_nodes([b]) == [a, b]


def test_upstream_flat():
    a = Node()
    b = Node(a)
    c = Node(b)

    assert upstream_nodes_flat([c]) == [a, b, c]
    assert upstream_nodes_flat([c], node_block=lambda node: node is b) == [c]
    assert upstream_nodes_flat(
        [c], edge_block=lambda child, parent: child is c and parent is b
    ) == [c]


def test_upstream_dupe():
    a = Node()
    b = Node(a)
    assert upstream_nodes([a, b]) == [a, b]


def test_upstream_dupe2():
    a = Node()
    b = Node(a)
    c = Node(b)
    # 000
    assert upstream_nodes([]) == []
    # 001
    assert upstream_nodes([c]) == [a, b, c]
    # 010
    assert upstream_nodes([b]) == [a, b]
    # 011
    assert upstream_nodes([b, c]) == [a, b, c]
    assert upstream_nodes([c, b]) == [a, b, c]
    # 100
    assert upstream_nodes([a]) == [a]
    # 101
    assert upstream_nodes([a, c]) == [a, b, c]
    # 110
    assert upstream_nodes([a, b]) == [a, b]
    assert upstream_nodes([b, a]) == [a, b]
    # 111
    assert upstream_nodes([a, b, c]) == [a, b, c]
    assert upstream_nodes([a, c, b]) == [a, b, c]
    assert upstream_nodes([b, a, c]) == [a, b, c]
    assert upstream_nodes([b, c, a]) == [a, b, c]
    assert upstream_nodes([c, a, b]) == [a, b, c]
    assert upstream_nodes([c, b, a]) == [a, b, c]


def test_upstream_collider():
    a = Node()
    b = Node(a)
    c = Node(a)
    d = Node(b, c)
    assert upstream_nodes([d]) == [a, b, c, d]
    assert upstream_nodes([d, b]) == [a, b, c, d]
    assert upstream_nodes([b, d]) == [a, b, c, d]
    assert upstream_nodes([b, d, a]) == [a, b, c, d]
    assert upstream_nodes([b, d, a, d, b, a]) == [a, b, c, d]


def test_upstream_nested_pytree():
    a = Node()
    b = Node(a)
    c = Node(a)
    d = Node(b, c)

    assert upstream_nodes({"early": [b], "later": (d,)}) == [a, b, c, d]


def test_graph_scan_preserves_requested_pytree_and_emits_every_scanned_node():
    a = Node()
    b = Node(a)
    c = Node(a)
    d = Node(b, c)
    names = {a: "a", b: "b", c: "c", d: "d"}

    def step(node, parent_carries):
        carry = names[node] + "(" + ",".join(parent_carries) + ")"
        return carry, len(node.parents)

    requested = {"early": [b], "outputs": (d, d)}
    carries, emissions = graph_scan(requested, step)

    assert carries == {
        "early": ["b(a())"],
        "outputs": ("d(b(a()),c(a()))", "d(b(a()),c(a()))"),
    }
    assert list(emissions) == [a, b, c, d]
    assert emissions == {a: 0, b: 1, c: 1, d: 2}


def test_graph_scan_without_boundary_traces_every_path_to_parentless_roots():
    left_root = Node()
    left = Node(left_root)
    right_root = Node()
    output = Node(left, right_root)
    names = {
        left_root: "left_root",
        left: "left",
        right_root: "right_root",
        output: "output",
    }
    calls = []

    def step(node, parent_carries):
        calls.append((names[node], parent_carries))
        return names[node], names[node]

    carry, emissions = graph_scan(output, step)

    assert carry == "output"
    assert calls == [
        ("left_root", ()),
        ("left", ("left_root",)),
        ("right_root", ()),
        ("output", ("left", "right_root")),
    ]
    assert list(emissions.values()) == ["left_root", "left", "right_root", "output"]


def test_graph_scan_visits_shared_ancestors_once_for_repeated_outputs():
    root = Node()
    left = Node(root)
    right = Node(root)
    output = Node(left, right)
    calls = {root: 0, left: 0, right: 0, output: 0}

    def step(node, parent_carries):
        calls[node] += 1
        return 1 + sum(parent_carries), None

    carries, emissions = graph_scan([output, output, right], step)

    assert carries == [5, 5, 2]
    assert calls == {root: 1, left: 1, right: 1, output: 1}
    assert list(emissions) == [root, left, right, output]


def test_graph_scan_boundary_supplies_carry_without_emitting():
    a = Node()
    b = Node(a)
    c = Node(b)

    def step(node, parent_carries):
        carry = 1 + sum(parent_carries)
        return carry, (node, parent_carries)

    carries, emissions = graph_scan({"boundary": b, "output": c}, step, boundary={b: 10})

    assert carries == {"boundary": 10, "output": 11}
    assert emissions == {c: (c, (10,))}
    assert a not in emissions
    assert b not in emissions


def test_graph_scan_boundary_cuts_only_the_paths_that_reach_it():
    shared_root = Node()
    boundary_node = Node(shared_root)
    open_node = Node(shared_root)
    output = Node(boundary_node, open_node)
    names = {shared_root: "root", open_node: "open", output: "output"}
    calls = []

    def step(node, parent_carries):
        calls.append((names[node], parent_carries))
        return names[node], names[node]

    carry, emissions = graph_scan(output, step, boundary={boundary_node: "boundary"})

    assert carry == "output"
    assert calls == [
        ("root", ()),
        ("open", ("root",)),
        ("output", ("boundary", "open")),
    ]
    assert list(emissions) == [shared_root, open_node, output]
    assert boundary_node not in emissions


def test_graph_scan_preserves_parent_order_and_repeated_parents():
    a = Node()
    b = Node()
    c = Node(b, a, b)
    names = {a: "a", b: "b", c: "c"}

    def step(node, parent_carries):
        return names[node], parent_carries

    carry, emissions = graph_scan(c, step)

    assert carry == "c"
    assert emissions[c] == ("b", "a", "b")


def test_graph_scan_can_reconstruct_a_graph():
    class NamedNode(Node):
        def __init__(self, name, *parents):
            self.name = name
            super().__init__(*parents)

    a = NamedNode("a")
    b = NamedNode("b", a)
    c = NamedNode("c", a)
    d = NamedNode("d", b, c)

    def rebuild(node, rebuilt_parents):
        rebuilt = NamedNode(node.name.upper(), *rebuilt_parents)
        return rebuilt, (node, rebuilt)

    rebuilt, emissions = graph_scan({"left": b, "output": [d]}, rebuild)
    rebuilt_b = rebuilt["left"]
    rebuilt_d = rebuilt["output"][0]

    assert rebuilt_b.name == "B"
    assert rebuilt_d.name == "D"
    assert rebuilt_d.parents[0] is rebuilt_b
    assert rebuilt_d.parents[0].parents[0] is rebuilt_d.parents[1].parents[0]
    assert emissions[d] == (d, rebuilt_d)


def test_graph_scan_empty_pytree():
    def step(node, parent_carries):
        raise AssertionError("step should not be called")

    carries, emissions = graph_scan([], step)
    assert carries == []
    assert emissions == {}


def test_graph_fold_adds_emissions_by_default():
    a = Node()
    b = Node(a)
    c = Node(a)
    d = Node(b, c)
    node_emissions = {a: 1, b: 2, c: 3, d: 4}

    def step(node, parent_carries):
        return 1 + sum(parent_carries), node_emissions[node]

    assert graph_fold(d, step) == 10


def test_graph_fold_accepts_custom_operator():
    a = Node()
    b = Node(a)

    def step(node, parent_carries):
        carry = 1 + sum(parent_carries)
        return carry, carry

    assert graph_fold(b, step, lambda left, right: left * right) == 2


def test_graph_fold_single_emission_does_not_call_combine():
    root = Node()

    def step(node, parent_carries):
        return 1, 7

    def combine(left, right):
        raise AssertionError("combine should not be called")

    assert graph_fold(root, step, combine) == 7


def test_graph_fold_uses_boundary_carries_but_not_boundary_emissions():
    hidden_root = Node()
    boundary_node = Node(hidden_root)
    first = Node(boundary_node)
    output = Node(first)
    calls = []

    def step(node, parent_carries):
        carry = 1 + sum(parent_carries)
        calls.append((node, parent_carries, carry))
        return carry, carry

    folded = graph_fold(output, step, boundary={boundary_node: 10})

    assert folded == 23
    assert calls == [(first, (10,), 11), (output, (11,), 12)]


def test_graph_fold_rejects_a_boundary_only_graph():
    a = Node()
    b = Node(a)

    def step(node, parent_carries):
        return 1 + sum(parent_carries), 100

    with pytest.raises(ValueError, match="no emissions"):
        graph_fold(b, step, boundary={b: 10})


def test_graph_fold_rejects_an_empty_pytree():
    def step(node, parent_carries):
        raise AssertionError("step should not be called")

    with pytest.raises(ValueError, match="no emissions"):
        graph_fold([], step)


def test_upstream_edge_block():
    a = Node()
    b = Node(a)
    c = Node(a)
    d = Node(b, c)

    def block_d_to_b(child, parent):
        return child is d and parent is b

    assert upstream_nodes(d, edge_block=block_d_to_b) == [a, c, d]


def test_upstream_blockers():
    a = Node()
    b = Node(a)
    c = Node(b)
    d = Node(c)
    assert upstream_nodes(d, lambda n: n in [a]) == [b, c, d]
    assert upstream_nodes(d, lambda n: n in [b]) == [c, d]
    assert upstream_nodes(d, lambda n: n in [c]) == [d]


def test_collider_blockers2():
    a = Node()
    b = Node(a)
    c = Node(a)
    d = Node(b, c)
    assert upstream_nodes(d) == [a, b, c, d]
    assert upstream_nodes(d, lambda n: n in [b]) == [a, c, d]
    assert upstream_nodes(d, lambda n: n in [c]) == [a, b, d]
    assert upstream_nodes(d, lambda n: n in [b, c]) == [d]


def test_children1():
    a = Node()
    b = Node(a)
    c = Node(a)
    d = Node(b, c)
    children = _get_children([d])
    assert set(children[a]) == {b, c}
    assert set(children[b]) == {d}
    assert set(children[c]) == {d}
    assert set(children[d]) == set()


def test_has_second_path():
    a = Node()
    b = Node(a)
    assert not _has_second_path(b, 0)

    a = Node()
    b = Node(a, a)
    assert _has_second_path(b, 0)
    assert _has_second_path(b, 1)

    a = Node()
    b = Node(a)
    c = Node(a, b)
    assert _has_second_path(c, 0)
    assert not _has_second_path(c, 1)
