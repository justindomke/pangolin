from typing import assert_type

from pangolin import functional, ir
from pangolin.util import PyTree


def test_graph_scan_has_plain_ir_types():
    root = ir.RV(ir.Constant(1))
    nodes = {"root": [root]}

    def rebuild(node: ir.RV, parents: tuple[ir.RV, ...]) -> tuple[ir.RV, str]:
        return ir.RV(node.op, *parents), type(node.op).__name__

    replacements, emissions = functional.graph_scan(nodes, rebuild)

    assert_type(replacements, dict[str, list[ir.RV]])
    assert_type(emissions, dict[ir.RV, str])


def test_graph_scan_preserves_an_outer_tuple_of_rv_pytrees():
    root = ir.RV(ir.Constant(1))
    left = {"root": root}
    right = [root]

    def preserve(node: ir.RV, parents: tuple[ir.RV, ...]) -> tuple[ir.RV, None]:
        return node, None

    replacements, emissions = functional.graph_scan((left, right), preserve)

    assert_type(replacements, tuple[dict[str, ir.RV], list[ir.RV]])
    assert_type(emissions, dict[ir.RV, None])


def test_graph_scan_preserves_a_pair_of_flat_lists():
    root = ir.RV(ir.Constant(1))

    def rebuild(node: ir.RV, parents: tuple[ir.RV, ...]) -> tuple[ir.RV, None]:
        return ir.RV(node.op, *parents), None

    (vars, given), _ = functional.graph_scan(([root], [root]), rebuild)

    assert_type(vars, list[ir.RV])
    assert_type(given, list[ir.RV])
    assert vars[0] is given[0]
    assert vars[0] is not root


def test_graph_scan_preserves_a_single_rv_type():
    root = ir.RV(ir.Constant(1))

    def preserve(node: ir.RV, parents: tuple[ir.RV, ...]) -> tuple[ir.RV, None]:
        return node, None

    replacement, _ = functional.graph_scan(root, preserve)

    assert_type(replacement, ir.RV)
    assert replacement is root


def test_graph_scan_accepts_a_general_pytree():
    def preserve(node: ir.RV, parents: tuple[ir.RV, ...]) -> tuple[ir.RV, None]:
        return node, None

    def scan_tree(nodes: PyTree[ir.RV]) -> PyTree[ir.RV]:
        replacements, _ = functional.graph_scan(nodes, preserve)
        assert_type(replacements, PyTree[ir.RV])
        return replacements

    root = ir.RV(ir.Constant(1))
    assert scan_tree({"root": [root]}) == {"root": [root]}
