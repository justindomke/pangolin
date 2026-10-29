from typing import assert_type

from pangolin import dag, ir


class ShapedNode(dag.Node):
    shape = ()


def test_returns_rvs_in_topological_order():
    mean = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))
    x = ir.RV(ir.Normal(), mean, scale)

    result = dag.upstream_nodes(x)

    assert_type(result, list[ir.RV])
    assert result == [mean, scale, x]


def test_supports_pytrees_and_blocking_predicates():
    root = ir.RV(ir.Constant(1))
    middle = ir.RV(ir.Exp(), root)
    output = ir.RV(ir.Exp(), middle)

    assert dag.upstream_nodes({"output": output}) == [root, middle, output]
    assert dag.upstream_nodes(output, node_block=lambda rv: rv is middle) == [output]
    assert dag.upstream_nodes(output, edge_block=lambda child, parent: parent is middle) == [output]


def test_preserves_a_custom_node_subclass():
    root = ShapedNode()
    output = ShapedNode(root)

    result = dag.upstream_nodes(output)

    assert_type(result, list[ShapedNode])
    assert result == [root, output]
