from pangolin import interface as pi, ir
from pangolin.ir import RV


from typing import assert_type

from pangolin.interface import constant
from pangolin.interface import normal
from pangolin.interface import upstream_nodes


def test_parents_and_upstream_nodes_preserve_interface_rv_type():
    mean = constant(0)
    scale = constant(1)
    x = normal(mean, scale)

    assert_type(x.parents, tuple[RV, ...])
    result = upstream_nodes(x)
    assert_type(result, list[RV])
    assert result == [mean, scale, x]


def test_supports_pytrees_and_blocking_predicates():
    root = constant(1)
    middle = RV(ir.Exp(), root, interface=pi.default)
    output = RV(ir.Exp(), middle, interface=pi.default)

    assert upstream_nodes({"output": output}) == [root, middle, output]
    assert upstream_nodes(output, node_block=lambda rv: rv is middle) == [output]
    assert upstream_nodes(output, edge_block=lambda child, parent: parent is middle) == [output]


def test_canonical_function_also_accepts_base_ir_rvs():
    base_rv = ir.RV(ir.Constant(1))

    result = upstream_nodes(base_rv)

    assert_type(result, list[ir.RV])
    assert result == [base_rv]
