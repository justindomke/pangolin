"""Check shared exports and the remaining configured graph-building conveniences."""

import numpy as np
import pytest

from pangolin import dag, functional, ir
from pangolin import interface as pi


@pytest.mark.parametrize(
    "name",
    ["composite", "vmap", "scan", "Transform", "compose_transforms"],
)
def test_generic_exports_are_the_original_objects(name):
    original = getattr(functional, name)
    assert name in pi.__all__
    assert getattr(pi, name) is original
    assert getattr(pi.Interface(), name) is original


@pytest.mark.parametrize("name", ["upstream_nodes", "print_upstream", "graph_upstream"])
def test_graph_inspection_exports_are_the_original_objects(name):
    original = dag.upstream_nodes if name == "upstream_nodes" else getattr(ir, name)
    assert name in pi.__all__
    assert getattr(pi, name) is original
    assert getattr(pi.Interface(), name) is original


def test_new_conveniences_are_public():
    assert {"RVLike", "sqrt", "expit", "sigmoid", "fill_tril", "extract_tril"} <= set(pi.__all__)
    for name in pi.__all__:
        assert hasattr(pi, name)


@pytest.mark.parametrize("combinator", ["composite", "vmap", "scan", "exp"])
def test_generic_exports_do_not_impose_the_accessing_interface(combinator):
    caller = pi.Interface(broadcasting="off")
    producer = pi.Interface(broadcasting="simple")
    if combinator == "composite":
        result = caller.composite(producer.normal)(0, 1)
    elif combinator == "vmap":
        result = caller.vmap(producer.normal)(producer.constant([0, 1]), producer.constant([1, 1]))
    elif combinator == "scan":
        result = caller.scan(lambda carry: producer.normal(carry, 1), 3)(producer.constant(0))
    else:
        result = caller.tforms.exp(producer.normal)(0, 1)
    assert result.interface is producer


@pytest.mark.parametrize("name", ["sqrt", "expit", "sigmoid"])
@pytest.mark.parametrize("mode", ["off", "simple", "rank", "numpy"])
def test_scalar_conveniences_use_the_called_interface_and_broadcasting(name, mode):
    interface = pi.Interface(broadcasting=mode)
    foreign = pi.Interface().constant([1, 4])
    fun = getattr(interface, name)
    if mode == "off":
        with pytest.raises(ValueError):
            fun(foreign)
        value = foreign.interface.constant(4)
    else:
        value = foreign
    result = fun(value)
    expected = interface.pow(value, 0.5) if name == "sqrt" else interface.inv_logit(value)
    assert ir.rv_equal(result, expected)
    assert result.interface is interface
    assert result.parents[0] is value


@pytest.mark.parametrize("name", ["sqrt", "expit", "sigmoid"])
def test_saved_conveniences_observe_overloads(name):
    interface = pi.Interface(broadcasting="off")
    fun = getattr(interface, name)
    with interface.overload(broadcasting="simple"):
        assert fun([1, 4]).shape == (2,)
    with pytest.raises(ValueError):
        fun([1, 4])


@pytest.mark.parametrize("mode", ["off", "simple", "rank", "numpy"])
@pytest.mark.parametrize(
    "name,values",
    [("fill_tril", [1, 2, 3]), ("extract_tril", [[1, 0], [2, 3]])],
)
def test_triangular_helpers_reuse_functional_implementation_and_preserve_input(mode, name, values):
    interface = pi.Interface(broadcasting=mode)
    foreign = pi.Interface().constant(values)
    result = getattr(interface, name)(foreign)
    expected = getattr(functional.tforms, name).forward(foreign)
    assert ir.rv_equal(result, expected)
    assert result.interface is interface
    assert foreign in dag.upstream_nodes([result])
    for node in dag.upstream_nodes([result]):
        if node is not foreign:
            assert node.interface is interface
    assert interface._effective_config.broadcasting == mode


@pytest.mark.parametrize(
    "name,values",
    [
        ("fill_tril", 1),
        ("fill_tril", [1, 2]),
        ("extract_tril", [1]),
        ("extract_tril", np.ones((2, 3))),
    ],
)
def test_triangular_helpers_reject_invalid_shapes(name, values):
    with pytest.raises(ValueError):
        getattr(pi, name)(values)


def test_default_conveniences_use_the_default_interface():
    for value in (
        pi.sqrt(x=4),
        pi.expit(a=0),
        pi.sigmoid(a=0),
        pi.fill_tril(params=[1, 2, 3]),
        pi.extract_tril(L=[[1, 0], [2, 3]]),
    ):
        assert value.interface is pi.default
