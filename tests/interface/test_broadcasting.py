"""Check configured broadcasting through scalar interface functions and RV syntax."""

import numpy as np
import pytest

from pangolin import dag, ir
from pangolin.interface import Interface


@pytest.mark.parametrize("mode", ["off", "simple", "rank", "numpy"])
@pytest.mark.parametrize(
    "shapes,allowed",
    [
        (((), ()), {"off", "simple", "rank", "numpy"}),
        (((3,), ()), {"simple", "rank", "numpy"}),
        (((2, 3), (2, 3)), {"simple", "rank", "numpy"}),
        (((2, 3), (3,)), {"rank", "numpy"}),
        (((2, 1), (3,)), {"numpy"}),
        (((2,), (3,)), set()),
    ],
)
def test_scalar_functions_obey_config(mode, shapes, allowed):
    pi = Interface(broadcasting=mode)
    mean, scale = (np.ones(shape) for shape in shapes)
    if mode not in allowed:
        with pytest.raises(ValueError, match="Can't broadcast shapes"):
            pi.normal(mean, scale)
    else:
        result = pi.normal(mean, scale)
        assert result.shape == np.broadcast_shapes(*shapes)
        assert result.interface is pi


def test_keyword_arguments_keep_parameter_order():
    pi = Interface(broadcasting="simple")
    mean, scale = pi.constant([1, 2, 3]), pi.constant(4)
    result = pi.normal(sigma=scale, mu=mean)
    assert result.op == ir.VMap(ir.Normal(), (0, None), 3)
    assert result.parents == (mean, scale)
    assert result.interface is pi


def test_unary_function_and_infix_use_broadcasting():
    pi = Interface(broadcasting="simple")
    x = pi.constant([1, 2, 3])
    assert pi.exp(x).op == ir.VMap(ir.Exp(), (0,), 3)
    assert (x + 2).op == ir.VMap(ir.Add(), (0, None), 3)
    assert (2 - x).op == ir.VMap(ir.Sub(), (None, 0), 3)
    assert (-x).shape == (3,)
    assert (-x).interface is pi


def test_called_interface_selects_rule_and_output_interface():
    pi, foreign = Interface(broadcasting="numpy"), Interface(broadcasting="off")
    x, y = foreign.constant([[1], [2]]), foreign.constant([3, 4, 5])
    result = pi.add(x, y)
    assert result.shape == (2, 3)
    assert result.interface is pi
    assert x in dag.upstream_nodes(result)
    assert y in dag.upstream_nodes(result)
    assert x.interface is foreign
    assert y.interface is foreign
    with pytest.raises(ValueError, match="broadcasting=OFF"):
        foreign.add(x, y)


def test_scalar_function_reads_config_at_call_time():
    pi = Interface(broadcasting="off")
    normal = pi.normal
    with pytest.raises(ValueError, match="broadcasting=OFF"):
        normal([1, 2, 3], 1)

    with pi.overload(broadcasting="simple"):
        result = normal([1, 2, 3], 1)
        assert result.shape == (3,)
        assert result.interface is pi

    with pytest.raises(ValueError, match="broadcasting=OFF"):
        normal([1, 2, 3], 1)


@pytest.mark.parametrize("mode", ["off", "simple", "rank", "numpy"])
def test_constant_does_not_broadcast(mode):
    pi = Interface(broadcasting=mode)
    x = pi.constant(np.ones((2, 3)))
    assert isinstance(x.op, ir.Constant)
    assert x.shape == (2, 3)
    assert not x.parents
