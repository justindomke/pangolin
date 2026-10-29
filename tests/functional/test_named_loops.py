import numpy as np
import pytest

from pangolin import functional, ir
from pangolin.functional import sfor, vfor
from pangolin.interface import Interface


def test_exports():
    assert functional.vfor is vfor
    assert functional.sfor is sfor


def test_vfor_coordinates_support_the_supplied_interface_syntax():
    pi = Interface(broadcasting="off")
    result = vfor(lambda i, j: i + j, interface=pi, i=2, j=3)
    assert result.shape == (2, 3)
    assert result.interface is pi


def test_sfor_supports_infix_and_indexing_in_the_step():
    pi = Interface(broadcasting="off")
    start = pi.constant([0.0, 0.0])
    result = sfor(
        lambda previous, t, i: pi.normal(previous[i] + t, 1), init=start, interface=pi, t=3, i=2
    )
    assert result.shape == (3, 2)
    assert result.interface is pi


def test_vfor_outer_sum_has_parameter_order_and_coordinate_values():
    interface = ir.RVInterface()
    observed = []

    def fun(i, j):
        observed.extend((i, j))
        return ir.RV(ir.Add(), i, j, interface=interface)

    result = vfor(fun, interface=interface, j=3, i=2)
    assert result.shape == (2, 3)
    assert result.op == ir.VMap(ir.VMap(ir.Add(), (None, 0), 3), (0, None), 2)
    assert [parent.op for parent in result.parents] == [ir.Constant([0, 1]), ir.Constant([0, 1, 2])]
    assert all(rv.shape == () and rv.interface is interface for rv in observed)
    assert result.interface is interface


def test_vfor_preserves_each_output_interface_and_tree():
    first, second, coordinates = ir.RVInterface(), ir.RVInterface(), ir.RVInterface()
    result = vfor(
        lambda i: {
            "a": ir.RV(ir.Exp(), i, interface=first),
            "b": (ir.RV(ir.Log(), i, interface=second),),
        },
        interface=coordinates,
        i=2,
    )
    assert result["a"].interface is first
    assert result["b"][0].interface is second
    assert result["a"].shape == result["b"][0].shape == (2,)


def test_vfor_unused_coordinate():
    result = vfor(lambda i: ir.RV(ir.Constant(7)), interface=ir.NO_INTERFACE, i=3)
    assert result.shape == (3,)


def test_vfor_decorator():
    @vfor(interface=ir.NO_INTERFACE, i=3)
    def result(i):
        return ir.RV(ir.Exp(), i)

    assert result.shape == (3,)


def test_vfor_no_coordinates_calls_directly():
    existing = ir.RV(ir.Constant(1))
    assert vfor(lambda: existing, interface=ir.NO_INTERFACE) is existing


@pytest.mark.parametrize("fun", [lambda i: i, lambda i: captured])
def test_vfor_rejects_existing_outputs(fun):
    with pytest.raises(ValueError, match="created"):
        vfor(fun, interface=ir.NO_INTERFACE, i=2)


captured = ir.RV(ir.Constant(0))


@pytest.mark.parametrize(
    "size, error", [(0, ValueError), (-1, ValueError), (True, TypeError), (2.5, TypeError)]
)
def test_both_reject_invalid_sizes(size, error):
    with pytest.raises(error, match="size"):
        vfor(lambda i: ir.RV(ir.Exp(), i), interface=ir.NO_INTERFACE, i=size)
    with pytest.raises(error, match="size"):
        sfor(
            lambda previous, t: ir.RV(ir.Add(), previous, t),
            init=captured,
            interface=ir.NO_INTERFACE,
            t=size,
        )


def test_numpy_integer_sizes():
    assert vfor(lambda i: ir.RV(ir.Exp(), i), interface=ir.NO_INTERFACE, i=np.int64(2)).shape == (
        2,
    )


@pytest.mark.parametrize("sizes, message", [({}, "missing"), ({"z": 2}, "unknown")])
def test_both_reject_incorrect_axis_names(sizes, message):
    with pytest.raises(ValueError, match=message):
        vfor(lambda i: ir.RV(ir.Exp(), i), interface=ir.NO_INTERFACE, **sizes)
    with pytest.raises(ValueError, match=message):
        sfor(
            lambda previous, t: ir.RV(ir.Add(), previous, t),
            init=captured,
            interface=ir.NO_INTERFACE,
            **sizes,
        )


@pytest.mark.parametrize("fun", [lambda i=2: i, lambda *i: i, lambda *, i: i])
def test_vfor_rejects_non_fixed_signatures(fun):
    with pytest.raises(TypeError, match="fixed positional"):
        vfor(fun, interface=ir.NO_INTERFACE, i=2)


def test_reserved_interface_parameter():
    with pytest.raises(ValueError, match="reserved"):
        vfor(lambda interface: interface, interface=ir.NO_INTERFACE)
    with pytest.raises(ValueError, match="reserved"):
        sfor(lambda previous, interface: previous, init=captured, interface=ir.NO_INTERFACE)


@pytest.mark.parametrize("mapped", [True, False])
def test_vfor_rejects_non_rv_output(mapped):
    with pytest.raises(TypeError, match="RV"):
        if mapped:
            vfor(lambda i: 2, interface=ir.NO_INTERFACE, i=2)
        else:
            vfor(lambda: 2, interface=ir.NO_INTERFACE)


def test_sfor_coordinate_and_carry_interfaces_are_independent():
    carry_interface, coordinate_interface = ir.RVInterface(), ir.RVInterface()
    start = ir.RV(ir.Constant(0), interface=carry_interface)
    observed = []

    def step(previous, t):
        observed.append((previous, t))
        return ir.RV(ir.Add(), previous, t, interface=previous.interface)

    result = sfor(step, init=start, interface=coordinate_interface, t=4)
    assert result.shape == (4,)
    assert result.interface is carry_interface
    assert result.parents[0] is start
    assert result.parents[1].op == ir.Constant([0, 1, 2, 3])
    assert result.parents[1].interface is coordinate_interface
    assert observed[0][0].interface is carry_interface
    assert observed[0][1].interface is coordinate_interface


def test_sfor_parallel_axes_receive_whole_previous_state():
    interface = ir.RVInterface()
    start = ir.RV(ir.Constant(np.zeros((2, 3))), interface=interface)
    observed = []

    def step(previous, t, i, j):
        observed.append((previous, t, i, j))
        value = ir.RV(ir.Index(), previous, i, j, interface=interface)
        return ir.RV(ir.Add(), value, t, interface=interface)

    result = sfor(step, init=start, interface=interface, j=3, t=4, i=2)
    assert result.shape == (4, 2, 3)
    assert isinstance(result.op, ir.Scan)
    assert result.interface is interface
    assert observed[0][0].shape == (2, 3)
    assert all(rv.shape == () for rv in observed[0][1:])
    assert all(rv.interface is interface for rv in observed[0])


def test_sfor_decorator_and_unused_time():
    @sfor(init=captured, interface=ir.NO_INTERFACE, t=3)
    def result(previous, t):
        return ir.RV(ir.Exp(), previous)

    assert result.shape == (3,)
    assert result.parents == (captured,)


def test_sfor_rejects_non_rv_init():
    with pytest.raises(TypeError, match="initial state must be an RV"):
        sfor(lambda previous, t: ir.RV(ir.Exp(), previous), init=0, interface=ir.NO_INTERFACE, t=2)


@pytest.mark.parametrize(
    "fun, error, message",
    [
        (lambda previous, t: previous, ValueError, "created"),
        (lambda previous, t: ir.RV(ir.Exp(), t), ValueError, "depend"),
        (lambda previous, t: (previous, t), TypeError, "exactly one"),
        (
            lambda previous, t: ir.RV(ir.Exp(), previous, interface=ir.RVInterface()),
            ValueError,
            "interface",
        ),
    ],
)
def test_sfor_step_contract(fun, error, message):
    with pytest.raises(error, match=message):
        sfor(fun, init=captured, interface=ir.NO_INTERFACE, t=2)


def test_sfor_rejects_missing_coordinate():
    with pytest.raises(ValueError, match="sequential coordinate"):
        sfor(lambda previous: previous, init=captured, interface=ir.NO_INTERFACE)


def test_sfor_rejects_wrong_state_shape():
    with pytest.raises(ValueError, match="shape"):
        sfor(
            lambda previous, t, i: ir.RV(ir.Add(), previous, i),
            init=captured,
            interface=ir.NO_INTERFACE,
            t=2,
            i=3,
        )
