import pytest

from pangolin import ir
from pangolin import interface as default
from pangolin.interface import Interface


def test_default_exports_bound_methods():
    assert default.vfor.__self__ is default.default
    assert default.sfor.__self__ is default.default
    assert {"vfor", "sfor"} <= set(default.__all__)


@pytest.mark.parametrize("decorator", [False, True])
def test_vfor_coordinates_use_called_interface_and_outputs_keep_theirs(decorator):
    pi, other = Interface(), Interface()
    seen = []

    def body(i, j):
        seen.extend((i, j))
        return {"local": i + j, "foreign": (other.add(i, j), ir.RV(ir.Add(), i, j))}

    output = pi.vfor(i=2, j=3)(body) if decorator else pi.vfor(body, j=3, i=2)
    assert all(rv.interface is pi and rv.shape == () for rv in seen)
    assert output["local"].interface is pi
    assert output["foreign"][0].interface is other
    assert output["foreign"][1].interface is ir.NO_INTERFACE
    assert output["local"].shape == output["foreign"][0].shape == (2, 3)


@pytest.mark.parametrize("decorator", [False, True])
def test_sfor_coerces_initial_value_and_attaches_coordinates(decorator):
    pi = Interface()
    seen = []

    def body(previous, t):
        seen.append((previous, t))
        return pi.normal(previous + t, 1)

    output = pi.sfor(init=0, t=4)(body) if decorator else pi.sfor(body, init=0, t=4)
    assert output.interface is pi
    assert output.shape == (4,)
    assert output.parents[0].op == ir.Constant(0)
    assert output.parents[0].interface is pi
    assert all(rv.interface is pi for rv in seen[0])


@pytest.mark.parametrize("decorator", [False, True])
def test_sfor_preserves_foreign_initial_rv_and_output(decorator):
    pi, other = Interface(), Interface()
    start = other.constant(0)
    seen = []

    def body(previous, t):
        seen.append((previous, t))
        return other.add(previous, t)

    output = pi.sfor(init=start, t=4)(body) if decorator else pi.sfor(body, init=start, t=4)
    assert output.interface is other
    assert output.parents[0] is start
    assert seen[0][0].interface is other
    assert seen[0][1].interface is pi


def test_sfor_accepts_no_interface_carry_and_output():
    pi = Interface()
    start = ir.RV(ir.Constant(0))
    output = pi.sfor(lambda previous, t: ir.RV(ir.Add(), previous, t), init=start, t=3)
    assert output.interface is ir.NO_INTERFACE
    assert output.parents[0] is start


def test_sfor_parallel_coordinates_and_array_initial_value():
    pi = Interface()
    seen = []

    def body(previous, t, i, j):
        seen.append((previous, t, i, j))
        return pi.normal(previous[i, j] + t, 1)

    output = pi.sfor(body, init=[[0, 0, 0], [0, 0, 0]], t=4, j=3, i=2)
    assert output.shape == (4, 2, 3)
    assert output.interface is pi
    assert seen[0][0].shape == (2, 3)
    assert all(rv.interface is pi for rv in seen[0])


def test_sfor_still_rejects_carry_interface_change():
    pi, other = Interface(), Interface()
    with pytest.raises(ValueError, match="output interface must match"):
        pi.sfor(lambda previous, t: other.normal(previous, 1), init=0, t=3)


def test_vfor_saved_decorator_observes_current_configuration():
    pi = Interface(broadcasting="off")
    make = pi.vfor(i=2)
    with pi.overload(broadcasting="simple"):
        output = make(lambda i: pi.add(i, [1, 2, 3]))
    assert output.shape == (2, 3)
    assert output.interface is pi
    assert pi._effective_config.broadcasting == "off"


def test_nested_wrappers():
    pi = Interface()
    output = pi.vfor(lambda i: pi.sfor(lambda previous, t: previous + i + t, init=0, t=3), i=2)
    assert output.shape == (2, 3)
    assert output.interface is pi


def test_vfor_inherits_existing_output_rejection():
    pi = Interface()
    with pytest.raises(ValueError, match="created"):
        pi.vfor(lambda i: i, i=3)


def test_sfor_inherits_missing_size_rejection():
    pi = Interface()
    with pytest.raises(ValueError, match="missing size"):
        pi.sfor(lambda previous, t: previous + t, init=0)
