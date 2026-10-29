from pangolin import functional
import pytest

from pangolin import dag, ir
from pangolin.functional import tforms
from pangolin.ir import RV
from pangolin.functional import unconstrain


@pytest.mark.parametrize(
    ("op", "parent_values"),
    [
        (ir.Cauchy(), (0.0, 1.0)),
        (ir.MultiNormal(), ([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])),
        (ir.Normal(), (0.0, 1.0)),
        (ir.NormalPrec(), (0.0, 1.0)),
        (ir.StudentT(), (2.0, 0.0, 1.0)),
    ],
)
def test_free_primitives_are_returned_unchanged(op, parent_values):
    parents = tuple(ir.RV(ir.Constant(value)) for value in parent_values)
    var = ir.RV(op, *parents)

    assert functional.unconstrain(var, ())[0].op == var.op


@pytest.mark.parametrize(
    ("op", "parent_values", "n_biject_params"),
    [
        (ir.Exponential(), (1.0,), 0),
        (ir.Gamma(), (2.0, 1.0), 0),
        (ir.Lognormal(), (0.0, 1.0), 0),
        (ir.Beta(), (2.0, 3.0), 0),
        (ir.Uniform(), (1.0, 5.0), 2),
    ],
)
def test_easy_constrained_primitives_use_only_captured_transform_parameters(
    op, parent_values, n_biject_params
):
    parents = tuple(ir.RV(ir.Constant(value)) for value in parent_values)
    var = ir.RV(op, *parents)

    reconstructed = functional.unconstrain(var, ())[0]
    free = reconstructed.parents[0]

    assert isinstance(free.op, ir.Bijected)
    assert free.op.base_op == op
    assert free.op.bijector.n_biject_params == n_biject_params
    assert tuple(p.op for p in free.parents[n_biject_params:]) == tuple(p.op for p in parents)


def test_positive_primitive_reuses_the_functional_log_transform():
    rate = ir.RV(ir.Constant(2.0))
    reconstructed = functional.unconstrain(ir.RV(ir.Exponential(), rate), ())[0]
    free = reconstructed.parents[0]
    assert isinstance(free.op, ir.Bijected)

    interface_rate = RV(ir.Constant(2.0))

    def interface_exponential() -> RV:
        return RV(ir.Exponential(), interface_rate)

    expected = tforms.log(interface_exponential)()
    assert isinstance(expected.op, ir.Bijected)
    assert free.op.bijector == expected.op.bijector


def test_uniform_reuses_the_functional_scaled_logit_transform():
    lower = ir.RV(ir.Constant(1.0))
    upper = ir.RV(ir.Constant(5.0))
    reconstructed = functional.unconstrain(ir.RV(ir.Uniform(), lower, upper), ())[0]
    free = reconstructed.parents[0]
    assert isinstance(free.op, ir.Bijected)

    interface_lower = RV(ir.Constant(1.0))
    interface_upper = RV(ir.Constant(5.0))

    def interface_uniform() -> RV:
        return RV(ir.Uniform(), interface_lower, interface_upper)

    expected = tforms.scaled_logit(interface_lower, interface_upper)(interface_uniform)()
    assert isinstance(expected.op, ir.Bijected)
    assert free.op.bijector == expected.op.bijector


def test_unconstrain_rematerializes_without_leaking_interface_rvs():
    rate = ir.RV(ir.Constant(2.0))
    x = ir.RV(ir.Exponential(), rate)
    y = ir.RV(ir.Add(), x, x)

    new_y, new_given = unconstrain(y, ())

    assert new_given == ()
    assert isinstance(new_y, ir.RV)
    reconstructed = new_y.parents[0]
    assert reconstructed is new_y.parents[1]
    assert not reconstructed.op.random

    free = reconstructed.parents[0]
    assert isinstance(free.op, ir.Bijected)
    assert free.op.base_op == ir.Exponential()
    assert len(free.parents) == 1
    assert free.parents[0].op == ir.Constant(2.0)
    assert all(type(var) is ir.RV for var in dag.upstream_nodes(new_y))


def test_unconstrain_keeps_given_random_nodes_constrained():
    rate = ir.RV(ir.Constant(2.0))
    observed = ir.RV(ir.Exponential(), rate)

    new_vars, new_given = unconstrain(observed, observed)

    assert new_vars is new_given
    assert isinstance(new_given, ir.RV)
    assert new_given.op == ir.Exponential()
    assert new_given.op.random


def test_unconstrain_leaves_free_latents_random():
    zero = ir.RV(ir.Constant(0.0))
    one = ir.RV(ir.Constant(1.0))
    x = ir.RV(ir.Normal(), zero, one)

    new_x, _ = unconstrain(x, ())

    assert isinstance(new_x, ir.RV)
    assert new_x.op == ir.Normal()
    assert new_x.op.random


def test_unconstrain_accepts_a_custom_transform_selector():
    rate = ir.RV(ir.Constant(2.0))
    x = ir.RV(ir.Exponential(), rate)
    seen = []

    def leave_unchanged(op, parents):
        seen.append((op, parents))
        return None

    new_x, _ = unconstrain(x, (), transform_for=leave_unchanged)

    assert isinstance(new_x, ir.RV)
    assert new_x.op == ir.Exponential()
    assert seen == [(new_x.op, new_x.parents)]
