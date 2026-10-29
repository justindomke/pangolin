"""Truncation selects transforms without evaluating bounds or distribution functions."""

import pytest

from pangolin import dag, functional, interface as pi, ir
from pangolin.functional import unconstraining


def test_two_bounds_do_not_require_a_base_transform(monkeypatch):
    def unexpected(*args):
        raise AssertionError("The base policy must not be consulted")

    monkeypatch.setattr(unconstraining, "transform_for", unexpected)
    lo, hi, mean, scale = [pi.constant(x) for x in (-1.0, 2.0, 0.0, 1.0)]
    transform = unconstraining.transform_for_truncated(
        ir.Truncated(ir.Normal()), (lo, hi, mean, scale)
    )
    assert isinstance(transform, functional.Transform)
    nodes = dag.upstream_nodes(transform.forward(pi.constant(0.5)))
    assert lo in nodes and hi in nodes
    assert mean not in nodes and scale not in nodes


def test_no_bounds_delegate_with_only_base_parents(monkeypatch):
    parents = (pi.constant(2.0),)
    seen = []

    def select(op, args):
        seen.append((op, args))
        return functional.tforms.log

    monkeypatch.setattr(unconstraining, "transform_for", select)
    result = unconstraining.transform_for_truncated(
        ir.Truncated(ir.Exponential(), lower=False, upper=False), parents
    )
    assert result is functional.tforms.log
    assert seen == [(ir.Exponential(), parents)]


def test_no_bounds_can_remain_unconstrained():
    assert (
        functional.transform_for(
            ir.Truncated(ir.Normal(), lower=False, upper=False),
            (pi.constant(0.0), pi.constant(1.0)),
        )
        is None
    )


@pytest.mark.parametrize("lo, hi", [(0.5, None), (None, 2.0)])
def test_one_bound_requires_monotonic_base_transform(lo, hi):
    exp = functional.tforms.exp
    unsigned = functional.Transform(exp.forward, exp.inverse, exp.log_det_jac)
    x = pi.truncated(unsigned(pi.normal), lo, hi)(0.0, 1.0)
    with pytest.raises(NotImplementedError, match="known sign"):
        functional.unconstrain(x, ())


def test_two_bounds_do_not_require_a_monotonic_base_transform():
    exp = functional.tforms.exp
    unsigned = functional.Transform(exp.forward, exp.inverse, exp.log_det_jac)
    x = pi.truncated(unsigned(pi.normal), 0.5, 2.0)(0.0, 1.0)
    reconstructed, _ = functional.unconstrain(x, ())
    assert reconstructed.shape == ()
    assert isinstance(reconstructed.parents[0].op, ir.Bijected)


@pytest.mark.parametrize("lo, hi", [(0.5, None), (None, 2.0), (0.5, 2.0)])
def test_transform_is_deterministic_and_preserves_interface_without_cdf_or_quantile(lo, hi):
    interface = pi.Interface(broadcasting="off")
    x = interface.truncated(interface.tforms.exp(interface.normal), lo, hi)(0.0, 1.0)
    transform = functional.transform_for(x.op, x.parents)
    assert transform is not None
    value = interface.constant(1.0)
    free = transform.forward(value)
    restored = transform.inverse(free)
    log_det = transform.log_det_jac(value, free)
    for node in dag.upstream_nodes((free, restored, log_det)):
        assert not node.op.random
        assert not isinstance(node.op, (ir.CDF, ir.Quantile))
    assert free.interface is restored.interface is log_det.interface is value.interface
    reconstructed, _ = functional.unconstrain(x, ())
    assert reconstructed.interface is x.interface
