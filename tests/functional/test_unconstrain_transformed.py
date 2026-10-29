import pytest

from pangolin import dag, functional, ir

import pangolin.functional.unconstraining as unconstraining


def constant(value, interface: ir.RVInterface = ir.NO_INTERFACE) -> ir.RV:
    return ir.RV(ir.Constant(value), interface=interface)


def normal(mean: ir.RV, scale: ir.RV) -> ir.RV:
    return ir.RV(ir.Normal(), mean, scale, interface=mean.interface)


def exponential(rate: ir.RV) -> ir.RV:
    return ir.RV(ir.Exponential(), rate, interface=rate.interface)


def exp_transformed_normal(interface: ir.RVInterface = ir.NO_INTERFACE) -> ir.RV:
    mean = constant(0.0, interface)
    scale = constant(1.0, interface)
    return functional.tforms.exp(normal)(mean, scale)


def require_transform(transform: functional.Transform | None) -> functional.Transform:
    assert isinstance(transform, functional.Transform)
    return transform


def require_transformed_op(value: ir.RV) -> ir.Bijected:
    assert isinstance(value.op, ir.Bijected)
    return value.op


class TestTransformForTransformed:
    def test_policy_receives_only_distribution_parents_without_bijector_parameters(
        self, monkeypatch
    ):
        transformed = exp_transformed_normal()
        seen: list[tuple[ir.RV, ...]] = []

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            seen.append(parents)
            return None

        op = require_transformed_op(transformed)
        monkeypatch.setattr(unconstraining, "transform_for", policy)
        unconstraining.transform_for_transformed(op, transformed.parents)

        assert seen == [transformed.parents]

    def test_policy_receives_only_distribution_parents_with_bijector_parameters(self, monkeypatch):
        output_lower = constant(-5.0)
        output_upper = constant(5.0)
        base_lower = constant(1.0)
        base_upper = constant(3.0)

        def uniform(lower: ir.RV, upper: ir.RV) -> ir.RV:
            return ir.RV(ir.Uniform(), lower, upper)

        transformed = functional.tforms.scaled_logit(output_lower, output_upper)(uniform)(
            base_lower, base_upper
        )
        seen: list[tuple[ir.RV, ...]] = []

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            seen.append(parents)
            return functional.tforms.logit

        op = require_transformed_op(transformed)
        monkeypatch.setattr(unconstraining, "transform_for", policy)
        unconstraining.transform_for_transformed(op, transformed.parents)

        assert seen == [(base_lower, base_upper)]

    def test_policy_receives_the_exact_base_op(self, monkeypatch):
        transformed = exp_transformed_normal()
        seen: list[ir.Op] = []

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            seen.append(op)
            return None

        op = require_transformed_op(transformed)
        monkeypatch.setattr(unconstraining, "transform_for", policy)
        unconstraining.transform_for_transformed(op, transformed.parents)

        assert seen == [ir.Normal()]

    def test_already_free_base_is_unconstrained_by_stored_inverse(self, monkeypatch):
        transformed = exp_transformed_normal()

        op = require_transformed_op(transformed)
        monkeypatch.setattr(unconstraining, "transform_for", lambda op, parents: None)
        transform = require_transform(
            unconstraining.transform_for_transformed(op, transformed.parents)
        )
        value = constant(2.0)

        assert transform.forward(value).op == op.bijector.inverse

    def test_already_free_base_is_reconstructed_by_stored_forward(self, monkeypatch):
        transformed = exp_transformed_normal()

        op = require_transformed_op(transformed)
        monkeypatch.setattr(unconstraining, "transform_for", lambda op, parents: None)
        transform = require_transform(
            unconstraining.transform_for_transformed(op, transformed.parents)
        )
        value = constant(2.0)

        assert transform.inverse(value).op == op.bijector.forward

    def test_already_free_base_negates_stored_log_determinant(self, monkeypatch):
        transformed = exp_transformed_normal()
        op = require_transformed_op(transformed)
        monkeypatch.setattr(unconstraining, "transform_for", lambda op, parents: None)
        transform = require_transform(
            unconstraining.transform_for_transformed(op, transformed.parents)
        )
        constrained = constant(2.0)
        free = transform.forward(constrained)

        log_det = transform.log_det_jac(constrained, free)

        assert log_det.op == ir.Mul()
        assert log_det.parents[0].op == ir.Constant(-1)
        assert log_det.parents[1].op == op.bijector.log_det_jac

    def test_constrained_base_applies_stored_inverse_before_base_forward(self, monkeypatch):
        transformed = functional.tforms.exp(exponential)(constant(2.0))

        op = require_transformed_op(transformed)
        monkeypatch.setattr(
            unconstraining, "transform_for", lambda op, parents: functional.tforms.log
        )
        transform = require_transform(
            unconstraining.transform_for_transformed(
                op,
                transformed.parents,
            )
        )
        constrained = constant(3.0)
        free = transform.forward(constrained)

        assert free.op == ir.Log()
        assert free.parents[0].op == op.bijector.inverse
        assert free.parents[0].parents == (constrained,)

    def test_constrained_base_applies_base_inverse_before_stored_forward(self, monkeypatch):
        transformed = functional.tforms.exp(exponential)(constant(2.0))
        op = require_transformed_op(transformed)
        monkeypatch.setattr(
            unconstraining, "transform_for", lambda op, parents: functional.tforms.log
        )
        transform = require_transform(
            unconstraining.transform_for_transformed(
                op,
                transformed.parents,
            )
        )
        free = constant(0.0)
        constrained = transform.inverse(free)

        assert constrained.op == op.bijector.forward
        assert constrained.parents[0].op == ir.Exp()
        assert constrained.parents[0].parents == (free,)

    def test_parameterized_stored_transform_reuses_exact_bijector_parents(self, monkeypatch):
        output_lower = constant(-5.0)
        output_upper = constant(5.0)
        base_lower = constant(1.0)
        base_upper = constant(3.0)

        def uniform(lower: ir.RV, upper: ir.RV) -> ir.RV:
            return ir.RV(ir.Uniform(), lower, upper)

        transformed = functional.tforms.scaled_logit(output_lower, output_upper)(uniform)(
            base_lower, base_upper
        )
        op = require_transformed_op(transformed)
        monkeypatch.setattr(
            unconstraining, "transform_for", lambda op, parents: functional.tforms.logit
        )
        transform = require_transform(
            unconstraining.transform_for_transformed(
                op,
                transformed.parents,
            )
        )

        nodes = dag.upstream_nodes(transform.forward(constant(0.0)))

        assert output_lower in nodes
        assert output_upper in nodes
        assert base_lower not in nodes
        assert base_upper not in nodes

    def test_recursive_policy_exception_is_propagated(self, monkeypatch):
        transformed = exp_transformed_normal()

        def reject(op: ir.Op, parents: tuple[ir.RV, ...]):
            raise LookupError("base policy failed")

        op = require_transformed_op(transformed)
        with pytest.raises(LookupError, match="base policy failed"):
            monkeypatch.setattr(unconstraining, "transform_for", reject)
            unconstraining.transform_for_transformed(op, transformed.parents)

    def test_transform_functions_preserve_interfaces(self, monkeypatch):
        interface = ir.RVInterface()
        transformed = exp_transformed_normal(interface)
        op = require_transformed_op(transformed)
        monkeypatch.setattr(unconstraining, "transform_for", lambda op, parents: None)
        transform = require_transform(
            unconstraining.transform_for_transformed(op, transformed.parents)
        )
        constrained = constant(2.0, interface)
        free = transform.forward(constrained)
        reconstructed = transform.inverse(free)
        log_det = transform.log_det_jac(constrained, free)

        assert free.interface is interface
        assert reconstructed.interface is interface
        assert all(node.interface is interface for node in dag.upstream_nodes(log_det))
