import numpy as np
import pytest

from pangolin import dag, functional, ir

import pangolin.functional.unconstraining as unconstraining


def constant(value, interface: ir.RVInterface = ir.NO_INTERFACE) -> ir.RV:
    return ir.RV(ir.Constant(value), interface=interface)


def require_transform(transform: functional.Transform | None) -> functional.Transform:
    assert isinstance(transform, functional.Transform)
    return transform


class TestLaggedCarries:
    def test_scalar_sequence_prepends_start_and_drops_final_value(self):
        start = constant(0.0)
        values = constant([1.0, 2.0, 3.0])

        result = unconstraining.lagged_carries(start, values, 3)

        assert result.op == ir.Concatenate()
        assert result.shape == (3,)
        expanded_start, prefix = result.parents
        assert expanded_start.op == ir.VMap(ir.Identity(), (None,), 1)
        assert expanded_start.parents == (start,)
        assert prefix.op == ir.VMap(ir.Index(), (None, 0), 2)
        assert prefix.parents[0] is values
        first_index = prefix.parents[1].op
        assert isinstance(first_index, ir.Constant)
        np.testing.assert_array_equal(first_index.value, [0, 1])

    def test_array_carry_retains_all_carry_dimensions(self):
        start = constant([0.0, 1.0])
        values = constant([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])

        result = unconstraining.lagged_carries(start, values, 3)

        assert result.shape == (3, 2)
        prefix = result.parents[1]
        assert prefix.shape == (2, 2)
        assert prefix.op == ir.VMap(
            ir.VMap(ir.Index(), (None, None, 0), 2), (None, 0, None), 2
        )
        row_index = prefix.parents[1].op
        column_index = prefix.parents[2].op
        assert isinstance(row_index, ir.Constant)
        assert isinstance(column_index, ir.Constant)
        np.testing.assert_array_equal(row_index.value, [0, 1])
        np.testing.assert_array_equal(column_index.value, [0, 1])

    def test_length_one_only_expands_the_start(self):
        start = constant([0.0, 1.0])
        values = constant([[2.0, 3.0]])

        result = unconstraining.lagged_carries(start, values, 1)

        assert result.op == ir.VMap(ir.Identity(), (None,), 1)
        assert result.parents == (start,)
        assert result.shape == (1, 2)

    def test_values_must_have_the_scan_output_shape(self):
        with pytest.raises(ValueError, match="Scan values must have shape"):
            unconstraining.lagged_carries(constant(0.0), constant([1.0, 2.0]), 3)

    def test_preserves_interfaces(self):
        interface = ir.RVInterface()
        start = constant(0.0, interface)
        values = constant([1.0, 2.0, 3.0], interface)

        result = unconstraining.lagged_carries(start, values, 3)

        assert all(node.interface is interface for node in dag.upstream_nodes(result))


class TestTransformForScan:
    def test_none_from_base_policy_returns_none(self, monkeypatch):
        start = constant(0.0)
        scale = constant(1.0)
        op = ir.Scan(ir.Normal(), 3, (None,))

        monkeypatch.setattr(unconstraining, "transform_for", lambda op, parents: None)
        result = unconstraining.transform_for_scan(op, (start, scale))

        assert result is None

    def test_initial_policy_receives_abstract_carry_and_sliced_inputs(self, monkeypatch):
        start = constant(0.0)
        scale = constant([1.0, 2.0, 3.0])
        op = ir.Scan(ir.Normal(), 3, (0,))
        seen: list[tuple[ir.Shape, ...]] = []

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            seen.append(tuple(parent.shape for parent in parents))
            return None

        monkeypatch.setattr(unconstraining, "transform_for", policy)
        unconstraining.transform_for_scan(op, (start, scale))

        assert seen == [((), ())]

    def test_forward_maps_base_forward_over_scan_outputs(self):
        start = constant(2.0)
        op = ir.Scan(ir.Exponential(), 3, ())
        transform = require_transform(unconstraining.transform_for_scan(op, (start,)))

        result = transform.forward(constant([1.0, 2.0, 3.0]))

        assert result.op == ir.VMap(ir.Log(), (0,), 3)
        assert result.shape == (3,)

    def test_inverse_maps_base_inverse_over_scan_outputs(self):
        start = constant(2.0)
        op = ir.Scan(ir.Exponential(), 3, ())
        transform = require_transform(unconstraining.transform_for_scan(op, (start,)))

        result = transform.inverse(constant([0.0, 1.0, 2.0]))

        assert result.op == ir.VMap(ir.Exp(), (0,), 3)
        assert result.shape == (3,)

    def test_log_determinants_are_mapped_then_summed(self):
        start = constant(2.0)
        op = ir.Scan(ir.Exponential(), 3, ())
        transform = require_transform(unconstraining.transform_for_scan(op, (start,)))
        constrained = constant([1.0, 2.0, 3.0])
        free = transform.forward(constrained)

        result = transform.log_det_jac(constrained, free)

        assert result.op == ir.Sum(0)
        assert isinstance(result.parents[0].op, ir.VMap)
        assert result.parents[0].shape == (3,)
        assert result.shape == ()

    def test_transform_may_capture_a_mapped_noncarry_input(self, monkeypatch):
        start = constant(0.0)
        offsets = constant([1.0, 2.0, 3.0])
        op = ir.Scan(ir.Normal(), 3, (0,))

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            return functional.tforms.shift_by(parents[1])

        monkeypatch.setattr(unconstraining, "transform_for", policy)
        transform = require_transform(unconstraining.transform_for_scan(op, (start, offsets)))
        result = transform.forward(constant([4.0, 5.0, 6.0]))

        assert result.op == ir.VMap(ir.Add(), (0, 0), 3)
        assert offsets in dag.upstream_nodes(result)

    def test_transform_may_capture_an_unmapped_noncarry_input(self, monkeypatch):
        start = constant(0.0)
        offset = constant(1.0)
        op = ir.Scan(ir.Normal(), 3, (None,))

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            return functional.tforms.shift_by(parents[1])

        monkeypatch.setattr(unconstraining, "transform_for", policy)
        transform = require_transform(unconstraining.transform_for_scan(op, (start, offset)))
        result = transform.forward(constant([4.0, 5.0, 6.0]))

        assert result.op == ir.VMap(ir.Add(), (0, None), 3)
        assert offset in dag.upstream_nodes(result)

    def test_shape_changing_dirichlet_transform_is_applied_to_scan_outputs(self):
        start = constant([1.0, 2.0, 3.0])
        op = ir.Scan(ir.Dirichlet(), 2, ())
        transform = require_transform(unconstraining.transform_for_scan(op, (start,)))
        constrained = constant([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]])

        free = transform.forward(constrained)
        reconstructed = transform.inverse(free)

        assert free.shape == (2, 2)
        assert reconstructed.shape == (2, 3)
        assert transform.log_det_jac(constrained, free).shape == ()

    def test_direct_carry_dependency_uses_lagged_forward_and_recursive_inverse(self, monkeypatch):
        start = constant(0.0)
        scale = constant(1.0)
        op = ir.Scan(ir.Normal(), 3, (None,))

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            return functional.tforms.shift_by(parents[0])

        monkeypatch.setattr(unconstraining, "transform_for", policy)
        transform = require_transform(unconstraining.transform_for_scan(op, (start, scale)))
        constrained = constant([1.0, 2.0, 3.0])

        free = transform.forward(constrained)
        reconstructed = transform.inverse(free)

        assert free.op == ir.VMap(ir.Add(), (0, 0), 3)
        lagged = free.parents[1]
        assert lagged.op == ir.Concatenate()
        assert lagged.parents[0].parents == (start,)
        assert lagged.parents[1].parents[0] is constrained
        assert isinstance(reconstructed.op, ir.Scan)
        assert not reconstructed.op.random
        assert reconstructed.op.in_axes == (0,)
        assert reconstructed.parents == (start, free)

    def test_transitive_carry_dependency_uses_recursive_inverse(self, monkeypatch):
        start = constant(0.0)
        scale = constant(1.0)
        op = ir.Scan(ir.Normal(), 3, (None,))

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            derived_offset = type(parents[0])(ir.Exp(), parents[0])
            return functional.tforms.shift_by(derived_offset)

        monkeypatch.setattr(unconstraining, "transform_for", policy)
        transform = require_transform(unconstraining.transform_for_scan(op, (start, scale)))
        constrained = constant([1.0, 2.0, 3.0])

        free = transform.forward(constrained)
        reconstructed = transform.inverse(free)

        assert free.op == ir.VMap(ir.Add(), (0, 0), 3)
        assert free.parents[1].op == ir.VMap(ir.Exp(), (0,), 3)
        assert free.parents[1].parents[0].op == ir.Concatenate()
        assert isinstance(reconstructed.op, ir.Scan)
        assert reconstructed.op.base_op == ir.Composite(
            2,
            (ir.Exp(), ir.Sub()),
            ((0,), (1, 2)),
        )

    def test_default_uniform_transform_depends_on_carry(self):
        start = constant(0.0)
        upper = constant(10.0)
        op = ir.Scan(ir.Uniform(), 3, (None,))
        transform = require_transform(unconstraining.transform_for_scan(op, (start, upper)))
        constrained = constant([1.0, 2.0, 3.0])

        free = transform.forward(constrained)
        reconstructed = transform.inverse(free)
        log_det = transform.log_det_jac(constrained, free)

        assert free.shape == (3,)
        assert any(isinstance(node.op, ir.Concatenate) for node in dag.upstream_nodes(free))
        assert isinstance(reconstructed.op, ir.Scan)
        assert reconstructed.op.in_axes == (0, None)
        assert reconstructed.parents == (start, free, upper)
        assert log_det.op == ir.Sum(0)
        assert any(isinstance(node.op, ir.Concatenate) for node in dag.upstream_nodes(log_det))

    def test_carry_dependent_inverse_accepts_a_mapped_input(self):
        start = constant(0.0)
        upper = constant([10.0, 11.0, 12.0])
        op = ir.Scan(ir.Uniform(), 3, (0,))
        transform = require_transform(unconstraining.transform_for_scan(op, (start, upper)))

        free = transform.forward(constant([1.0, 2.0, 3.0]))
        reconstructed = transform.inverse(free)

        assert isinstance(reconstructed.op, ir.Scan)
        assert reconstructed.op.in_axes == (0, 0)
        assert reconstructed.parents == (start, free, upper)

    def test_carry_dependent_length_one_uses_only_the_expanded_start(self, monkeypatch):
        start = constant(0.0)
        scale = constant(1.0)
        op = ir.Scan(ir.Normal(), 1, (None,))

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            return functional.tforms.shift_by(parents[0])

        monkeypatch.setattr(unconstraining, "transform_for", policy)
        transform = require_transform(unconstraining.transform_for_scan(op, (start, scale)))
        free = transform.forward(constant([1.0]))
        reconstructed = transform.inverse(free)

        assert free.parents[1].op == ir.VMap(ir.Identity(), (None,), 1)
        assert isinstance(reconstructed.op, ir.Scan)
        assert reconstructed.op.length == 1

    def test_carry_dependent_transform_may_change_each_value_shape(self, monkeypatch):
        start = constant([0.2, 0.3, 0.5])
        op = ir.Scan(ir.Dirichlet(), 2, ())

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            return functional.compose_transforms(
                [
                    functional.tforms.shift_by(parents[0]),
                    functional.tforms.softmax_centered_ilr,
                ]
            )

        monkeypatch.setattr(unconstraining, "transform_for", policy)
        transform = require_transform(unconstraining.transform_for_scan(op, (start,)))
        constrained = constant([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]])

        free = transform.forward(constrained)
        reconstructed = transform.inverse(free)

        assert free.shape == (2, 2)
        assert isinstance(reconstructed.op, ir.Scan)
        assert reconstructed.shape == (2, 3)
        assert reconstructed.op.in_axes == (0,)

    def test_default_unconstrain_reconstructs_a_carry_dependent_scan(self):
        start = constant(0.0)
        upper = constant(10.0)
        original = ir.RV(ir.Scan(ir.Uniform(), 3, (None,)), start, upper)

        reconstructed = functional.unconstrain(original, ())[0]

        assert reconstructed.shape == original.shape
        assert isinstance(reconstructed.op, ir.Composite)
        transformed = reconstructed.parents[0]
        assert isinstance(transformed.op, ir.Bijected)
        assert isinstance(transformed.op.base_op, ir.Scan)
        forward = transformed.op.bijector.forward
        inverse = transformed.op.bijector.inverse
        assert isinstance(forward, ir.Composite)
        assert isinstance(inverse, ir.Composite)
        assert any(isinstance(op, ir.Concatenate) for op in forward.ops)
        assert any(isinstance(op, ir.Scan) for op in inverse.ops)

    def test_initial_policy_exception_is_propagated(self, monkeypatch):
        start = constant(2.0)
        op = ir.Scan(ir.Exponential(), 3, ())

        def reject(op: ir.Op, parents: tuple[ir.RV, ...]):
            raise LookupError("base policy failed")

        with pytest.raises(LookupError, match="base policy failed"):
            monkeypatch.setattr(unconstraining, "transform_for", reject)
            unconstraining.transform_for_scan(op, (start,))

    def test_functions_preserve_interfaces(self):
        interface = ir.RVInterface()
        start = constant(2.0, interface)
        op = ir.Scan(ir.Exponential(), 3, ())
        transform = require_transform(unconstraining.transform_for_scan(op, (start,)))
        constrained = constant([1.0, 2.0, 3.0], interface)
        free = transform.forward(constrained)
        reconstructed = transform.inverse(free)
        log_det = transform.log_det_jac(constrained, free)

        assert free.interface is interface
        assert reconstructed.interface is interface
        assert all(node.interface is interface for node in dag.upstream_nodes(log_det))

    def test_invalid_mapped_input_shape_is_rejected_before_policy_dispatch(self, monkeypatch):
        start = constant(0.0)
        scale = constant([1.0, 2.0])
        op = ir.Scan(ir.Normal(), 3, (0,))
        called = False

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            nonlocal called
            called = True
            return None

        with pytest.raises(ValueError, match="length"):
            monkeypatch.setattr(unconstraining, "transform_for", policy)
            unconstraining.transform_for_scan(op, (start, scale))

        assert not called
