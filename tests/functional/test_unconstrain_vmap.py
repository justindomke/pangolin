import pytest

from pangolin import dag, functional, ir

import pangolin.functional.unconstraining as unconstraining


def constant(value, interface: ir.RVInterface = ir.NO_INTERFACE) -> ir.RV:
    return ir.RV(ir.Constant(value), interface=interface)


def require_transform(transform: functional.Transform | None) -> functional.Transform:
    assert isinstance(transform, functional.Transform)
    return transform


class TestTransformForVMap:
    def test_none_from_base_policy_returns_none(self, monkeypatch):
        mean = constant([0.0, 1.0])
        scale = constant(1.0)
        op = ir.VMap(ir.Normal(), (0, None))

        monkeypatch.setattr(unconstraining, "transform_for", lambda op, parents: None)
        result = unconstraining.transform_for_vmap(op, (mean, scale))

        assert result is None

    def test_initial_policy_receives_abstract_sliced_parent_shapes(self, monkeypatch):
        lower = constant([[0.0, 1.0], [2.0, 3.0]])
        upper = constant([4.0, 5.0])
        op = ir.VMap(ir.Uniform(), (1, None))
        seen: list[tuple[ir.Shape, ...]] = []

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            seen.append(tuple(parent.shape for parent in parents))
            return functional.tforms.logit

        monkeypatch.setattr(unconstraining, "transform_for", policy)
        unconstraining.transform_for_vmap(op, (lower, upper))

        assert seen[0] == ((2,), (2,))

    def test_forward_maps_base_forward_over_leading_output_axis(self):
        rate = constant([1.0, 2.0, 3.0])
        op = ir.VMap(ir.Exponential(), (0,))
        transform = require_transform(unconstraining.transform_for_vmap(op, (rate,)))

        result = transform.forward(constant([1.0, 2.0, 3.0]))

        assert result.op == ir.VMap(ir.Log(), (0,), 3)
        assert result.shape == (3,)

    def test_inverse_maps_base_inverse_over_leading_output_axis(self):
        rate = constant([1.0, 2.0, 3.0])
        op = ir.VMap(ir.Exponential(), (0,))
        transform = require_transform(unconstraining.transform_for_vmap(op, (rate,)))

        result = transform.inverse(constant([0.0, 1.0, 2.0]))

        assert result.op == ir.VMap(ir.Exp(), (0,), 3)
        assert result.shape == (3,)

    def test_log_determinants_are_mapped_then_summed(self):
        rate = constant([1.0, 2.0, 3.0])
        op = ir.VMap(ir.Exponential(), (0,))
        transform = require_transform(unconstraining.transform_for_vmap(op, (rate,)))
        constrained = constant([1.0, 2.0, 3.0])
        free = transform.forward(constrained)

        result = transform.log_det_jac(constrained, free)

        assert result.op == ir.Sum(0)
        assert isinstance(result.parents[0].op, ir.VMap)
        assert result.parents[0].shape == (3,)
        assert result.shape == ()

    def test_mapped_and_unmapped_parents_are_reused(self):
        lower = constant([-2.0, -1.0, 0.0])
        upper = constant(4.0)
        op = ir.VMap(ir.Uniform(), (0, None))
        transform = require_transform(unconstraining.transform_for_vmap(op, (lower, upper)))

        result = transform.forward(constant([0.0, 1.0, 2.0]))
        nodes = dag.upstream_nodes(result)

        assert lower in nodes
        assert upper in nodes

    def test_all_unmapped_parents_use_explicit_axis_size(self):
        rate = constant(2.0)
        op = ir.VMap(ir.Exponential(), (None,), axis_size=4)
        transform = require_transform(unconstraining.transform_for_vmap(op, (rate,)))

        result = transform.forward(constant([1.0, 2.0, 3.0, 4.0]))

        assert result.op == ir.VMap(ir.Log(), (0,), 4)
        assert result.shape == (4,)

    def test_negative_mapped_axis_is_respected(self):
        nu = constant(3.0)
        scale = constant(
            [
                [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
            ]
        )
        op = ir.VMap(ir.Wishart(), (None, -1))

        transform = unconstraining.transform_for_vmap(op, (nu, scale))

        assert isinstance(transform, functional.Transform)
        assert transform.forward(constant([[[1.0, 0.0], [0.0, 1.0]]] * 3)).shape == (3, 3)

    def test_nested_vmaps_lift_recursively(self):
        rate = constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        op = ir.VMap(ir.VMap(ir.Exponential(), (0,)), (0,))

        transform = require_transform(unconstraining.transform_for_vmap(op, (rate,)))

        assert isinstance(transform, functional.Transform)
        assert transform.forward(constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).shape == (
            2,
            3,
        )

    def test_shape_changing_dirichlet_transform_is_mapped_pointwise(self):
        concentration = constant([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        op = ir.VMap(ir.Dirichlet(), (0,))
        transform = require_transform(unconstraining.transform_for_vmap(op, (concentration,)))
        constrained = constant([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]])

        free = transform.forward(constrained)
        reconstructed = transform.inverse(free)

        assert free.shape == (2, 2)
        assert reconstructed.shape == (2, 3)
        assert transform.log_det_jac(constrained, free).shape == ()

    def test_functions_preserve_interfaces(self):
        interface = ir.RVInterface()
        rate = constant([1.0, 2.0, 3.0], interface)
        op = ir.VMap(ir.Exponential(), (0,))
        transform = require_transform(unconstraining.transform_for_vmap(op, (rate,)))
        constrained = constant([1.0, 2.0, 3.0], interface)
        free = transform.forward(constrained)
        reconstructed = transform.inverse(free)
        log_det = transform.log_det_jac(constrained, free)

        assert free.interface is interface
        assert reconstructed.interface is interface
        assert all(node.interface is interface for node in dag.upstream_nodes(log_det))

    def test_inconsistent_policy_is_rejected_when_forward_is_applied(self, monkeypatch):
        rate = constant([1.0, 2.0])
        op = ir.VMap(ir.Exponential(), (0,))
        calls = 0

        def inconsistent(op: ir.Op, parents: tuple[ir.RV, ...]):
            nonlocal calls
            calls += 1
            return functional.tforms.log if calls == 1 else None

        monkeypatch.setattr(unconstraining, "transform_for", inconsistent)
        transform = require_transform(unconstraining.transform_for_vmap(op, (rate,)))

        with pytest.raises(ValueError, match="inconsistent results"):
            transform.forward(constant([1.0, 2.0]))

    def test_initial_policy_exception_is_propagated(self, monkeypatch):
        rate = constant([1.0, 2.0])
        op = ir.VMap(ir.Exponential(), (0,))

        def reject(op: ir.Op, parents: tuple[ir.RV, ...]):
            raise LookupError("base policy failed")

        with pytest.raises(LookupError, match="base policy failed"):
            monkeypatch.setattr(unconstraining, "transform_for", reject)
            unconstraining.transform_for_vmap(op, (rate,))

    def test_invalid_parent_shape_is_rejected_before_policy_dispatch(self, monkeypatch):
        rate = constant([1.0, 2.0])
        op = ir.VMap(ir.Exponential(), (1,))
        called = False

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            nonlocal called
            called = True
            return functional.tforms.log

        with pytest.raises(ValueError, match="axis"):
            monkeypatch.setattr(unconstraining, "transform_for", policy)
            unconstraining.transform_for_vmap(op, (rate,))

        assert not called
