import pytest

from pangolin import dag, functional, ir

import pangolin.functional.unconstraining as unconstraining


class UnsupportedContinuous(ir.Op):
    random: bool = True
    """True"""
    discrete: bool = False
    """False"""

    def get_shape(self) -> ir.Shape:
        return ()


def constant(value, interface: ir.RVInterface = ir.NO_INTERFACE) -> ir.RV:
    return ir.RV(ir.Constant(value), interface=interface)


def normal(mean: ir.RV, scale: ir.RV) -> ir.RV:
    return ir.RV(ir.Normal(), mean, scale, interface=mean.interface)


class TestTransformFor:
    def test_validates_parent_shapes_before_dispatch(self, monkeypatch):
        called = False

        def primitive(op, parents):
            nonlocal called
            called = True
            return None

        monkeypatch.setattr(unconstraining, "transform_for_primitive", primitive)

        with pytest.raises(ValueError, match="scalar"):
            unconstraining.transform_for(ir.Normal(), (constant([0.0, 1.0]), constant([1.0, 1.0])))

        assert not called

    def test_rejects_deterministic_ops(self):
        with pytest.raises(ValueError, match="Cannot unconstrain deterministic Op"):
            unconstraining.transform_for(ir.Exp(), (constant(1.0),))

    def test_discrete_ops_are_rejected(self):
        with pytest.raises(NotImplementedError, match="discrete"):
            unconstraining.transform_for(ir.Bernoulli(), (constant(0.5),))

    def test_discrete_composite_is_rejected_without_handler(self, monkeypatch):
        op = ir.Composite(1, (ir.Bernoulli(),), ((0,),))

        def fail(*args):
            raise AssertionError("composite handler should not be called")

        monkeypatch.setattr(unconstraining, "transform_for_composite", fail)

        with pytest.raises(NotImplementedError, match="discrete"):
            unconstraining.transform_for(op, (constant(0.5),))

    def test_dispatches_primitive_with_exact_parents(self, monkeypatch):
        rate = constant(2.0)
        seen: list[object] = []

        def primitive(op, parents):
            seen.extend((op, parents))
            return functional.tforms.log

        monkeypatch.setattr(unconstraining, "transform_for_primitive", primitive)

        result = unconstraining.transform_for(ir.Exponential(), (rate,))

        assert result is functional.tforms.log
        assert seen == [ir.Exponential(), (rate,)]

    def test_dispatches_composite_with_exact_parents(self, monkeypatch):
        rate = constant(2.0)
        op = ir.Composite(1, (ir.Exponential(),), ((0,),))
        seen: list[object] = []

        def composite(op, parents):
            seen.extend((op, parents))
            return functional.tforms.log

        monkeypatch.setattr(unconstraining, "transform_for_composite", composite)

        result = unconstraining.transform_for(op, (rate,))

        assert result is functional.tforms.log
        assert seen == [op, (rate,)]

    def test_dispatches_vmap_with_exact_parents(self, monkeypatch):
        rate = constant([1.0, 2.0])
        op = ir.VMap(ir.Exponential(), (0,))
        seen: list[object] = []

        def vmap_handler(op, parents):
            seen.extend((op, parents))
            return functional.tforms.log

        monkeypatch.setattr(unconstraining, "transform_for_vmap", vmap_handler)

        result = unconstraining.transform_for(op, (rate,))

        assert result is functional.tforms.log
        assert seen == [op, (rate,)]

    def test_dispatches_scan_with_exact_parents(self, monkeypatch):
        start = constant(2.0)
        op = ir.Scan(ir.Exponential(), 3, ())
        seen: list[object] = []

        def scan_handler(op, parents):
            seen.extend((op, parents))
            return functional.tforms.log

        monkeypatch.setattr(unconstraining, "transform_for_scan", scan_handler)

        result = unconstraining.transform_for(op, (start,))

        assert result is functional.tforms.log
        assert seen == [op, (start,)]

    def test_dispatches_transformed_with_exact_parents(self, monkeypatch):
        transformed = functional.tforms.exp(normal)(constant(0.0), constant(1.0))
        seen: list[object] = []

        def transformed_handler(op, parents):
            seen.extend((op, parents))
            return functional.tforms.log

        monkeypatch.setattr(unconstraining, "transform_for_transformed", transformed_handler)

        result = unconstraining.transform_for(transformed.op, transformed.parents)

        assert result is functional.tforms.log
        assert seen == [transformed.op, transformed.parents]

    def test_unsupported_continuous_primitive_error_is_propagated(self):
        with pytest.raises(NotImplementedError, match="UnsupportedContinuous"):
            unconstraining.transform_for(UnsupportedContinuous(), ())

    def test_handler_exception_is_propagated(self, monkeypatch):
        rate = constant(2.0)

        def reject(op, parents):
            raise LookupError("primitive handler failed")

        monkeypatch.setattr(unconstraining, "transform_for_primitive", reject)

        with pytest.raises(LookupError, match="primitive handler failed"):
            unconstraining.transform_for(ir.Exponential(), (rate,))


class TestDefaultUnconstrain:
    def test_already_free_continuous_rv_is_returned_unchanged(self):
        value = normal(constant(0.0), constant(1.0))

        assert functional.unconstrain(value, ())[0].op == value.op

    def test_discrete_rv_is_rejected(self):
        value = ir.RV(ir.Bernoulli(), constant(0.5))

        with pytest.raises(NotImplementedError, match="Cannot unconstrain discrete Op bernoulli"):
            functional.unconstrain(value, ())[0]

    def test_positive_rv_becomes_a_deterministic_reconstruction(self):
        value = ir.RV(ir.Exponential(), constant(2.0))

        reconstructed = functional.unconstrain(value, ())[0]

        assert not reconstructed.op.random
        assert reconstructed.op == ir.Exp()

    def test_free_random_parent_uses_transformed_original_distribution(self):
        rate = constant(2.0)
        value = ir.RV(ir.Exponential(), rate)

        reconstructed = functional.unconstrain(value, ())[0]
        free = reconstructed.parents[0]

        assert isinstance(free.op, ir.Bijected)
        assert free.op.base_op == value.op
        assert len(free.parents) == 1
        assert free.parents[0].op == rate.op

    def test_reconstruction_preserves_original_shape(self):
        nu = constant(3.0)
        scale = constant([[1.0, 0.0], [0.0, 1.0]])
        value = ir.RV(ir.Wishart(), nu, scale)

        reconstructed = functional.unconstrain(value, ())[0]

        assert reconstructed.shape == (2, 2)
        assert reconstructed.parents[0].shape == (3,)

    def test_parameterized_transform_captures_exact_bounds(self):
        lower = constant(-2.0)
        upper = constant(3.0)
        value = ir.RV(ir.Uniform(), lower, upper)

        reconstructed = functional.unconstrain(value, ())[0]
        free = reconstructed.parents[0]

        new_lower, new_upper = free.parents[:2]
        assert new_lower.op == lower.op and new_upper.op == upper.op
        assert free.parents == (new_lower, new_upper, new_lower, new_upper)
        assert reconstructed.parents == (free, new_lower, new_upper)

    def test_composite_distribution_is_wrapped_as_a_whole(self):
        rate = constant(2.0)
        op = ir.Composite(1, (ir.Identity(), ir.Exponential()), ((0,), (1,)))
        value = ir.RV(op, rate)

        reconstructed = functional.unconstrain(value, ())[0]

        free_op = reconstructed.parents[0].op
        assert isinstance(free_op, ir.Bijected)
        assert free_op.base_op == op

    def test_vmap_distribution_is_wrapped_as_a_whole(self):
        rate = constant([1.0, 2.0, 3.0])
        op = ir.VMap(ir.Exponential(), (0,))
        value = ir.RV(op, rate)

        reconstructed = functional.unconstrain(value, ())[0]

        free_op = reconstructed.parents[0].op
        assert isinstance(free_op, ir.Bijected)
        assert free_op.base_op == op
        assert reconstructed.shape == value.shape

    def test_scan_distribution_is_wrapped_as_a_whole(self):
        start = constant(2.0)
        op = ir.Scan(ir.Exponential(), 3, ())
        value = ir.RV(op, start)

        reconstructed = functional.unconstrain(value, ())[0]

        free_op = reconstructed.parents[0].op
        assert isinstance(free_op, ir.Bijected)
        assert free_op.base_op == op
        assert reconstructed.shape == value.shape

    def test_existing_transformed_distribution_is_wrapped_as_a_whole(self):
        value = functional.tforms.exp(normal)(constant(0.0), constant(1.0))

        reconstructed = functional.unconstrain(value, ())[0]

        assert isinstance(reconstructed.parents[0].op, ir.Bijected)
        assert reconstructed.parents[0].op.base_op == value.op

    def test_interface_is_preserved_throughout_new_graph(self):
        interface = ir.RVInterface()
        rate = ir.RV(ir.Constant(2.0), interface=interface)
        value = ir.RV(ir.Exponential(), rate, interface=interface)

        reconstructed = functional.unconstrain(value, ())[0]

        assert all(node.interface is interface for node in dag.upstream_nodes(reconstructed))

    def test_unsupported_continuous_distribution_raises(self):
        value = ir.RV(UnsupportedContinuous())

        with pytest.raises(NotImplementedError, match="UnsupportedContinuous"):
            functional.unconstrain(value, ())[0]

    def test_dirichlet_is_reconstructed_from_fewer_free_coordinates(self):
        concentration = constant([1.0, 2.0, 3.0])
        value = ir.RV(ir.Dirichlet(), concentration)

        reconstructed = functional.unconstrain(value, ())[0]
        free = reconstructed.parents[0]

        assert reconstructed.shape == (3,)
        assert free.shape == (2,)
        assert isinstance(free.op, ir.Bijected)
        assert free.op.base_op == ir.Dirichlet()
        assert len(free.parents) == 1
        assert free.parents[0].op == concentration.op

    def test_input_graph_is_not_modified(self):
        rate = constant(2.0)
        value = ir.RV(ir.Exponential(), rate)
        original_op = value.op
        original_parents = value.parents

        functional.unconstrain(value, ())[0]

        assert value.op is original_op
        assert value.parents is original_parents
