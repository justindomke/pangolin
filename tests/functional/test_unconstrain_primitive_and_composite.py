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


def constant(value) -> ir.RV:
    return ir.RV(ir.Constant(value))


class TestTransformForPrimitive:
    def test_inventory_includes_every_continuous_primitive(self):
        discovered = {
            value
            for name in ir.__all__
            if isinstance((value := getattr(ir, name)), type)
            and issubclass(value, ir.Op)
            and value not in (ir.Bijected, ir.Truncated)
            and getattr(value, "random", None) is True
            and getattr(value, "discrete", None) is False
        }

        assert discovered == {
            ir.Beta,
            ir.Cauchy,
            ir.Dirichlet,
            ir.Exponential,
            ir.Gamma,
            ir.Lognormal,
            ir.MultiNormal,
            ir.Normal,
            ir.NormalPrec,
            ir.StudentT,
            ir.Uniform,
            ir.Wishart,
        }

    def test_cauchy_is_already_unconstrained(self):
        assert (
            unconstraining.transform_for_primitive(ir.Cauchy(), (constant(0.0), constant(1.0)))
            is None
        )

    def test_multi_normal_is_already_unconstrained(self):
        assert (
            unconstraining.transform_for_primitive(
                ir.MultiNormal(), (constant([0.0, 0.0]), constant([[1.0, 0.0], [0.0, 1.0]]))
            )
            is None
        )

    def test_normal_is_already_unconstrained(self):
        assert (
            unconstraining.transform_for_primitive(ir.Normal(), (constant(0.0), constant(1.0)))
            is None
        )

    def test_normal_prec_is_already_unconstrained(self):
        assert (
            unconstraining.transform_for_primitive(ir.NormalPrec(), (constant(0.0), constant(1.0)))
            is None
        )

    def test_student_t_is_already_unconstrained(self):
        assert (
            unconstraining.transform_for_primitive(
                ir.StudentT(), (constant(3.0), constant(0.0), constant(1.0))
            )
            is None
        )

    def test_exponential_uses_log(self):
        assert (
            unconstraining.transform_for_primitive(ir.Exponential(), (constant(1.0),))
            is functional.tforms.log
        )

    def test_gamma_uses_log(self):
        assert (
            unconstraining.transform_for_primitive(ir.Gamma(), (constant(2.0), constant(3.0)))
            is functional.tforms.log
        )

    def test_lognormal_uses_log(self):
        assert (
            unconstraining.transform_for_primitive(ir.Lognormal(), (constant(0.0), constant(1.0)))
            is functional.tforms.log
        )

    def test_beta_uses_logit(self):
        assert (
            unconstraining.transform_for_primitive(ir.Beta(), (constant(2.0), constant(3.0)))
            is functional.tforms.logit
        )

    def test_dirichlet_uses_softmax_centered_ilr(self):
        assert (
            unconstraining.transform_for_primitive(ir.Dirichlet(), (constant([1.0, 1.0, 1.0]),))
            is functional.tforms.softmax_centered_ilr
        )

    def test_uniform_uses_its_exact_parent_objects_as_bounds(self):
        lower = constant(-2.0)
        upper = constant(3.0)

        transform = unconstraining.transform_for_primitive(ir.Uniform(), (lower, upper))
        assert isinstance(transform, functional.Transform)
        result = transform.forward(constant(0.0))

        assert lower in dag.upstream_nodes(result)
        assert upper in dag.upstream_nodes(result)

    def test_wishart_uses_spd_transform(self):
        assert (
            unconstraining.transform_for_primitive(
                ir.Wishart(), (constant(3.0), constant([[1.0, 0.0], [0.0, 1.0]]))
            )
            is functional.tforms.unconstrain_spd
        )

    def test_unsupported_continuous_primitive_raises(self):
        with pytest.raises(NotImplementedError, match="UnsupportedContinuous"):
            unconstraining.transform_for_primitive(UnsupportedContinuous(), ())

    def test_discrete_primitive_has_no_private_fallback(self):
        with pytest.raises(NotImplementedError, match="Bernoulli"):
            unconstraining.transform_for_primitive(ir.Bernoulli(), (constant(0.5),))


class TestTransformForComposite:
    def test_single_constrained_op_uses_recursive_policy(self):
        rate = constant(2.0)
        op = ir.Composite(1, (ir.Exponential(),), ((0,),))

        result = unconstraining.transform_for_composite(op, (rate,))

        assert result is functional.tforms.log

    def test_single_free_op_returns_none(self):
        mean = constant(0.0)
        scale = constant(1.0)
        op = ir.Composite(2, (ir.Normal(),), ((0, 1),))

        result = unconstraining.transform_for_composite(op, (mean, scale))

        assert result is None

    def test_computed_parent_is_passed_to_final_policy(self, monkeypatch):
        lower = constant(-2.0)
        width = constant(5.0)
        op = ir.Composite(2, (ir.Add(), ir.Uniform()), ((0, 1), (0, 2)))
        seen: list[tuple[ir.RV, ...]] = []

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            seen.append(parents)
            return functional.tforms.logit

        monkeypatch.setattr(unconstraining, "transform_for", policy)
        unconstraining.transform_for_composite(op, (lower, width))

        assert seen[0][0] is lower
        assert seen[0][1].op == ir.Add()
        assert seen[0][1].parents == (lower, width)

    def test_multiple_intermediates_are_materialized_in_order(self, monkeypatch):
        left = constant(2.0)
        right = constant(3.0)
        op = ir.Composite(
            2,
            (ir.Add(), ir.Mul(), ir.Exponential()),
            ((0, 1), (2, 1), (3,)),
        )
        seen: list[tuple[ir.RV, ...]] = []

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            seen.append(parents)
            return functional.tforms.log

        monkeypatch.setattr(unconstraining, "transform_for", policy)
        unconstraining.transform_for_composite(op, (left, right))

        final_parent = seen[0][0]
        assert final_parent.op == ir.Mul()
        assert final_parent.parents[0].op == ir.Add()
        assert final_parent.parents[0].parents == (left, right)
        assert final_parent.parents[1] is right

    def test_repeated_parent_references_preserve_identity(self, monkeypatch):
        value = constant(2.0)
        op = ir.Composite(1, (ir.Mul(), ir.Exponential()), ((0, 0), (1,)))
        seen: list[tuple[ir.RV, ...]] = []

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            seen.append(parents)
            return functional.tforms.log

        monkeypatch.setattr(unconstraining, "transform_for", policy)
        unconstraining.transform_for_composite(op, (value,))

        squared = seen[0][0]
        assert squared.parents == (value, value)
        assert squared.parents[0] is squared.parents[1]

    def test_final_parent_order_follows_parent_numbers(self, monkeypatch):
        lower = constant(-2.0)
        upper = constant(3.0)
        op = ir.Composite(2, (ir.Uniform(),), ((1, 0),))
        seen: list[tuple[ir.RV, ...]] = []

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            seen.append(parents)
            return functional.tforms.logit

        monkeypatch.setattr(unconstraining, "transform_for", policy)
        unconstraining.transform_for_composite(op, (lower, upper))

        assert seen == [(upper, lower)]

    def test_recursive_policy_is_called_exactly_once(self, monkeypatch):
        rate = constant(2.0)
        op = ir.Composite(1, (ir.Identity(), ir.Exponential()), ((0,), (1,)))
        calls: list[ir.Op] = []

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            calls.append(op)
            return functional.tforms.log

        monkeypatch.setattr(unconstraining, "transform_for", policy)
        unconstraining.transform_for_composite(op, (rate,))

        assert calls == [ir.Exponential()]

    def test_recursive_policy_exception_is_propagated(self, monkeypatch):
        rate = constant(2.0)
        op = ir.Composite(1, (ir.Exponential(),), ((0,),))

        def reject(op: ir.Op, parents: tuple[ir.RV, ...]):
            raise LookupError("policy rejected final op")

        with pytest.raises(LookupError, match="policy rejected final op"):
            monkeypatch.setattr(unconstraining, "transform_for", reject)
            unconstraining.transform_for_composite(op, (rate,))

    def test_nested_final_composite_is_given_back_to_recursive_policy(self, monkeypatch):
        rate = constant(2.0)
        inner = ir.Composite(1, (ir.Exponential(),), ((0,),))
        outer = ir.Composite(1, (inner,), ((0,),))
        seen: list[ir.Op] = []

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            seen.append(op)
            return functional.tforms.log

        monkeypatch.setattr(unconstraining, "transform_for", policy)
        unconstraining.transform_for_composite(outer, (rate,))

        assert seen == [inner]

    def test_intermediate_nodes_preserve_parent_interface(self, monkeypatch):
        interface = ir.RVInterface()
        lower = ir.RV(ir.Constant(-2.0), interface=interface)
        width = ir.RV(ir.Constant(5.0), interface=interface)
        op = ir.Composite(2, (ir.Add(), ir.Uniform()), ((0, 1), (0, 2)))
        seen: list[tuple[ir.RV, ...]] = []

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            seen.append(parents)
            return functional.tforms.logit

        monkeypatch.setattr(unconstraining, "transform_for", policy)
        unconstraining.transform_for_composite(op, (lower, width))

        assert seen[0][1].interface is interface

    def test_zero_input_composite_materializes_internal_constants(self, monkeypatch):
        op = ir.Composite(0, (ir.Constant(2.0), ir.Exponential()), ((), (0,)))
        seen: list[tuple[ir.RV, ...]] = []

        def policy(op: ir.Op, parents: tuple[ir.RV, ...]):
            seen.append(parents)
            return functional.tforms.log

        monkeypatch.setattr(unconstraining, "transform_for", policy)
        result = unconstraining.transform_for_composite(op, ())

        assert result is functional.tforms.log
        assert seen[0][0].op == ir.Constant(2.0)
        assert seen[0][0].parents == ()
