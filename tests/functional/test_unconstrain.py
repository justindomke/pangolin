import pytest

from pangolin import dag, functional, ir

import pangolin.functional.unconstraining as unconstraining


def normal(mean: ir.RV, scale: ir.RV) -> ir.RV:
    return ir.RV(ir.Normal(), mean, scale)


def test_public_api_contains_unconstraining_tools():
    assert unconstraining.__all__ == ["transform_for", "unconstrain"]
    assert functional.unconstrain is unconstraining.unconstrain
    assert functional.transform_for is unconstraining.transform_for


def test_unconstrain_rebuilds_deterministic_nodes_without_calling_the_policy():
    x = ir.RV(ir.Constant(1.0))
    y = ir.RV(ir.Exp(), x)

    new_y, new_given = functional.unconstrain(y, ())

    assert isinstance(new_y, ir.RV)
    assert new_given == ()
    assert new_y is not y
    assert new_y.op == ir.Exp()
    assert new_y.parents[0] is not x
    assert new_y.parents[0].op == ir.Constant(1.0)


def test_unconstrain_calls_the_policy_once_per_random_node_in_parent_first_order():
    zero = ir.RV(ir.Constant(0.0))
    one = ir.RV(ir.Constant(1.0))
    x = normal(zero, one)
    y = normal(x, one)
    output = ir.RV(ir.Add(), y, x)
    seen: list[tuple[ir.Op, tuple[ir.RV, ...]]] = []

    def preserve(op: ir.Op, parents: tuple[ir.RV, ...]):
        seen.append((op, parents))
        return None

    new_output, _ = functional.unconstrain(output, (), transform_for=preserve)

    assert isinstance(new_output, ir.RV)
    new_y, new_x = new_output.parents
    assert seen == [(new_x.op, new_x.parents), (new_y.op, new_y.parents)]
    assert new_y.parents[0] is new_x
    assert new_output.parents == (new_y, new_x)


def test_unconstrain_skips_given_nodes_but_rewrites_their_latent_ancestors():
    zero = ir.RV(ir.Constant(0.0))
    one = ir.RV(ir.Constant(1.0))
    latent = normal(zero, one)
    observed = normal(latent, one)
    seen: list[tuple[ir.Op, tuple[ir.RV, ...]]] = []

    def preserve(op: ir.Op, parents: tuple[ir.RV, ...]):
        seen.append((op, parents))
        return None

    new_vars, new_given = functional.unconstrain(
        {"latent": latent, "observed": observed},
        (observed,),
        transform_for=preserve,
    )

    assert isinstance(new_vars, dict)
    assert isinstance(new_given, tuple)
    new_latent = new_vars["latent"]
    new_observed = new_vars["observed"]
    given_observed = new_given[0]
    assert isinstance(new_latent, ir.RV)
    assert isinstance(new_observed, ir.RV)
    assert isinstance(given_observed, ir.RV)
    assert seen == [(new_latent.op, new_latent.parents)]
    assert new_observed is given_observed
    assert given_observed.op == ir.Normal()
    assert given_observed.parents[0] is new_latent


def test_unconstrain_preserves_tree_structure_and_shared_nodes():
    zero = ir.RV(ir.Constant(0.0))
    one = ir.RV(ir.Constant(1.0))
    x = normal(zero, one)

    new_vars, new_given = functional.unconstrain(
        {"left": [x], "right": (x,)},
        (),
        transform_for=lambda op, parents: None,
    )

    assert isinstance(new_vars, dict)
    assert isinstance(new_vars["left"], list)
    assert isinstance(new_vars["right"], tuple)
    assert new_vars["left"][0] is new_vars["right"][0]
    assert new_given == ()


def test_custom_selector_may_leave_discrete_nodes_unchanged():
    probability = ir.RV(ir.Constant(0.5))
    choice = ir.RV(ir.Bernoulli(), probability)
    result, _ = functional.unconstrain(choice, (), transform_for=lambda op, parents: None)
    assert result.op == ir.Bernoulli()


def test_unconstrain_does_not_revisit_nodes_introduced_by_transform():
    x = normal(ir.RV(ir.Constant(0.0)), ir.RV(ir.Constant(1.0)))
    seen = []

    def select(op, parents):
        seen.append(op)
        return functional.tforms.scale_by(ir.RV(ir.Constant(2.0)))

    result, _ = functional.unconstrain(x, (), transform_for=select)
    assert seen == [ir.Normal()]
    assert not result.op.random
    assert isinstance(result.parents[0].op, ir.Bijected)


def test_unconstrain_preserves_interfaces_while_rebuilding():
    interface = ir.RVInterface()
    zero = ir.RV(ir.Constant(0.0), interface=interface)
    one = ir.RV(ir.Constant(1.0), interface=interface)
    x = ir.RV(ir.Normal(), zero, one, interface=interface)

    new_x, _ = functional.unconstrain(x, (), transform_for=lambda op, parents: None)

    assert isinstance(new_x, ir.RV)
    assert all(var.interface is interface for var in dag.upstream_nodes(new_x))


def test_unconstrain_rejects_non_rv_input_leaves():
    with pytest.raises(TypeError, match="unconstrain vars must be a PyTree whose leaves are RV"):
        functional.unconstrain({"bad": 1}, ())  # type: ignore


@pytest.mark.parametrize("result", [1, ir.RV(ir.Constant(1.0)), "log"])
def test_unconstrain_rejects_invalid_selector_results(result):
    x = normal(ir.RV(ir.Constant(0.0)), ir.RV(ir.Constant(1.0)))
    with pytest.raises(TypeError, match="transform_for must return a Transform or None"):
        functional.unconstrain(x, (), transform_for=lambda op, parents: result)


def test_unconstrain_propagates_policy_exceptions():
    zero = ir.RV(ir.Constant(0.0))
    one = ir.RV(ir.Constant(1.0))
    x = normal(zero, one)

    def reject(op, parents):
        raise LookupError("no rewrite rule")

    with pytest.raises(LookupError, match="no rewrite rule"):
        functional.unconstrain(x, (), transform_for=reject)


def test_default_policy_leaves_unconstrained_random_nodes_unchanged():
    zero = ir.RV(ir.Constant(0.0))
    one = ir.RV(ir.Constant(1.0))
    x = normal(zero, one)

    new_x, _ = functional.unconstrain(x, ())

    assert isinstance(new_x, ir.RV)
    assert new_x is not x
    assert new_x.op == ir.Normal()


def test_default_policy_rejects_discrete_latent_nodes():
    probability = ir.RV(ir.Constant(0.5))
    choice = ir.RV(ir.Bernoulli(), probability)

    with pytest.raises(NotImplementedError, match="Cannot unconstrain discrete Op bernoulli"):
        functional.unconstrain(choice, ())


def test_default_policy_allows_a_discrete_given_node():
    probability = ir.RV(ir.Constant(0.5))
    choice = ir.RV(ir.Bernoulli(), probability)

    new_choice, new_given = functional.unconstrain(choice, choice)

    assert new_choice is new_given
    assert isinstance(new_choice, ir.RV)
    assert new_choice.op == ir.Bernoulli()


def test_transform_for_returns_primitive_transforms():
    zero = ir.RV(ir.Constant(0.0))
    one = ir.RV(ir.Constant(1.0))

    assert unconstraining.transform_for(ir.Normal(), (zero, one)) is None
    assert unconstraining.transform_for(ir.Exponential(), (one,)) is functional.tforms.log
    assert unconstraining.transform_for(ir.Beta(), (one, one)) is functional.tforms.logit
    assert (
        unconstraining.transform_for(ir.Dirichlet(), (ir.RV(ir.Constant([1.0, 1.0])),))
        is functional.tforms.softmax_centered_ilr
    )
    assert (
        unconstraining.transform_for(ir.Wishart(), (one, ir.RV(ir.Constant([[1.0]]))))
        is functional.tforms.unconstrain_spd
    )


def test_transform_for_constructs_parameterized_transforms_from_parents():
    lower = ir.RV(ir.Constant(-2.0))
    upper = ir.RV(ir.Constant(3.0))

    transform = unconstraining.transform_for(ir.Uniform(), (lower, upper))

    assert isinstance(transform, functional.Transform)
    value = ir.RV(ir.Constant(0.0))
    forward_nodes = dag.upstream_nodes(transform.forward(value))
    assert lower in forward_nodes
    assert upper in forward_nodes


def test_transform_for_composite_resolves_its_final_applied_op(monkeypatch):
    lower = ir.RV(ir.Constant(-2.0))
    width = ir.RV(ir.Constant(5.0))
    op = ir.Composite(
        2,
        (ir.Add(), ir.Uniform()),
        ((0, 1), (0, 2)),
    )
    seen_ops: list[ir.Op] = []
    seen_parents: list[tuple[ir.RV, ...]] = []

    def resolve(
        final_op: ir.Op,
        final_parents: tuple[ir.RV, ...],
    ) -> functional.Transform | None:
        seen_ops.append(final_op)
        seen_parents.append(final_parents)
        return functional.tforms.logit

    monkeypatch.setattr(unconstraining, "transform_for", resolve)
    transform = unconstraining.transform_for_composite(op, (lower, width))

    assert transform is functional.tforms.logit
    assert seen_ops == [ir.Uniform()]
    (final_parents,) = seen_parents
    assert final_parents[0] is lower
    computed_upper = final_parents[1]
    assert isinstance(computed_upper, ir.RV)
    assert computed_upper.op == ir.Add()
    assert computed_upper.parents == (lower, width)


def test_transform_for_recurses_through_nested_composites():
    rate = ir.RV(ir.Constant(2.0))
    inner = ir.Composite(1, (ir.Exponential(),), ((0,),))
    outer = ir.Composite(1, (inner,), ((0,),))

    transform = unconstraining.transform_for(outer, (rate,))

    assert transform is functional.tforms.log


def test_unconstrain_uses_computed_composite_bounds():
    lower = ir.RV(ir.Constant(-2.0))
    width = ir.RV(ir.Constant(5.0))
    op = ir.Composite(
        2,
        (ir.Add(), ir.Uniform()),
        ((0, 1), (0, 2)),
    )
    x = ir.RV(op, lower, width)

    reconstructed = functional.unconstrain(x, ())[0]

    free = reconstructed.parents[0]
    assert isinstance(free.op, ir.Bijected)
    assert free.op.base_op == op
    assert free.op.bijector.n_biject_params == 2
    captured_lower, captured_upper = free.parents[:2]
    new_lower, new_width = free.parents[2:]
    assert new_lower.op == lower.op and new_width.op == width.op
    assert captured_lower is new_lower
    assert captured_upper.op == ir.Add()
    assert captured_upper.parents == (new_lower, new_width)


def test_transform_for_transformed_undoes_the_stored_bijection():
    mean = ir.RV(ir.Constant(0.0))
    scale = ir.RV(ir.Constant(1.0))
    transformed = functional.tforms.exp(normal)(mean, scale)
    assert isinstance(transformed.op, ir.Bijected)

    transform = unconstraining.transform_for(transformed.op, transformed.parents)

    assert isinstance(transform, functional.Transform)
    y = ir.RV(ir.Constant(2.0))
    z = transform.forward(y)
    reconstructed = transform.inverse(z)
    assert z.op == transformed.op.bijector.inverse
    assert z.parents == (y,)
    assert reconstructed.op == transformed.op.bijector.forward
    assert reconstructed.parents == (z,)


def test_transform_for_transformed_then_unconstrains_the_base():
    mean = ir.RV(ir.Constant(0.0))
    scale = ir.RV(ir.Constant(1.0))

    def lognormal(mean: ir.RV, scale: ir.RV) -> ir.RV:
        return ir.RV(ir.Lognormal(), mean, scale)

    transformed = functional.tforms.log(lognormal)(mean, scale)
    assert isinstance(transformed.op, ir.Bijected)

    transform = unconstraining.transform_for(transformed.op, transformed.parents)

    assert isinstance(transform, functional.Transform)
    y = ir.RV(ir.Constant(0.5))
    z = transform.forward(y)
    assert z.op == ir.Log()
    restored_base = z.parents[0]
    assert restored_base.op == transformed.op.bijector.inverse
    assert restored_base.parents == (y,)


def test_transform_for_transformed_splits_bijector_and_distribution_parents():
    output_lower = ir.RV(ir.Constant(1.0))
    output_upper = ir.RV(ir.Constant(3.0))
    distribution_lower = ir.RV(ir.Constant(1.0))
    distribution_upper = ir.RV(ir.Constant(3.0))

    def uniform(lower: ir.RV, upper: ir.RV) -> ir.RV:
        return ir.RV(ir.Uniform(), lower, upper)

    transformed = functional.tforms.scaled_logit(output_lower, output_upper)(uniform)(
        distribution_lower,
        distribution_upper,
    )
    assert isinstance(transformed.op, ir.Bijected)

    transform = unconstraining.transform_for(transformed.op, transformed.parents)

    assert isinstance(transform, functional.Transform)
    y = ir.RV(ir.Constant(0.0))
    forward_nodes = dag.upstream_nodes(transform.forward(y))
    assert output_lower in forward_nodes
    assert output_upper in forward_nodes
    assert distribution_lower in forward_nodes
    assert distribution_upper in forward_nodes


def test_transform_for_vmap_lifts_a_base_transform_pointwise():
    rate = ir.RV(ir.Constant([1.0, 2.0, 3.0]))
    op = ir.VMap(ir.Exponential(), (0,))

    transform = unconstraining.transform_for(op, (rate,))

    assert isinstance(transform, functional.Transform)
    x = ir.RV(ir.Constant([1.0, 2.0, 3.0]))
    y = transform.forward(x)
    reconstructed = transform.inverse(y)
    log_det = transform.log_det_jac(x, y)
    assert y.op == ir.VMap(ir.Log(), (0,), 3)
    assert reconstructed.op == ir.VMap(ir.Exp(), (0,), 3)
    assert log_det.shape == ()


def test_transform_for_vmap_maps_and_reuses_distribution_parents():
    lower = ir.RV(ir.Constant([-2.0, -1.0, 0.0]))
    upper = ir.RV(ir.Constant(4.0))
    op = ir.VMap(ir.Uniform(), (0, None))

    transform = unconstraining.transform_for(op, (lower, upper))

    assert isinstance(transform, functional.Transform)
    x = ir.RV(ir.Constant([0.0, 1.0, 2.0]))
    forward_nodes = dag.upstream_nodes(transform.forward(x))
    assert lower in forward_nodes
    assert upper in forward_nodes


def test_transform_for_vmap_recurses_through_nested_vmaps():
    rate = ir.RV(ir.Constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    inner = ir.VMap(ir.Exponential(), (0,))
    outer = ir.VMap(inner, (0,))

    transform = unconstraining.transform_for(outer, (rate,))

    assert isinstance(transform, functional.Transform)
    x = ir.RV(ir.Constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert transform.forward(x).shape == (2, 3)


def test_unconstrain_wraps_the_entire_vmap_distribution():
    rate = ir.RV(ir.Constant([1.0, 2.0, 3.0]))
    op = ir.VMap(ir.Exponential(), (0,))
    x = ir.RV(op, rate)

    reconstructed = functional.unconstrain(x, ())[0]

    free = reconstructed.parents[0]
    assert isinstance(free.op, ir.Bijected)
    assert free.op.base_op == op
    assert free.shape == x.shape
    assert reconstructed.shape == x.shape


def test_transform_for_scan_lifts_a_carry_independent_transform_pointwise():
    start = ir.RV(ir.Constant(2.0))
    op = ir.Scan(ir.Exponential(), 3, ())

    transform = unconstraining.transform_for(op, (start,))

    assert isinstance(transform, functional.Transform)
    x = ir.RV(ir.Constant([1.0, 2.0, 3.0]))
    y = transform.forward(x)
    reconstructed = transform.inverse(y)
    log_det = transform.log_det_jac(x, y)
    assert y.op == ir.VMap(ir.Log(), (0,), 3)
    assert reconstructed.op == ir.VMap(ir.Exp(), (0,), 3)
    assert log_det.shape == ()


def test_transform_for_scan_leaves_a_free_base_distribution_unchanged():
    start = ir.RV(ir.Constant(0.0))
    scale = ir.RV(ir.Constant(1.0))
    op = ir.Scan(ir.Normal(), 3, (None,))

    assert unconstraining.transform_for(op, (start, scale)) is None


def test_transform_for_scan_recursively_inverts_a_carry_dependent_transform():
    start = ir.RV(ir.Constant(-2.0))
    upper = ir.RV(ir.Constant(3.0))
    op = ir.Scan(ir.Uniform(), 3, (None,))

    transform = unconstraining.transform_for(op, (start, upper))

    assert isinstance(transform, functional.Transform)
    constrained = ir.RV(ir.Constant([-1.0, 0.0, 1.0]))
    free = transform.forward(constrained)
    reconstructed = transform.inverse(free)
    assert any(isinstance(node.op, ir.Concatenate) for node in dag.upstream_nodes(free))
    assert isinstance(reconstructed.op, ir.Scan)
    assert not reconstructed.op.random


def test_unconstrain_wraps_the_entire_scan_distribution():
    start = ir.RV(ir.Constant(2.0))
    op = ir.Scan(ir.Exponential(), 3, ())
    x = ir.RV(op, start)

    reconstructed = functional.unconstrain(x, ())[0]

    free = reconstructed.parents[0]
    assert isinstance(free.op, ir.Bijected)
    assert free.op.base_op == op
    assert free.shape == x.shape
    assert reconstructed.shape == x.shape


def test_transform_for_dispatches_to_vmap(monkeypatch):
    rate = ir.RV(ir.Constant([1.0, 2.0]))
    op = ir.VMap(ir.Exponential(), (0,))
    seen: list[object] = []

    def handle_vmap(op, parents):
        seen.extend((op, parents))
        return functional.tforms.log

    monkeypatch.setattr(unconstraining, "transform_for_vmap", handle_vmap)

    transform = unconstraining.transform_for(op, (rate,))

    assert transform is functional.tforms.log
    assert seen == [op, (rate,)]


def test_transform_for_supports_dirichlet():
    concentration = ir.RV(ir.Constant([1.0, 1.0]))

    assert (
        unconstraining.transform_for(ir.Dirichlet(), (concentration,))
        is functional.tforms.softmax_centered_ilr
    )


def test_unconstrain_reconstructs_a_positive_random_variable():
    rate = ir.RV(ir.Constant(2.0))
    x = ir.RV(ir.Exponential(), rate)

    reconstructed = functional.unconstrain(x, ())[0]

    assert reconstructed.shape == x.shape
    assert not reconstructed.op.random
    free = reconstructed.parents[0]
    assert isinstance(free.op, ir.Bijected)
    assert free.op.base_op == ir.Exponential()
    assert len(free.parents) == 1
    assert free.parents[0].op == rate.op


def test_unconstrain_captures_distribution_bounds():
    lower = ir.RV(ir.Constant(-2.0))
    upper = ir.RV(ir.Constant(3.0))
    x = ir.RV(ir.Uniform(), lower, upper)

    reconstructed = functional.unconstrain(x, ())[0]

    free = reconstructed.parents[0]
    assert isinstance(free.op, ir.Bijected)
    assert free.op.base_op == ir.Uniform()
    assert free.op.bijector.n_biject_params == 2
    new_lower, new_upper = free.parents[:2]
    assert new_lower.op == lower.op and new_upper.op == upper.op
    assert free.parents == (new_lower, new_upper, new_lower, new_upper)
    assert reconstructed.parents == (free, new_lower, new_upper)


def test_unconstrain_preserves_the_interface():
    interface = ir.RVInterface()
    rate = ir.RV(ir.Constant(2.0), interface=interface)
    x = ir.RV(ir.Exponential(), rate, interface=interface)

    reconstructed = functional.unconstrain(x, ())[0]

    assert all(var.interface is interface for var in dag.upstream_nodes(reconstructed))


def test_custom_selector_is_not_injected_into_default_recursion():
    rate = ir.RV(ir.Constant([1.0, 2.0]))
    x = ir.RV(ir.VMap(ir.Exponential(), (0,)), rate)
    seen = []

    def select(op, parents):
        seen.append(op)
        if isinstance(op, ir.Exponential):
            raise AssertionError("The default VMap helper must use the default selector")
        return functional.transform_for(op, parents)

    result, _ = functional.unconstrain(x, (), transform_for=select)
    assert seen == [x.op]
    assert result.op == ir.VMap(ir.Exp(), (0,), 2)


def test_custom_selector_can_handle_composed_ops_itself():
    rate = ir.RV(ir.Constant([1.0, 2.0]))
    x = ir.RV(ir.VMap(ir.Exponential(), (0,)), rate)

    def select(op, parents):
        assert isinstance(op, ir.VMap)
        return None

    result, _ = functional.unconstrain(x, (), transform_for=select)
    assert result.op == x.op


def test_custom_selector_receives_reconstructed_parent_rvs():
    x = ir.RV(ir.Exponential(), ir.RV(ir.Constant(2.0)))
    y = ir.RV(ir.Uniform(), x, ir.RV(ir.Add(), x, ir.RV(ir.Constant(1.0))))
    seen = []

    def select(op, parents):
        seen.append((op, parents))
        return functional.transform_for(op, parents)

    (new_x, new_y), _ = functional.unconstrain((x, y), (), transform_for=select)
    assert seen[1][0] == ir.Uniform()
    assert seen[1][1][0] is new_x
    assert seen[1][1][1].parents[0] is new_x
    assert new_x in dag.upstream_nodes(new_y)
