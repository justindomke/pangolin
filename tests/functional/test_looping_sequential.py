"""Sequential loop nesting, initializer sharing, and predecessor indexing."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import distributions as dist

from pangolin import dag, interface as pi, ir, jax_backend
from pangolin.functional import loop


def evaluate(output):
    return jax.jit(jax_backend.ancestor_sampler(output))(jax.random.key(0))


def check_density(output, values, reference):
    density = jax_backend.ancestor_log_prob(output)
    np.testing.assert_allclose(density(values), reference(values), rtol=1e-6)
    np.testing.assert_allclose(jax.jit(density)(values), reference(values), rtol=1e-6)
    actual = jax.jit(jax.grad(density))(values)
    expected = jax.grad(reference)(values)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_scalar_recurrence_matches_sfor_and_excludes_initializer():
    def model(prev, t):
        if t is None:
            return pi.constant(2)
        return prev[t - 1] + t + 1

    x = loop(model, interface=pi.default, t=[4])
    old = pi.sfor(lambda previous, t: previous + t + 1, init=2, t=4)
    assert x.op == old.op
    assert x.shape == (4,)
    np.testing.assert_array_equal(evaluate(x), [3, 5, 8, 12])


@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_sequential_axis_in_each_nesting_position(position):
    def first(prev, t, i, j):
        if t is None:
            return i + j
        return prev[t - 1, i, j] + 100 * i + 10 * j + t

    def middle(prev, i, t, j):
        if t is None:
            return i + j
        return prev[i, t - 1, j] + 100 * i + 10 * j + t

    def last(prev, i, j, t):
        if t is None:
            return i + j
        return prev[i, j, t - 1] + 100 * i + 10 * j + t

    model = {"first": first, "middle": middle, "last": last}[position]
    x = loop(model, interface=pi.default, j=3, t=[4], i=2)
    state = np.arange(2)[:, None] + np.arange(3)
    path = []
    for t in range(4):
        state = state + 100 * np.arange(2)[:, None] + 10 * np.arange(3) + t
        path.append(state)
    expected = np.stack(path)
    if position == "middle":
        expected = expected.transpose(1, 0, 2)
    elif position == "last":
        expected = expected.transpose(1, 2, 0)
    assert x.shape == expected.shape
    np.testing.assert_array_equal(evaluate(x), expected)
    op = x.op
    for _ in range({"first": 0, "middle": 1, "last": 2}[position]):
        assert isinstance(op, ir.VMap)
        op = op.base_op
    assert isinstance(op, ir.Scan)


def test_whole_inner_array_and_other_component_reads():
    def model(prev, i, t, j):
        if t is None:
            return i + j
        return pi.sum(prev[i, t - 1, :], 0) + prev[i, t - 1, 2 - j] + i + t

    x = loop(model, interface=pi.default, i=2, t=[3], j=3)
    state = np.arange(2)[:, None] + np.arange(3)
    path = []
    for t in range(3):
        state = state.sum(axis=1, keepdims=True) + state[:, ::-1] + np.arange(2)[:, None] + t
        path.append(state)
    assert x.shape == (2, 3, 3)
    np.testing.assert_array_equal(evaluate(x), np.stack(path, axis=1))


def test_array_state_slices_and_negative_index():
    def initial(i):
        return pi.constant([1.0, 2.0]) + i

    def model(prev, i, t):
        if t is None:
            return initial(i)
        return prev[i, t - 1, :] + prev[i, t - 1, -1] + t

    x = loop(model, interface=pi.default, i=2, t=[3])
    state = np.array([[1.0, 2.0], [2.0, 3.0]])
    path = []
    for t in range(3):
        state = state + state[:, -1:] + t
        path.append(state)
    assert x.shape == (2, 3, 2)
    np.testing.assert_array_equal(evaluate(x), np.stack(path, axis=1))


def test_partial_slices_of_two_inner_axes():
    def model(prev, t, i, j):
        if t is None:
            return i + j
        return pi.sum(prev[t - 1, ::-1, j], 0) + prev[t - 1, i, -1]

    x = loop(model, interface=pi.default, t=[3], i=2, j=3)
    state = np.arange(2)[:, None] + np.arange(3)
    path = []
    for _ in range(3):
        state = state.sum(axis=0)[None, :] + state[:, -1:]
        path.append(state)
    np.testing.assert_array_equal(evaluate(x), np.stack(path))


def test_random_recurrence_joint_density_and_gradient():
    def model(prev, i, t, j):
        if t is None:
            return i + j + 0.0
        return pi.normal(prev[i, t - 1, j] + i + t, 0.7)

    x = loop(
        model,
        interface=pi.default,
        i=2,
        t=[3],
        j=2,
    )
    values = jnp.arange(12, dtype=float).reshape(2, 3, 2) / 10

    def reference(v):
        start = jnp.arange(2)[:, None] + jnp.arange(2)[None, :]
        previous = jnp.concatenate((start[:, None, :], v[:, :-1, :]), axis=1)
        means = previous + jnp.arange(2)[:, None, None] + jnp.arange(3)[None, :, None]
        return dist.Normal(means, 0.7).log_prob(v).sum()

    check_density(x, values, reference)


def test_fresh_random_initializer_is_an_ancestor_outside_scan():
    def model(prev, i, t, j):
        if t is None:
            return pi.normal(i + j, 1)
        return pi.normal(prev[i, t - 1, j], 0.5)

    x = loop(
        model,
        interface=pi.default,
        i=2,
        t=[3],
        j=2,
    )
    random_nodes = [v for v in dag.upstream_nodes(x) if v.op.random]
    assert len(random_nodes) == 2
    initial = next(v for v in random_nodes if v is not x)
    assert initial.shape == (2, 2)
    assert initial.op == ir.VMap(ir.VMap(ir.Normal(), (0, None), 2), (0, None), 2)
    start = jnp.array([[0.1, 1.2], [1.3, 2.4]])
    path = jnp.arange(12, dtype=float).reshape(2, 3, 2) / 5

    def reference(v):
        a, b = v
        means = jnp.arange(2)[:, None] + jnp.arange(2)[None, :]
        previous = jnp.concatenate((a[:, None, :], b[:, :-1, :]), axis=1)
        return (
            dist.Normal(means, 1).log_prob(a).sum() + dist.Normal(previous, 0.5).log_prob(b).sum()
        )

    check_density((initial, x), (start, path), reference)


def test_captured_initializer_draw_is_shared():
    shared = pi.normal(0.0, 1.0)

    def model(prev, i, t, j):
        if t is None:
            return shared
        return prev[i, t - 1, j] + 1.0

    x = loop(
        model,
        interface=pi.default,
        i=2,
        t=[3],
        j=4,
    )
    assert [v for v in dag.upstream_nodes(x) if v.op.random] == [shared]
    a, b = evaluate((shared, x))
    np.testing.assert_allclose(
        b, np.broadcast_to(np.asarray(a) + np.arange(1, 4)[None, :, None], (2, 3, 4)), rtol=1e-6
    )


def test_initializer_can_create_multiple_random_nodes():
    def initial(i):
        a = pi.normal(i, 1.0)
        return pi.normal(a, 1.0)

    def model(prev, t, i):
        if t is None:
            return initial(i)
        return prev[t - 1, i] + 1.0

    x = loop(model, interface=pi.default, t=[3], i=2)
    random_nodes = [v for v in dag.upstream_nodes(x) if v.op.random]
    assert len(random_nodes) == 2
    assert all(v.shape == (2,) for v in random_nodes)
    start, values = evaluate((random_nodes[-1], x))
    np.testing.assert_allclose(values, start[None, :] + np.arange(1, 4)[:, None], rtol=1e-6)


def test_recursive_sequential_loop_inside_initializer():
    def initial(i):
        def model(prev, s):
            if s is None:
                return pi.constant(0)
            return prev[s - 1] + i + s

        return loop(model, interface=pi.default, s=[2])

    def model_2(prev, i, t):
        if t is None:
            return initial(i)
        return prev[i, t - 1, :] + 1

    x = loop(model_2, interface=pi.default, i=2, t=[3])
    assert x.shape == (2, 3, 2)
    starts = np.array([[0, 1], [1, 3]])
    np.testing.assert_array_equal(evaluate(x), starts[:, None, :] + np.arange(1, 4)[None, :, None])


def test_initialization_and_body_each_called_once():
    seen = []

    def initial(i, j):
        seen.append(("init", i.shape, j.shape))
        return pi.constant(0.0)

    def model(prev, i, t, j):
        if t is None:
            return initial(i, j)
        seen.append(("body", i.shape, t.shape, j.shape))
        return prev[i, t - 1, j] + 1.0

    x = loop(model, interface=pi.default, i=20, t=[100], j=30)
    assert x.shape == (20, 100, 30)
    assert seen == [("init", (), ()), ("body", (), (), ())]


def test_foreign_state_interface_and_raw_coordinate_interface():
    coordinate_interface = ir.RVInterface()
    state_interface = ir.RVInterface()
    one = ir.RV(ir.Constant(1))

    def model(prev, t):
        if t is None:
            return ir.RV(ir.Constant(0), interface=state_interface)
        assert t.interface is coordinate_interface
        previous = prev[ir.RV(ir.Sub(), t, one)]
        assert previous.interface is state_interface
        return ir.RV(ir.Add(), previous, one, interface=state_interface)

    x = loop(
        model,
        interface=coordinate_interface,
        t=[3],
    )
    assert x.interface is state_interface
    np.testing.assert_array_equal(evaluate(x), [1, 2, 3])


def test_sequential_singleton_axis():
    def model(prev, t):
        if t is None:
            return pi.constant(5)
        return prev[t - 1] + 2

    x = loop(model, interface=pi.default, t=[1])
    assert x.shape == (1,)
    np.testing.assert_array_equal(evaluate(x), [7])


@pytest.mark.parametrize(
    "read",
    [
        lambda p, i, t, j: p[i, t, j],
        lambda p, i, t, j: p[i, t - 2, j],
        lambda p, i, t, j: p[i, t + (-1), j],
        lambda p, i, t, j: p[i + 1, t - 1, j],
        lambda p, i, t, j: p[:, t - 1, j],
        lambda p, i, t, j: p[i, j, t - 1],
    ],
)
def test_invalid_predecessor_reads(read):
    with pytest.raises(ValueError, match="coordinate"):

        def model(prev, i, t, j):
            if t is None:
                return i + j
            return read(prev, i, t, j) + 1

        loop(
            model,
            interface=pi.default,
            i=2,
            t=[3],
            j=4,
        )


def test_empty_suffix_returns_array_carry():
    def model(prev, t):
        if t is None:
            return pi.constant([1, 2])
        return prev[t - 1] + 1

    x = loop(model, interface=pi.default, t=[3])
    np.testing.assert_array_equal(evaluate(x), [[2, 3], [3, 4], [4, 5]])


def test_suffix_ellipsis_uses_normal_interface_indexing():
    def model(prev, t):
        if t is None:
            return pi.constant([[1, 2], [3, 4]])
        return prev[t - 1, ...] + 1

    x = loop(
        model,
        interface=pi.default,
        t=[3],
    )
    np.testing.assert_array_equal(
        evaluate(x), np.array([[1, 2], [3, 4]]) + np.arange(1, 4)[:, None, None]
    )


def test_unused_carry_and_discarded_previous_read_are_rejected():
    def unused(prev, t):
        prev[t - 1] + 1
        return pi.normal(t, 1)

    for body in (lambda prev, t: pi.normal(t, 1), unused):
        with pytest.raises(ValueError, match="depend on its carry"):
            loop(body, interface=pi.default, t=[3, pi.constant(0.0)])


def test_returning_previous_state_itself_is_rejected():
    with pytest.raises(ValueError, match="every output must be created"):

        def model(prev, t):
            if t is None:
                return pi.constant(0.0)
            return prev[t - 1]

        loop(
            model,
            interface=pi.default,
            t=[3],
        )


def test_returning_captured_rv_is_rejected():
    captured = pi.normal(0, 1)
    with pytest.raises(ValueError, match="every output must be created"):

        def model(prev, t):
            if t is None:
                return pi.constant(0.0)
            return captured

        loop(
            model,
            interface=pi.default,
            t=[3],
        )


def test_random_intermediate_rejected_in_recurrence():
    def model(prev, t):
        if t is None:
            return pi.constant(0.0)
        a = pi.normal(prev[t - 1], 1.0)
        return pi.normal(a, 1.0)

    with pytest.raises(ValueError, match="all but the last Op must be deterministic"):
        loop(model, interface=pi.default, t=[3])


def test_shape_and_interface_must_match_initializer():
    with pytest.raises(ValueError, match="match carry shape"):

        def model(prev, t):
            if t is None:
                return pi.constant(0)
            return prev[t - 1] + pi.constant([1, 2])

        loop(
            model,
            interface=pi.default,
            t=[3],
        )
    other = pi.Interface()
    with pytest.raises(ValueError, match="interface must match"):

        def model_2(prev, t):
            if t is None:
                return pi.constant(0)
            return other.add(prev[t - 1], 1)

        loop(
            model_2,
            interface=pi.default,
            t=[3],
        )


@pytest.mark.parametrize("specification", [[], [3, 0.0], [3, lambda: pi.constant(0), 4]])
def test_invalid_initializer_specifications(specification):
    with pytest.raises(TypeError, match="initializer"):
        loop(lambda prev, t: prev[t - 1] + 1, interface=pi.default, t=specification)


@pytest.mark.parametrize("decorator", [False, True])
@pytest.mark.parametrize("container", [list, tuple])
def test_callable_initializers_are_rejected_without_execution(decorator, container):
    def initial(s):
        pytest.fail("initializer must not run")

    def body(prev, t, s):
        pytest.fail("body must not run")

    spec = container((2, initial))
    with pytest.raises(TypeError, match="initializer must be an RV or None"):
        if decorator:

            def declaration(prev, t=spec, s=[3]):
                return body(prev, t, s)

            loop(declaration, interface=pi.default)
        else:
            loop(body, interface=pi.default, t=spec, s=[3])


def test_sequential_main_signature():
    with pytest.raises(ValueError, match="parameter named 'prev'"):
        loop(lambda t: pi.constant(0), interface=pi.default, t=[3, pi.constant(0)])
    with pytest.raises(ValueError, match="unknown axis 'prev'"):
        loop(lambda prev, t: prev[t - 1], interface=pi.default, t=[3, pi.constant(0)], prev=2)


def test_initializer_and_body_must_return_rvs():
    for value in (1, None, (pi.constant(0),)):
        with pytest.raises(TypeError, match="must return a single RV"):

            def model(prev, t):
                if t is None:
                    return value
                return prev[t - 1] + 1

            loop(model, interface=pi.default, t=[3])
        with pytest.raises(TypeError, match="must return a single RV"):

            def model_2(prev, t):
                if t is None:
                    return pi.constant(0)
                return value

            loop(model_2, interface=pi.default, t=[3])


def test_random_increments_are_independent_across_axes_and_time():
    def model(prev, i, t, j):
        if t is None:
            return pi.constant(0.0)
        return pi.normal(prev[i, t - 1, j], 1.0)

    x = loop(
        model,
        interface=pi.default,
        i=2,
        t=[3],
        j=2,
    )
    sampler = jax_backend.ancestor_sampler(x)
    samples = np.asarray(jax.jit(jax.vmap(sampler))(jax.random.split(jax.random.key(9), 4000)))
    assert samples.shape == (4000, 2, 3, 2)
    increments = np.diff(samples, axis=2, prepend=np.zeros((4000, 2, 1, 2)))
    np.testing.assert_allclose(increments.mean(axis=0), 0.0, atol=0.06)
    np.testing.assert_allclose(
        np.cov(increments.reshape(4000, 12), rowvar=False), np.eye(12), atol=0.08
    )


def test_log_prob_composes_with_sequential_loop_and_grad():
    values = pi.constant([0.3, -0.1, 0.7])
    parameter = pi.constant(0.2)

    def distribution(drift):
        def model(prev, t):
            if t is None:
                return pi.constant(0.0)
            return pi.normal(prev[t - 1] + drift, 0.5)

        return loop(
            model,
            interface=pi.default,
            t=[3],
        )

    density = pi.log_prob(distribution)
    outputs = density(values, parameter), pi.grad(density, 1)(values, parameter)
    actual = evaluate(outputs)

    def reference(drift):
        x = jnp.array([0.3, -0.1, 0.7])
        return dist.Normal(jnp.concatenate((jnp.zeros(1), x[:-1])) + drift, 0.5).log_prob(x).sum()

    np.testing.assert_allclose(actual, (reference(0.2), jax.grad(reference)(0.2)), rtol=1e-6)


def test_body_can_nest_another_sequential_loop():
    # Nested calls have their own predecessor helper, which is always named prev.
    def model(prev, t):
        if t is None:
            return pi.constant([1, 2])
        old = prev[t - 1, :]

        def model_2(prev, s):
            if s is None:
                return pi.constant(0)
            return prev[s - 1] + old[s]

        return loop(
            model_2,
            interface=pi.default,
            s=[2],
        )

    x = loop(model, interface=pi.default, t=[3])
    np.testing.assert_array_equal(evaluate(x), [[1, 3], [1, 4], [1, 5]])


def test_unused_carry_rejected_with_inner_vectorization():
    with pytest.raises(ValueError, match="depend on its carry"):

        def model(prev, t, j):
            if t is None:
                return pi.constant(0.0)
            return pi.normal(t + j, 1.0)

        loop(
            model,
            interface=pi.default,
            t=[3],
            j=2,
        )


def test_sequential_validation_precedes_initializer_execution():
    def initial():
        pytest.fail("invalid axis size must be rejected first")

    for size in (True, "3", 1.5):
        with pytest.raises(TypeError, match="must be an integer"):

            def model(prev, t):
                if t is None:
                    return initial()
                return prev[t - 1] + 1

            loop(model, interface=pi.default, t=[size])
    for size in (0, -1):
        with pytest.raises(ValueError, match="must be positive"):

            def model_2(prev, t):
                if t is None:
                    return initial()
                return prev[t - 1] + 1

            loop(model_2, interface=pi.default, t=[size])


def test_explicit_slice_required_with_vectorized_inner_axes_and_array_state():
    def initial(i, j):
        return pi.constant([1, 2]) + i + j

    def model(prev, i, t, j):
        if t is None:
            return initial(i, j)
        return prev[i, t - 1, j, :] + 1

    x = loop(model, interface=pi.default, i=2, t=[3], j=4)
    expected = (
        np.array([1, 2])[None, None, None, :]
        + np.arange(2)[:, None, None, None]
        + np.arange(1, 4)[None, :, None, None]
        + np.arange(4)[None, None, :, None]
    )
    assert x.shape == (2, 3, 4, 2)
    np.testing.assert_array_equal(evaluate(x), expected)


def test_prebuilt_random_initialization_is_reused_by_recurrence():
    from pangolin.functional.looping import initialize_axis, loop_axes

    calls = []
    coordinates = tuple(pi.constant(np.arange(size)) for size in (2, 3, 4))
    shared = pi.normal(0.0, 1.0)

    def initializer(i, j):
        calls.append("initial")
        latent = pi.normal(shared + i, 1.0)
        return pi.normal(latent + j, 0.5)

    initial = initialize_axis(initializer, coordinates, 1)
    ancestors = [rv for rv in dag.upstream_nodes(initial) if rv.op.random]
    assert initial.shape == (2, 4)
    assert len(ancestors) == 3
    assert shared in ancestors
    assert calls == ["initial"]

    def model(prev, i, t, j):
        calls.append("body")
        return prev[i, t - 1, j] + shared

    output = loop_axes(model, coordinates, (None, initial, None), names=("i", "t", "j"))
    assert calls == ["initial", "body"]
    assert [rv for rv in dag.upstream_nodes(output) if rv.op.random] == ancestors
    a, start, path = evaluate((shared, initial, output))
    expected = start[:, None, :] + np.arange(1, 4)[None, :, None] * a
    np.testing.assert_allclose(path, expected, rtol=1e-6, atol=1e-6)


def test_initial_prefix_selection_needs_no_indexing_interface():
    from pangolin.functional.looping import initialize_axis, loop_axes

    state_interface = ir.RVInterface()
    coordinates = tuple(ir.RV(ir.Constant(range(size))) for size in (2, 3, 4))
    mean = ir.RV(ir.Constant(np.zeros(2)))
    covariance = ir.RV(ir.Constant(np.eye(2)))
    increment = ir.RV(ir.Constant([1.0, 2.0]))
    one = ir.RV(ir.Constant(1))

    def initializer(i, j):
        return ir.RV(ir.MultiNormal(), mean, covariance, interface=state_interface)

    initial = initialize_axis(initializer, coordinates, 2)

    def model(prev, i, j, t):
        previous = prev[i, j, ir.RV(ir.Sub(), t, one)]
        return ir.RV(ir.VMap(ir.Add(), (0, 0), 2), previous, increment, interface=state_interface)

    output = loop_axes(model, coordinates, (None, None, initial), names=("i", "j", "t"))
    assert initial.shape == (2, 3, 2)
    assert output.shape == (2, 3, 4, 2)
    assert output.interface is state_interface
    assert [rv for rv in dag.upstream_nodes(output) if rv.op.random] == [initial]
    start, path = evaluate((initial, output))
    expected = start[:, :, None, :] + np.arange(1, 5)[None, None, :, None] * np.array([1, 2])
    np.testing.assert_allclose(path, expected, rtol=1e-6, atol=1e-6)


def test_log_prob_gradient_includes_parameter_dependent_initialization():
    values = jnp.arange(12, dtype=float).reshape(2, 3, 2) / 7 - 0.4
    parameter = pi.constant(0.3)

    def distribution(weight):
        def initial(i, j):
            return weight * (i + 1) + j

        def model(prev, i, t, j):
            if t is None:
                return initial(i, j)
            old = prev[i, t - 1, j]
            total = pi.sum(prev[i, t - 1, :], 0)
            return pi.normal(0.3 * old + 0.2 * total + weight * j, 0.6)

        return loop(model, interface=pi.default, i=2, t=[3], j=2)

    density = pi.log_prob(distribution)
    value_rv = pi.constant(values)
    actual = evaluate((density(value_rv, parameter), pi.grad(density, 1)(value_rv, parameter)))

    def reference(weight):
        initial = weight * (jnp.arange(2)[:, None] + 1) + jnp.arange(2)
        previous = jnp.concatenate((initial[:, None, :], values[:, :-1, :]), axis=1)
        means = 0.3 * previous + 0.2 * previous.sum(axis=-1, keepdims=True)
        means = means + weight * jnp.arange(2)
        return dist.Normal(means, 0.6).log_prob(values).sum()

    np.testing.assert_allclose(
        actual, (reference(0.3), jax.grad(reference)(0.3)), rtol=1e-5, atol=1e-6
    )


def test_hierarchical_initializer_and_trajectory_joint_density():
    shared = pi.normal(0.0, 1.0)

    def initial(i, j):
        latent = pi.normal(shared + i, 0.4)
        return pi.normal(latent + j, 0.7)

    def model(prev, i, t, j):
        if t is None:
            return initial(i, j)
        return pi.normal(0.5 * prev[i, t - 1, j] + shared, 0.6)

    x = loop(
        model,
        interface=pi.default,
        i=2,
        t=[3],
        j=2,
    )
    random_nodes = [v for v in dag.upstream_nodes(x) if v.op.random]
    assert len(random_nodes) == 4
    assert random_nodes[0] is shared
    assert random_nodes[-1] is x
    assert [v.shape for v in random_nodes] == [(), (2, 2), (2, 2), (2, 3, 2)]
    values = (
        jnp.array(0.2),
        jnp.array([[0.1, -0.2], [0.8, 1.3]]),
        jnp.array([[0.2, 0.7], [0.9, 1.9]]),
        jnp.arange(12, dtype=float).reshape(2, 3, 2) / 9,
    )

    def reference(v):
        z, latent, start, path = v
        previous = jnp.concatenate((start[:, None, :], path[:, :-1, :]), axis=1)
        return (
            dist.Normal(0, 1).log_prob(z)
            + dist.Normal(z + jnp.arange(2)[:, None], 0.4).log_prob(latent).sum()
            + dist.Normal(latent + jnp.arange(2), 0.7).log_prob(start).sum()
            + dist.Normal(0.5 * previous + z, 0.6).log_prob(path).sum()
        )

    check_density(tuple(random_nodes), values, reference)
