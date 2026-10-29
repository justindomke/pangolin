"""Scan index elimination, exact wiring, and equivalence to unsimplified graphs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pangolin import interface as pi, ir, jax_backend
from pangolin.functional import loop, scan
from pangolin.functional.simplifying import simplify_scan


def test_minimal_ir_rewrite_and_idempotence():
    init, indices, x = (ir.RV(ir.Constant(v)) for v in (0, [0, 1], [10, 20]))
    body = ir.Composite(3, [ir.Index(), ir.Add()], [[2, 1], [0, 3]])
    original = ir.Scan(body, 2, [0, None])
    op, parents = simplify_scan(original, (init, indices, x))
    assert op == ir.Scan(ir.Add(), 2, [0])
    assert parents == (init, x)
    unchanged, same_parents = simplify_scan(op, parents)
    assert unchanged is op
    assert same_parents is parents


def test_public_scan_default_and_opt_out():
    x, init, indices = pi.constant([10, 20]), pi.constant(0), pi.constant([0, 1])
    body = lambda carry, i: carry + x[i]
    simple = scan(body, 2)(init, indices)
    original = scan(body, 2, simplify=False)(init, indices)
    assert simple.op == ir.Scan(ir.Add(), 2, [0])
    assert simple.parents == (init, x)
    assert original.op == ir.Scan(
        ir.Composite(3, [ir.Index(), ir.Add()], [[2, 1], [0, 3]]), 2, [0, None]
    )
    assert original.parents == (init, indices, x)
    assert simple.interface is original.interface is init.interface
    for output in (simple, original):
        sample = jax.jit(jax_backend.ancestor_sampler(output))(jax.random.key(0))
        np.testing.assert_array_equal(sample, [10, 30])


def test_coordinate_used_in_arithmetic_remains():
    x, init, indices = pi.constant([10, 20]), pi.constant(0), pi.constant([0, 1])
    y = scan(lambda carry, i: carry + x[i] + i, 2)(init, indices)
    assert y.op == ir.Scan(ir.Composite(3, [ir.Add(), ir.Add()], [[0, 2], [3, 1]]), 2, [0, 0])
    assert y.parents == (init, indices, x)


def test_array_used_whole_and_indexed_remains_in_both_forms():
    x, init, indices = pi.constant([10, 20]), pi.constant(0), pi.constant([0, 1])
    y = scan(lambda carry, i: carry + x[i] + pi.sum(x, 0), 2)(init, indices)
    assert y.parents == (init, x, x)
    assert y.op == ir.Scan(
        ir.Composite(3, [ir.Add(), ir.Sum(0), ir.Add()], [[0, 2], [1], [3, 4]]),
        2,
        [None, 0],
    )


def test_repeated_indexing_reuses_one_scanned_input():
    x, init, indices = pi.constant([10, 20]), pi.constant(0), pi.constant([0, 1])
    y = scan(lambda carry, i: carry + x[i] + x[i], 2)(init, indices)
    assert y.op == ir.Scan(ir.Composite(2, [ir.Add(), ir.Add()], [[0, 1], [2, 1]]), 2, [0])
    assert y.parents == (init, x)


def test_existing_scanned_input_is_reused():
    x, init, indices = pi.constant([10, 20]), pi.constant(0), pi.constant([0, 1])
    y = scan(lambda carry, i, value: carry + x[i] + value, 2)(init, indices, x)
    assert y.op == ir.Scan(ir.Composite(2, [ir.Add(), ir.Add()], [[0, 1], [2, 1]]), 2, [0])
    assert y.parents == (init, x)


@pytest.mark.parametrize("indices", [[1, 0], [0, 0], [0], [0.0, 1.0]])
def test_noncanonical_coordinates_do_not_simplify(indices):
    init, x = pi.constant(0), pi.constant([10, 20])
    index = pi.constant(indices)
    body = ir.Composite(3, [ir.Index(), ir.Add()], [[2, 1], [0, 3]])
    op = ir.Scan(body, len(indices), [0, None])
    parents = (init, index, x)
    result, result_parents = simplify_scan(op, parents)
    assert result is op
    assert result_parents is parents


def test_carry_indexing_does_not_simplify():
    init, indices = pi.constant([10, 20]), pi.constant([0, 1])
    y = scan(lambda carry, i: carry + carry[i], 2, simplify=False)(init, indices)
    assert isinstance(y.op, ir.Scan)
    op, parents = simplify_scan(y.op, y.parents)
    assert op is y.op
    assert parents is y.parents


def test_computed_indices_do_not_simplify():
    x, init, indices = pi.constant([10, 20]), pi.constant(0), pi.constant([0, 1])
    y = scan(lambda carry, i: carry + x[1 - i], 2, simplify=False)(init, indices)
    assert isinstance(y.op, ir.Scan)
    op, parents = simplify_scan(y.op, y.parents)
    assert op is y.op
    assert parents is y.parents


@pytest.mark.parametrize("axis", [0, 1])
def test_vectorized_indexing_becomes_direct_scan_on_correct_axis(axis):
    x = pi.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    length = x.shape[axis]
    init = pi.constant(np.zeros(x.shape[1 - axis]))

    def step(carry, i):
        return carry + (x[i, :] if axis == 0 else x[:, i])

    y = scan(step, length)(init, pi.constant(range(length)))
    assert y.op == ir.Scan(ir.VMap(ir.Add(), [0, 0], init.shape[0]), length, [axis])
    assert y.parents == (init, x)
    sample = jax.jit(jax_backend.ancestor_sampler(y))(jax.random.key(0))
    np.testing.assert_array_equal(
        sample, np.moveaxis(np.cumsum([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], axis=axis), axis, 0)
    )


def test_scan_of_vmap_eliminates_all_coordinate_arrays():
    scales = pi.constant([[0.5, 0.7], [0.9, 1.1], [1.3, 1.5]])

    def model(prev, t, i):
        if t is None:
            return pi.constant(0.0)
        return pi.normal(prev[t - 1, i], scales[t, i])

    y = loop(
        model,
        interface=pi.default,
        t=[3],
        i=2,
    )
    assert y.op == ir.Scan(ir.VMap(ir.Normal(), [0, 0], 2), 3, [0])
    assert len(y.parents) == 2
    assert y.parents[1] is scales


def test_two_sequential_axes_eliminate_both_coordinates():
    first, second = pi.constant([1.0, 2.0]), pi.constant([3.0, 4.0, 5.0])

    def model(prev, t, s):
        if t is None:
            return first[s]
        if s is None:
            return second[t]
        return prev[t - 1, s] + prev[t, s - 1]

    y = loop(
        model,
        interface=pi.default,
        t=[3],
        s=[2],
    )
    assert y.op == ir.Scan(
        ir.Composite(2, [ir.Scan(ir.Composite(2, [ir.Add()], [[1, 0]]), 2, [0])], [[1, 0]]), 3, [0]
    )
    assert len(y.parents) == 2
    sample = jax.jit(jax_backend.ancestor_sampler(y))(jax.random.key(0))
    np.testing.assert_array_equal(sample, [[4.0, 6.0], [8.0, 14.0], [13.0, 27.0]])


def test_random_captured_array_density_and_gradients_are_preserved():
    scales = pi.exp(pi.normal(pi.constant([0.0, 0.1, 0.2]), 1.0))
    init, indices = pi.constant(0.0), pi.constant([0, 1, 2])
    body = lambda carry, i: pi.normal(carry, scales[i])
    simple = scan(body, 3)(init, indices)
    original = scan(body, 3, simplify=False)(init, indices)
    assert simple.op == ir.Scan(ir.Normal(), 3, [0])
    assert simple.parents == (init, scales)
    # Condition on the underlying normals and include both their density and the recurrence.
    latent = scales.parents[0]
    a = jax_backend.ancestor_log_prob((latent, simple))
    b = jax_backend.ancestor_log_prob((latent, original))
    values = (jnp.array([0.2, -0.1, 0.4]), jnp.array([0.1, 0.3, -0.2]))
    for transform in (lambda f: f, jax.jit, jax.grad, lambda f: jax.jit(jax.grad(f))):
        actual, expected = transform(a)(values), transform(b)(values)
        for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6)
