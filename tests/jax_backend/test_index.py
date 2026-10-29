"""Check scalar indexing validation, batching, compilation, and derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pangolin import functional, ir, jax_backend


def test_scalar_and_negative_indices():
    array = jnp.arange(6).reshape(2, 3)
    result = jax_backend.lower_deterministic(ir.Index(), (array, -1, 1))
    assert result.shape == ()
    assert result == 4
    assert jax_backend.index(jnp.array(7)) == 7


@pytest.mark.parametrize("indices,expected", [(1, 0), ([4, 0, 1], [8, 5, 0])])
def test_functional_orthogonal_indexing_values(indices, expected):
    array = ir.RV(ir.Constant([5, 0, 2, 7, 8]))
    index = ir.RV(ir.Constant(indices))
    result = functional.index_orthogonal(array, index)
    actual = jax_backend.ancestor_sample_graph(result, None)
    assert isinstance(actual, jax.Array)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("indices", [(), (0,), (0, 0, 0)])
def test_helper_rejects_wrong_index_count(indices):
    with pytest.raises(ValueError, match="one index per array dimension"):
        jax_backend.index(jnp.ones((2, 3)), *(jnp.asarray(i) for i in indices))


@pytest.mark.parametrize("indices", [([0], 1), (0, [1, 2]), ([[0]], 1)])
def test_helper_rejects_nonscalar_indices(indices):
    with pytest.raises(ValueError, match="scalar indices"):
        jax_backend.index(jnp.ones((2, 3)), *(jnp.asarray(i) for i in indices))


def test_helper_rejects_float_indices():
    with pytest.raises(TypeError):
        jax_backend.index(jnp.arange(3), jnp.array(1.0))


def test_jit_and_vmap_with_dynamic_indices():
    array = jnp.arange(6).reshape(2, 3)
    rows, columns = jnp.array([1, 0]), jnp.array([2, 0, 1])
    op = ir.VMap(ir.VMap(ir.Index(), (None, None, 0), 3), (None, 0, None), 2)
    fun = jax.jit(lambda a, i, j: jax_backend.lower_deterministic(op, (a, i, j)))
    np.testing.assert_array_equal(fun(array, rows, columns), [[5, 3, 4], [2, 0, 1]])
    paired = jax.vmap(jax_backend.index, in_axes=(None, 0, 0))
    np.testing.assert_array_equal(paired(array, rows, jnp.array([0, 2])), [3, 2])


def test_grad_accumulates_repeated_selections():
    indices = jnp.array([2, 0, 2])
    op = ir.VMap(ir.Index(), (None, 0), 3)
    fun = lambda a: jax_backend.lower_deterministic(op, (a, indices)).sum()
    np.testing.assert_array_equal(jax.jit(jax.grad(fun))(jnp.arange(4.0)), [1, 0, 2, 0])
