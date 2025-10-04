from pangolin.ir import index_orthogonal
import numpy as np
from jax import numpy as jnp


def test_scalar_index():
    x = np.array([5, 0, 2, 7, 8])
    out = index_orthogonal(x, 1)
    # assert isinstance(out, np.ndarray)
    expected = np.array(0)
    assert np.allclose(out, expected)


def test_scalar_index_jax():
    x = jnp.array([5, 0, 2, 7, 8])
    out = index_orthogonal(x, 1)
    assert isinstance(out, jnp.ndarray)
    expected = np.array(0)
    assert np.allclose(out, expected)


def test1D():
    x = np.array([5, 0, 2, 7, 8])
    out = index_orthogonal(x, [4, 0, 1])
    expected = np.array([8, 5, 0])
    assert np.allclose(out, expected)


def test1D_jax():
    x = jnp.array([5, 0, 2, 7, 8])
    out = index_orthogonal(x, [4, 0, 1])
    expected = jnp.array([8, 5, 0])
    assert jnp.allclose(out, expected)
