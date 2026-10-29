"""Shared numerical assertions for compositional model tests."""

import jax
import jax.numpy as jnp
import numpy as np


def assert_matches(
    actual: object,
    expected: tuple[jax.Array, ...],
    *,
    rtol: float = 2e-5,
    atol: float = 2e-6,
) -> None:
    """Check a tuple of finite JAX arrays against reference values with identical shapes."""
    assert isinstance(actual, tuple)
    assert len(actual) == len(expected)
    for found, wanted in zip(actual, expected, strict=True):
        assert isinstance(found, jax.Array)
        assert found.shape == wanted.shape
        assert jnp.all(jnp.isfinite(found))
        np.testing.assert_allclose(found, wanted, rtol=rtol, atol=atol)
