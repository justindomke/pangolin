from collections.abc import Callable, Iterator

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from pangolin import ir, jax_backend
from pangolin.functional import Transform, tforms


@pytest.fixture(autouse=True)
def enable_float64() -> Iterator[None]:
    """Use float64 for Jacobian checks without changing other tests' configuration."""
    with jax.enable_x64(True):
        yield


def compile_function(
    fun: Callable[..., ir.RV], *example_args: jax.Array
) -> Callable[..., jax.Array]:
    """Build an RV graph once and evaluate it with numerical inputs through the JAX backend."""
    inputs = tuple(ir.RV(ir.Constant(np.asarray(value))) for value in example_args)
    output = fun(*inputs)

    def call(*args: jax.Array) -> jax.Array:
        result = jax_backend.ancestor_sample_graph(
            output, None, dict(zip(inputs, args, strict=True))
        )
        assert isinstance(result, jax.Array)
        return result

    return call


# ==========================================
# Helpers for structured Matrix Jacobians
# ==========================================


def extract_tril(matrix):
    """Extracts the free parameters (lower triangle) of a square matrix."""
    return matrix[jnp.tril_indices(matrix.shape[0])]


def reconstruct_tril(flat_v, orig_x):
    """Reconstructs a lower triangular matrix from free parameters."""
    n = orig_x.shape[0]
    out = jnp.zeros((n, n), dtype=flat_v.dtype)
    return out.at[jnp.tril_indices(n)].set(flat_v)


def reconstruct_sym(flat_v, orig_x):
    """Reconstructs a symmetric matrix from its lower triangular free parameters."""
    L = reconstruct_tril(flat_v, orig_x)
    return L + L.T - jnp.diag(jnp.diag(L))


# ==========================================
# Shared Bijector Testing Utility
# ==========================================


def check_bijector(
    bijector: Transform,
    x,
    free_x_fn=lambda a: a.flatten(),
    free_y_fn=lambda a: a.flatten(),
    reconstruct_x_fn=lambda free_x, orig_x: free_x.reshape(orig_x.shape),
):
    """
    Tests forward/inverse roundtrips and validates log_det_jac against
    an exactly computed Jacobian of the free parameters using JAX.
    """
    forward = compile_function(bijector.forward, x)
    y = forward(x)
    inverse = compile_function(bijector.inverse, y)
    log_det_jac = compile_function(bijector.log_det_jac, x, y)
    reverse_log_det_jac = compile_function(bijector.reverse.log_det_jac, y, x)

    # 1. Forward -> Inverse Roundtrip
    x_rec = inverse(y)
    np.testing.assert_allclose(
        x, x_rec, rtol=1e-5, atol=1e-5, err_msg="Forward->Inverse roundtrip failed"
    )

    # 2. Inverse -> Forward Roundtrip
    y_rec = forward(x_rec)
    np.testing.assert_allclose(
        y, y_rec, rtol=1e-5, atol=1e-5, err_msg="Inverse->Forward roundtrip failed"
    )

    # 3. Exact Log-Det Jacobian (Forward)
    def flat_forward(free_x_val):
        x_in = reconstruct_x_fn(free_x_val, x)
        y_out = forward(x_in)
        return free_y_fn(y_out)

    free_x_val = free_x_fn(x)
    J = jax.jacfwd(flat_forward)(free_x_val)

    # Ensure mapping is between spaces of the same free-parameter dimension
    assert (
        J.shape[0] == J.shape[1]
    ), f"Jacobian not square! Mapping free dims: {J.shape[1]} -> {J.shape[0]}"

    sign, exact_ldj = jnp.linalg.slogdet(J)
    assert sign != 0, "Exact Jacobian is singular!"

    ldj = log_det_jac(x, y)
    np.testing.assert_allclose(
        ldj, exact_ldj, rtol=1e-5, atol=1e-5, err_msg="Forward log_det_jac mismatch"
    )

    # 4. Exact Log-Det Jacobian (Inverse)
    # The reverse bijector's ldj should exactly match -ldj of the forward mapping
    rev_ldj = reverse_log_det_jac(y, x_rec)
    np.testing.assert_allclose(
        rev_ldj, -ldj, rtol=1e-5, atol=1e-5, err_msg="Reverse log_det_jac mismatch"
    )


# ==========================================
# Pytest Test Cases
# ==========================================


def test_exp():
    x = jnp.array(2.5)
    bijector = tforms.exp
    check_bijector(bijector, x)


def test_log():
    x = jnp.array(2.5)  # Must be strictly positive
    bijector = tforms.log
    check_bijector(bijector, x)


def test_logit():
    x = jnp.array(0.75)  # Must be in (0, 1)
    bijector = tforms.logit
    check_bijector(bijector, x)


def test_inv_logit():
    y = jnp.array(1.5)
    bijector = tforms.inv_logit
    check_bijector(bijector, y)


def test_scaled_logit():
    x = jnp.array(4.0)
    bijector = tforms.scaled_logit(ir.RV(ir.Constant(1.0)), ir.RV(ir.Constant(5.0)))
    check_bijector(bijector, x)


def test_fill_tril():
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bijector = tforms.fill_tril
    check_bijector(
        bijector,
        x,
        free_x_fn=lambda a: a,  # Domain is already a flat vector
        free_y_fn=extract_tril,  # Codomain is lower triangular
        reconstruct_x_fn=lambda flat, orig: flat,
    )


def test_extract_tril():
    X = jnp.array([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]])
    bijector = tforms.extract_tril
    check_bijector(
        bijector,
        X,
        free_x_fn=extract_tril,  # Domain is lower triangular
        free_y_fn=lambda a: a,  # Codomain is flat vector
        reconstruct_x_fn=reconstruct_tril,
    )


def test_exp_diagonal():
    X = jnp.array([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]])
    bijector = tforms.exp_diagonal
    check_bijector(
        bijector,
        X,
        free_x_fn=extract_tril,  # Domain is lower triangular
        free_y_fn=extract_tril,  # Codomain is lower triangular
        reconstruct_x_fn=reconstruct_tril,
    )


def test_log_diagonal():
    # Diagonal must be strictly positive for log
    X = jnp.array([[1.0, 0.0, 0.0], [-2.0, 3.0, 0.0], [-4.0, 5.0, 6.0]])
    bijector = tforms.log_diagonal
    check_bijector(
        bijector,
        X,
        free_x_fn=extract_tril,
        free_y_fn=extract_tril,
        reconstruct_x_fn=reconstruct_tril,
    )


def test_cholesky():
    # Construct a strictly positive-definite matrix
    L = jnp.array([[2.0, 0.0, 0.0], [0.5, 2.0, 0.0], [-0.5, 0.5, 2.0]])
    X_spd = L @ L.T
    bijector = tforms.cholesky

    check_bijector(
        bijector,
        X_spd,
        free_x_fn=extract_tril,  # SPD parameterized by its lower triangle
        free_y_fn=extract_tril,  # Cholesky factor is lower triangular
        reconstruct_x_fn=reconstruct_sym,  # Must reconstruct symmetric matrix before forward pass
    )


def test_unconstrain_spd():
    # Construct a strictly positive-definite matrix
    L = jnp.array([[2.0, 0.0, 0.0], [0.5, 2.0, 0.0], [-0.5, 0.5, 2.0]])
    X_spd = L @ L.T
    bijector = tforms.unconstrain_spd

    check_bijector(
        bijector,
        X_spd,
        free_x_fn=extract_tril,  # SPD parameterized by its lower triangle
        free_y_fn=lambda a: a,  # Codomain is unconstrained flat vector
        reconstruct_x_fn=reconstruct_sym,
    )
