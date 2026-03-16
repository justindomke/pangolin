from math import log
import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Enable float64 for highly precise exact Jacobian testing
jax.config.update("jax_enable_x64", True)

from pangolin.jax_backend import (
    exp_bijector,
    log_bijector,
    logit_bijector,
    inv_logit_bijector,
    scaled_logit_bijector,
    fill_tril_bijector,
    extract_tril_bijector,
    exp_diagonal_bijector,
    log_diagonal_bijector,
    cholesky_bijector,
    unconstrain_spd_bijector,
)

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
    bijector,
    x,
    params=(),
    free_x_fn=lambda a: a.flatten(),
    free_y_fn=lambda a: a.flatten(),
    reconstruct_x_fn=lambda free_x, orig_x: free_x.reshape(orig_x.shape),
):
    """
    Tests forward/inverse roundtrips and validates log_det_jac against
    an exactly computed Jacobian of the free parameters using JAX.
    """
    # 1. Forward -> Inverse Roundtrip
    y = bijector.forward(x)
    x_rec = bijector.inverse(y)
    np.testing.assert_allclose(x, x_rec, rtol=1e-5, atol=1e-5, err_msg="Forward->Inverse roundtrip failed")

    # 2. Inverse -> Forward Roundtrip
    y_rec = bijector.forward(x_rec)
    np.testing.assert_allclose(y, y_rec, rtol=1e-5, atol=1e-5, err_msg="Inverse->Forward roundtrip failed")

    # 3. Exact Log-Det Jacobian (Forward)
    def flat_forward(free_x_val):
        x_in = reconstruct_x_fn(free_x_val, x)
        y_out = bijector.forward(x_in)
        return free_y_fn(y_out)

    free_x_val = free_x_fn(x)
    J = jax.jacfwd(flat_forward)(free_x_val)

    # Ensure mapping is between spaces of the same free-parameter dimension
    assert J.shape[0] == J.shape[1], f"Jacobian not square! Mapping free dims: {J.shape[1]} -> {J.shape[0]}"

    sign, exact_ldj = jnp.linalg.slogdet(J)
    assert sign != 0, "Exact Jacobian is singular!"

    _, ldj = bijector.forward_and_log_det_jac(x)
    np.testing.assert_allclose(ldj, exact_ldj, rtol=1e-5, atol=1e-5, err_msg="Forward log_det_jac mismatch")

    # 4. Exact Log-Det Jacobian (Inverse)
    # The reverse bijector's ldj should exactly match -ldj of the forward mapping
    _, rev_ldj = bijector.reverse.forward_and_log_det_jac(y)
    np.testing.assert_allclose(rev_ldj, -ldj, rtol=1e-5, atol=1e-5, err_msg="Reverse log_det_jac mismatch")


# ==========================================
# Pytest Test Cases
# ==========================================


def test_exp():
    x = jnp.array(2.5)
    bijector = exp_bijector()
    check_bijector(bijector, x)


def test_log():
    x = jnp.array(2.5)  # Must be strictly positive
    bijector = log_bijector()
    check_bijector(bijector, x)


def test_logit():
    x = jnp.array(0.75)  # Must be in (0, 1)
    bijector = logit_bijector()
    check_bijector(bijector, x)


def test_inv_logit():
    y = jnp.array(1.5)
    bijector = inv_logit_bijector()
    check_bijector(bijector, y)


def test_scaled_logit():
    x = jnp.array(4.0)
    bijector = scaled_logit_bijector(1.0, 5.0)
    check_bijector(bijector, x)


def test_fill_tril():
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bijector = fill_tril_bijector()
    check_bijector(
        bijector,
        x,
        free_x_fn=lambda a: a,  # Domain is already a flat vector
        free_y_fn=extract_tril,  # Codomain is lower triangular
        reconstruct_x_fn=lambda flat, orig: flat,
    )


def test_extract_tril():
    X = jnp.array([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]])
    bijector = extract_tril_bijector()
    check_bijector(
        bijector,
        X,
        free_x_fn=extract_tril,  # Domain is lower triangular
        free_y_fn=lambda a: a,  # Codomain is flat vector
        reconstruct_x_fn=reconstruct_tril,
    )


def test_exp_diagonal():
    X = jnp.array([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]])
    bijector = exp_diagonal_bijector()
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
    bijector = log_diagonal_bijector()
    check_bijector(bijector, X, free_x_fn=extract_tril, free_y_fn=extract_tril, reconstruct_x_fn=reconstruct_tril)


def test_cholesky():
    # Construct a strictly positive-definite matrix
    L = jnp.array([[2.0, 0.0, 0.0], [0.5, 2.0, 0.0], [-0.5, 0.5, 2.0]])
    X_spd = L @ L.T
    bijector = cholesky_bijector()

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
    bijector = unconstrain_spd_bijector()

    check_bijector(
        bijector,
        X_spd,
        free_x_fn=extract_tril,  # SPD parameterized by its lower triangle
        free_y_fn=lambda a: a,  # Codomain is unconstrained flat vector
        reconstruct_x_fn=reconstruct_sym,
    )
