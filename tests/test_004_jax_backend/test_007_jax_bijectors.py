import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Enable float64 for highly precise exact Jacobian testing
jax.config.update("jax_enable_x64", True)

from pangolin.jax_backend import bijectors, JaxBijector, compose_jax_bijectors

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
    y = bijector.forward(x, *params)
    x_rec = bijector.inverse(y, *params)
    np.testing.assert_allclose(x, x_rec, rtol=1e-5, atol=1e-5, err_msg="Forward->Inverse roundtrip failed")

    # 2. Inverse -> Forward Roundtrip
    y_rec = bijector.forward(x_rec, *params)
    np.testing.assert_allclose(y, y_rec, rtol=1e-5, atol=1e-5, err_msg="Inverse->Forward roundtrip failed")

    # 3. Exact Log-Det Jacobian (Forward)
    def flat_forward(free_x_val):
        x_in = reconstruct_x_fn(free_x_val, x)
        y_out = bijector.forward(x_in, *params)
        return free_y_fn(y_out)

    free_x_val = free_x_fn(x)
    J = jax.jacfwd(flat_forward)(free_x_val)

    # Ensure mapping is between spaces of the same free-parameter dimension
    assert J.shape[0] == J.shape[1], f"Jacobian not square! Mapping free dims: {J.shape[1]} -> {J.shape[0]}"

    sign, exact_ldj = jnp.linalg.slogdet(J)
    assert sign != 0, "Exact Jacobian is singular!"

    _, ldj = bijector.forward_and_log_det_jac(x, *params)
    np.testing.assert_allclose(ldj, exact_ldj, rtol=1e-5, atol=1e-5, err_msg="Forward log_det_jac mismatch")

    # 4. Exact Log-Det Jacobian (Inverse)
    # The reverse bijector's ldj should exactly match -ldj of the forward mapping
    _, rev_ldj = bijector.reverse.forward_and_log_det_jac(y, *params)
    np.testing.assert_allclose(rev_ldj, -ldj, rtol=1e-5, atol=1e-5, err_msg="Reverse log_det_jac mismatch")


# ==========================================
# Pytest Test Cases
# ==========================================


def test_exp():
    x = jnp.array(2.5)
    check_bijector(bijectors.exp, x)


def test_log():
    x = jnp.array(2.5)  # Must be strictly positive
    check_bijector(bijectors.log, x)


def test_logit():
    x = jnp.array(0.75)  # Must be in (0, 1)
    check_bijector(bijectors.logit, x)


def test_inv_logit():
    y = jnp.array(1.5)
    check_bijector(bijectors.inv_logit, y)


def test_scaled_logit():
    x = jnp.array(4.0)
    # Testing bounds [1.0, 5.0]. Parameter count must match n_biject_params=2
    check_bijector(bijectors.scaled_logit, x, params=(1.0, 5.0))


def test_fill_tril():
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    check_bijector(
        bijectors.fill_tril,
        x,
        free_x_fn=lambda a: a,  # Domain is already a flat vector
        free_y_fn=extract_tril,  # Codomain is lower triangular
        reconstruct_x_fn=lambda flat, orig: flat,
    )


def test_extract_tril():
    X = jnp.array([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]])
    check_bijector(
        bijectors.extract_tril,
        X,
        free_x_fn=extract_tril,  # Domain is lower triangular
        free_y_fn=lambda a: a,  # Codomain is flat vector
        reconstruct_x_fn=reconstruct_tril,
    )


def test_exp_diagonal():
    X = jnp.array([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]])
    check_bijector(
        bijectors.exp_diagonal,
        X,
        free_x_fn=extract_tril,  # Domain is lower triangular
        free_y_fn=extract_tril,  # Codomain is lower triangular
        reconstruct_x_fn=reconstruct_tril,
    )


def test_log_diagonal():
    # Diagonal must be strictly positive for log
    X = jnp.array([[1.0, 0.0, 0.0], [-2.0, 3.0, 0.0], [-4.0, 5.0, 6.0]])
    check_bijector(
        bijectors.log_diagonal, X, free_x_fn=extract_tril, free_y_fn=extract_tril, reconstruct_x_fn=reconstruct_tril
    )


def test_cholesky():
    # Construct a strictly positive-definite matrix
    L = jnp.array([[2.0, 0.0, 0.0], [0.5, 2.0, 0.0], [-0.5, 0.5, 2.0]])
    X_spd = L @ L.T

    check_bijector(
        bijectors.cholesky,
        X_spd,
        free_x_fn=extract_tril,  # SPD parameterized by its lower triangle
        free_y_fn=extract_tril,  # Cholesky factor is lower triangular
        reconstruct_x_fn=reconstruct_sym,  # Must reconstruct symmetric matrix before forward pass
    )


def test_unconstrain_spd():
    # Construct a strictly positive-definite matrix
    L = jnp.array([[2.0, 0.0, 0.0], [0.5, 2.0, 0.0], [-0.5, 0.5, 2.0]])
    X_spd = L @ L.T

    check_bijector(
        bijectors.unconstrain_spd,
        X_spd,
        free_x_fn=extract_tril,  # SPD parameterized by its lower triangle
        free_y_fn=lambda a: a,  # Codomain is unconstrained flat vector
        reconstruct_x_fn=reconstruct_sym,
    )


def test_bijectors_instantiation():
    """Ensure the namespace prevents accidental instantiation."""
    with pytest.raises(TypeError, match="do not instantiate"):
        _ = bijectors()

