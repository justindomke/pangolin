"""

This should be a dictionary mapping primitive random `Op` classes to `JaxBijector` instances. This is commonly used to transform constrained random `Op` (like `Dirichlet` or `Uniform`) to an unconstrained space to make gradient-based inference easier. This dictionary should cover all the base *random*, excluding `Composite`, `VMap` and `Scan`.

"""

from __future__ import annotations
from jax import numpy as jnp
import jax.tree_util
import numpy as np
from typing import Callable, Type, Sequence, Optional, Any
from pangolin import ir
from pangolin.ir import Op, RV

# from numpy.typing import ArrayLike
from numpyro import distributions as dist
from jax.scipy import special as jspecial
from jax import nn as jnn
from jax import Array as JaxArray
from pangolin import dag, util
from jaxtyping import PyTree
from jax.typing import ArrayLike


########################################################################################
# The core class
########################################################################################


class JaxBijector:
    """
    The idea is that if ``P(X)`` is some density and ``Y=T(X)`` is a diffeomorphism, then ``P(Y=y) = P(X=T⁻¹(y)) × |det ∇T⁻¹(y)|``

    Args:
        forward: jax function implementing forward transformation given ``x``
        inverse: jax function implementing inverse transformation given ``y``
        log_det_jax: jax function implementing the log determinant of the Jacobian of the forward transformation given both ``x`` and ``y`` (may use either as convenient)

    """

    def __init__(self, forward, inverse, log_det_jac):
        self._forward = forward
        self._inverse = inverse
        self._log_det_jac = log_det_jac

    def forward(self, x):
        """
        Computes ``T(x)``.
        """
        return self._forward(x)

    def inverse(self, y):
        """
        Computes ``T⁻¹(y)``
        """
        return self._inverse(y)

    def log_det_jac(self, x, y):
        """
        Computes ``log |det ∇T(x)| = -log |det ∇T⁻¹(y)|``. May use either ``x`` or ``y`` as convenient.
        """

        return self._log_det_jac(x, y)

    def forward_and_log_det_jac(self, x):
        """
        Computes ``T(x)`` and ``log |det ∇T(x)|``.
        """
        y = self.forward(x)
        ldj = self.log_det_jac(x, y)
        return y, ldj

    def inverse_and_log_det_jac(self, y):
        """
        Computes ``T⁻¹(y)`` and ``log |det ∇T(x)|``. (Flip the sign of the second return if you want the log determinant Jacobian of the inverse transformation.)
        """
        x = self.inverse(y)
        ldj = self.log_det_jac(x, y)
        return x, ldj

    @property
    def reverse(self):
        """
        Get a ``JaxBijector`` for ``T⁻¹``.
        """

        return JaxBijector(self.inverse, self.forward, lambda y, x: -self.log_det_jac(x, y))


########################################################################################
# Composition
########################################################################################


def compose_jax_bijectors(bijectors: Sequence[JaxBijector], log_det_direction: str = "forward") -> JaxBijector:
    bijectors = tuple(bijectors)

    def composed_forward(x):
        current_x = x
        for b in bijectors:
            current_x = b.forward(current_x)
        return current_x

    def composed_inverse(y):
        current_y = y
        for b in reversed(bijectors):
            # print(f"{current_y.shape=}")
            # print(f"{b=}")

            current_y = b.inverse(current_y)
        return current_y

    def _log_det_forward(x, y):
        log_det_sum = 0.0
        current_x = x
        for b in bijectors:
            next_x = b.forward(current_x)
            log_det_sum += b.log_det_jac(current_x, next_x)
            current_x = next_x

        return log_det_sum

    def _log_det_inverse(x, y):
        log_det_sum = 0.0
        current_y = y
        for b in reversed(bijectors):
            previous_x = b.inverse(current_y)
            log_det_sum += b.log_det_jac(previous_x, current_y)
            current_y = previous_x
        return log_det_sum

    if log_det_direction == "forward":
        composed_log_det_jac = _log_det_forward
    elif log_det_direction == "inverse":
        composed_log_det_jac = _log_det_inverse
    else:
        raise ValueError("log_det_direction must be 'forward' or 'inverse'")

    return JaxBijector(composed_forward, composed_inverse, composed_log_det_jac)


########################################################################################
# Exp / log bijectors
########################################################################################


def exp():
    """
    Creates a `JaxBijector` instance that applies the exponential function.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array(0.0)
        >>> exp().forward(x)
        Array(1., dtype=...)
    """
    # f(x) = exp(x)  <==>  df/dx = exp(x) = y  <==>  log df/dx = log(y) = x
    return JaxBijector(jnp.exp, jnp.log, lambda x, y: x)


def log():
    """
    Creates a `JaxBijector` instance that applies the natural logarithm.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array(1.0)
        >>> log().forward(x)
        Array(0., dtype=...)
    """

    # f(x) = log(x) <==> df/dx = 1/x <==> log df/dx = -log(x) = -y
    return JaxBijector(jnp.log, jnp.exp, lambda x, y: -y)


########################################################################################
# logit / inv_logit / scaled_logit bijectors
########################################################################################


def logit():
    """
    Create a `JaxBijector` instance that applies the logit bijector ``y = logit(x)``. Commonly used to transform from [0,1] to reals.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array(0.5)
        >>> logit().forward(x)
        Array(0., dtype=...)
    """
    return JaxBijector(jax.scipy.special.logit, jax.scipy.special.expit, lambda x, y: -jnp.log(x) - jnp.log1p(-x))


def inv_logit():
    """
    Create a `JaxBijector` instance that applies the inverse logit (expit/sigmoid).

    Example:
        >>> import jax.numpy as jnp
        >>> y = jnp.array(0.0)
        >>> inv_logit().forward(y)
        Array(0.5, dtype=...)
    """

    return logit().reverse


def scaled_logit(a, b):
    """
    Create a `JaxBijector` instance that applies the scaled logit ``y = logit((x-a)/(b-a))``. Commonly used to transform from [a,b] to reals.
    """
    return JaxBijector(
        lambda x: jax.scipy.special.logit((x - a) / (b - a)),
        lambda y: a + (b - a) * jax.scipy.special.expit(y),
        lambda x, y: jnp.log(b - a) - jnp.log(x - a) - jnp.log(b - x),
    )


########################################################################################
# fill / extract tril bijectors
########################################################################################


def _fill_tril(x: jax.Array) -> jax.Array:
    """Fill a lower triangular matrix from a packed 1D vector.

    This is the inverse operation of `_extract_tril`. Reconstructs a
    lower triangular matrix from its packed representation.

    Args:
        x: 1D array of packed lower triangular elements.

    Returns:
        Lower triangular matrix of shape $(n, n)$ reconstructed from the
        packed vector, where $n$ is determined from the length of x.

    Example:
        >>> x = jnp.array([1., 2., 3., 4., 5., 6.])
        >>> _fill_tril(x)
        Array([[1., 0., 0.],
               [2., 3., 0.],
               [4., 5., 6.]], dtype=...)

        >>> _fill_tril(jnp.array([1., 3., 4.]))
        Array([[1., 0.],
               [3., 4.]], dtype=...)
    """
    x = jnp.asarray(x)

    if x.ndim != 1:
        raise ValueError(f"Expected 1D input, got shape {x.shape}")

    m = x.shape[0]

    # Use standard Python math for static shape calculations to prevent JIT Concretization errors
    n = int(((8 * m + 1) ** 0.5 - 1) / 2)

    if n * (n + 1) // 2 != m:
        raise ValueError(f"Length {m} is not a triangular number n*(n+1)/2. Valid lengths: 1, 3, 6, 10, 15, ...")

    out = jnp.zeros((n, n), dtype=x.dtype)
    return out.at[jnp.tril_indices(n)].set(x)


def _extract_tril(X: jax.Array) -> jax.Array:
    """Extract lower triangular elements into a packed 1D vector.

    Packs the elements of the lower triangle of a square matrix into a
    1D array using row-major ordering (C-style).

    Args:
        X: Square matrix of shape $(n, n)$.

    Returns:
        1D array containing the packed lower triangular elements, including
        the diagonal. Length is $n(n+1)/2$.

    Example:
        >>> X = jnp.array([[1., 0., 0.],
        ...                [2., 3., 0.],
        ...                [4., 5., 6.]])
        >>> _extract_tril(X)
        Array([1., 2., 3., 4., 5., 6.], dtype=...)

        >>> _extract_tril(jnp.array([[1., 2.], [3., 4.]]))
        Array([1., 3., 4.], dtype=...)
    """
    X = jnp.asarray(X)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D input, got shape {X.shape}")

    if X.shape[0] != X.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {X.shape}")

    return X[jnp.tril_indices(X.shape[0])]


def fill_tril():
    """
    A `JaxBijector` instance that fills a lower-triangular matrix from a vector. Used to transform from real vectors to lower-triangular matrices.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([1., 2., 3.])
        >>> fill_tril().forward(x)
        Array([[1., 0.],
               [2., 3.]], dtype=...)
    """
    return JaxBijector(_fill_tril, _extract_tril, lambda x, y: jnp.array(0.0))


def extract_tril():
    """
    A `JaxBijector` instance that extracts the lower-triangular part of a matrix. Commonly used to transform from triangular lower-triangular matrices to real vectors.

    Example:
        >>> import jax.numpy as jnp
        >>> X = jnp.array([[1., 0.], [2., 3.]])
        >>> extract_tril().forward(X)
        Array([1., 2., 3.], dtype=...)
    """
    return JaxBijector(_extract_tril, _fill_tril, lambda x, y: jnp.array(0.0))


########################################################################################
#
########################################################################################


def _exp_diagonal(X: jax.Array):
    """
    Exponentiate diagonal elements: Y_ii = exp(X_ii), Y_ij = X_ij for i≠j.

    This is a bijection on square matrices that leaves off-diagonal
    elements unchanged and maps the diagonal through exp().

    Args:
        X: Square matrix of shape (n, n)

    Returns:
        Matrix of same shape with exponentiated diagonal

    Raises:
        ValueError: If X is not a 2D square matrix.

    Example:
        >>> X = jnp.array([[1., 2.], [3., 4.]])
        >>> _exp_diagonal(X)
        Array([[ 2.718...,  2.       ],
               [ 3.       , 54.598... ]], dtype=...)
    """
    X = jnp.asarray(X)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {X.shape}")

    x = jnp.diag(X)
    # X + diag(exp(x) - x) preserves off-diagonal and sets diagonal to exp(x)
    return X + jnp.diag(jnp.exp(x) - x)


def _log_diagonal(X: jax.Array):
    """
    Log-transform diagonal elements: Y_ii = log(X_ii), Y_ij = X_ij for i≠j.

    Inverse of _exp_diagonal. Requires strictly positive diagonal entries
    (otherwise returns NaN/Inf).

    Args:
        X: Square matrix with positive diagonal

    Returns:
        Matrix of same shape with log-transformed diagonal

    Raises:
        ValueError: If X is not a 2D square matrix.

    Example:
        >>> X = jnp.array([[1., 2.], [3., 4.]])
        >>> _log_diagonal(X)
        Array([[0.       , 2.       ],
               [3.       , 1.386...]], dtype=...)
    """
    X = jnp.asarray(X)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {X.shape}")

    x = jnp.diag(X)
    return X + jnp.diag(jnp.log(x) - x)


def _exp_diagonal_log_det_jac(X: jax.Array, Y: jax.Array):
    """
    Log determinant of Jacobian for _exp_diagonal.

    For Y = _exp_diagonal(X), the Jacobian is diagonal with:
    - dY_ii/dX_ii = exp(X_ii)  (n terms)
    - dY_ij/dX_ij = 1        for i≠j (n²-n terms)

    det(J) = ∏ exp(X_ii) = exp(∑ X_ii)
    log det(J) = ∑ X_ii

    Args:
        X: Input square matrix (n, n)
        Y: Output matrix (unused, kept for JaxBijector API consistency)

    Returns:
        Scalar log-determinant (sum of diagonal of X)

    Example:
        >>> X = jnp.array([[1., 2.], [3., 4.]])
        >>> Y = _exp_diagonal(X)
        >>> _exp_diagonal_log_det_jac(X, Y)  # 1 + 4 = 5
        Array(5., dtype=...)

        >>> # Verify: log(det) = log(exp(1)*exp(4)) = 5
        >>> float(_exp_diagonal_log_det_jac(X, Y)) == 5.0
        True
    """
    X = jnp.asarray(X)
    if X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {X.shape}")

    return jnp.trace(X)


def exp_diagonal():
    """
    Create a `JaxBijector` instance that exponentiates the diagonal of a matrix. Commonly used to transform real lower-triangular matrices into Cholesky factors.

    Example:
        >>> import jax.numpy as jnp
        >>> X = jnp.array([[0., 0.], [2., 0.]])
        >>> exp_diagonal().forward(X)
        Array([[1., 0.],
               [2., 1.]], dtype=...)
    """
    return JaxBijector(_exp_diagonal, _log_diagonal, _exp_diagonal_log_det_jac)


def log_diagonal():
    """
    Create a `JaxBijector` instance that takes the logarithm of the diagonal of a matrix. Commonly used to Cholesky factors into real lower-triangular matrices.

    Example:
        >>> import jax.numpy as jnp
        >>> X = jnp.array([[1., 0.], [2., 1.]])
        >>> log_diagonal().forward(X)
        Array([[0., 0.],
               [2., 0.]], dtype=...)
    """
    return exp_diagonal().reverse


########################################################################################
# cholesky bijector
########################################################################################


def _cholesky_log_det_jac(X, Y):
    """Logarithm of the absolute determinant of the Cholesky mapping Jacobian.

    The Cholesky decomposition maps a symmetric positive-definite matrix $X$
    to a lower triangular matrix $Y$ with positive diagonal elements such that
    $X = YY^T$. This function computes the log-determinant of the Jacobian
    of the forward transformation (from SPD matrix to Cholesky factor).

    The formula implemented is:
    $$ log |det J| = -k log(2) - sum_{i=1}^{k} (k-i+1) log(Y_{ii}) $$

    Args:
        X: Original symmetric positive-definite matrix of shape $(k, k)$ (unused
           in computation but kept for API consistency).
        Y: Cholesky factor, a lower triangular matrix of shape $(k, k)$ with
           positive diagonal elements.

    Returns:
        Scalar array containing $\log |\det J|$ where $J$ is the Jacobian of
        the Cholesky mapping.

    Example:
        >>> import jax.numpy as jnp
        >>> X = jnp.array([[4., 2.], [2., 2.]])
        >>> Y = jnp.linalg.cholesky(X)
        >>> _cholesky_log_det_jac(X, Y)
        Array(-2.7725887..., dtype=...)

        >>> X = jnp.array([[1., 0., 0.],
        ...                [0., 1., 0.],
        ...                [0., 0., 1.]])
        >>> Y = jnp.linalg.cholesky(X)
        >>> _cholesky_log_det_jac(X, Y)
        Array(-2.0794415, dtype=...)
    """
    k = Y.shape[0]
    # Match the dtype of Y to prevent implicit type upcasting
    powers = jnp.arange(k, 0, -1, dtype=Y.dtype)
    return -k * jnp.log(2.0) - powers @ jnp.log(jnp.diag(Y))


def cholesky():
    """
    Create a `JaxBijector` instance that applies a Cholesky decomposition. Commonly used to transform from symmetric positive definite matrices into triangular matrices.

    Example:
        >>> import jax.numpy as jnp
        >>> X = jnp.array([[1., 0.], [0., 1.]])
        >>> cholesky().forward(X)
        Array([[1., 0.],
               [0., 1.]], dtype=...)
    """
    return JaxBijector(
        lambda X: jnp.linalg.cholesky(X),
        lambda Y: Y @ Y.T,
        _cholesky_log_det_jac,
    )


########################################################################################
# bijector to unconstrain SPD matrices
########################################################################################


def spd_to_unconstrained():
    """
    Returns A `JaxBijector` instance that transforms a symmetric positive definite into the space of unconstrained reals. Accomplished by (1) taking a Cholesky decomposition (2) taking the logarithm of the diagonal (3) extracting the lower-triangular entries.

    Example:
        >>> import jax.numpy as jnp
        >>> # Identity matrix is symmetric positive definite
        >>> X = jnp.array([[1., 0.], [0., 1.]])
        >>> spd_to_unconstrained().forward(X)
        Array([0., 0., 0.], dtype=...)

        >>> # Transform back to SPD matrix
        >>> unconstrained_vec = jnp.array([0., 2., 0.])
        >>> spd_to_unconstrained().inverse(unconstrained_vec)
        Array([[1., 2.],
               [2., 5.]], dtype=float32)

        >>> X = jnp.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        >>> Y = spd_to_unconstrained().forward(X)
        >>> Y
        Array([0., 0., 0., 0., 0., 0.], dtype=...)
        >>> X_new = spd_to_unconstrained().inverse(Y)
        >>> jnp.allclose(X, X_new)
        Array(True, dtype=bool)

    """
    return compose_jax_bijectors([cholesky(), log_diagonal(), extract_tril()])


########################################################################################
# Stick-breaking
########################################################################################


def _stick_breaking_forward(x):
    """
    Forward: K-simplex -> R^(K-1)
    x: simplex vector of length K (positive, sums to 1)
    returns: y unconstrained (length K-1)

    Examples:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([0.5, 0.25, 0.25])
        >>> y = _stick_breaking_forward(x)
        >>> [round(val, 4) for val in y.tolist()]
        [0.0, 0.0]
    """
    x_trunc = x[:-1]

    # Exclusive cumulative sum: [0, x_1, x_1+x_2, ...]
    cumsum_exclusive = jnp.cumsum(x_trunc) - x_trunc

    # Denominator: [1.0, 1.0 - x_1, 1.0 - x_1 - x_2, ...]
    cumsum_remainder = 1.0 - cumsum_exclusive

    z = x_trunc / cumsum_remainder

    y = jnp.log(z) - jnp.log(1.0 - z)

    return y


def _stick_breaking_inverse(y):
    """
    Inverse: R^(K-1) -> K-simplex
    y: unconstrained real vector of length K-1
    returns: x on simplex (length K, sums to 1, positive)

    Examples:
        >>> import jax.numpy as jnp
        >>> y = jnp.array([0.0, 0.0])
        >>> x = _stick_breaking_inverse(y)
        >>> [round(val, 4) for val in x.tolist()]
        [0.5, 0.25, 0.25]
    """
    z = jax.nn.sigmoid(y)

    one_minus_z = 1.0 - z
    cumprod = jnp.concatenate([jnp.array([1.0]), jnp.cumprod(one_minus_z)])

    x = jnp.concatenate([z * cumprod[:-1], cumprod[-1:]])

    return x


def _stick_breaking_log_det_jac(x, y):
    """
    Log |det J| for forward transform: x (simplex) -> y (unconstrained)
    Uses only y for computation (x ignored but accepted for API compatibility)

    Examples:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([0.5, 0.25, 0.25])
        >>> y = jnp.array([0.0, 0.0])
        >>> log_det = _stick_breaking_log_det_jac(x, y)
        >>> round(float(log_det), 4)
        3.4657
    """
    z = jax.nn.sigmoid(y)
    log_z = jnp.log(z)
    log_one_minus_z = jnp.log1p(-z)

    log_remainder = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(log_one_minus_z)])

    # Forward Jacobian log|det| = - sum_{k=1}^{K-1} (log(z_k) + log(1-z_k) + log_r_k)
    return -jnp.sum(log_z + log_one_minus_z + log_remainder[:-1])


def stick_breaking():
    """
    Create a `JaxBijector` instance that applies the stick-breaking transformation.
    """

    return JaxBijector(_stick_breaking_forward, _stick_breaking_inverse, _stick_breaking_log_det_jac)


########################################################################################
# Softmax-centered
########################################################################################


def _softmax_centered_forward(x):
    """
    Forward: K-simplex -> R^(K-1)
    Maps a simplex to unconstrained space using the inverse softmax
    with a sum-to-zero constraint.
    """
    K = x.shape[0]
    # Log-transform and center (log-sum-exp for stability)
    log_x = jnp.log(x)
    y_full = log_x - jnp.mean(log_x)

    # Project from K-dim (sum-to-zero) to K-1 dim
    # Stan uses a specific basis; here we use a standard projection
    return _project_to_low_dim(y_full)


def _softmax_centered_inverse(y):
    """
    Inverse: R^(K-1) -> K-simplex
    Maps K-1 unconstrained values to a K-simplex.
    """
    # Project from K-1 dim to K-dim sum-to-zero space
    y_full = _project_to_high_dim(y)

    # Standard softmax
    return jax.nn.softmax(y_full)


def _project_to_high_dim(y):
    """
    Maps R^(K-1) to R^K such that the output sums to zero.
    Uses the Stan-style projection: y_k = -sum(y_{1:k-1}) / sqrt(k*(k-1))
    """
    K_minus_1 = y.shape[0]
    K = K_minus_1 + 1

    # This is a simplified version of the isometric mapping
    # that ensures the sum of the resulting vector is 0.
    means = jnp.concatenate([y, jnp.array([-jnp.sum(y)])])
    return means - jnp.mean(means)


def _project_to_low_dim(y_full):
    """
    Maps R^K (where sum(y_full) == 0) to R^(K-1).
    """
    return y_full[:-1] - jnp.mean(y_full)


def _softmax_centered_log_det_jac(x, y):
    """
    Log |det J| for the softmax-centered forward transform.
    For the simplex, this is: sum(log(x)) + log(K) + 0.5 * log(K)
    """
    K = x.shape[0]
    # The Jacobian for the softmax with sum-to-zero constraint
    # is simpler than the stick-breaking one.
    return -jnp.sum(jnp.log(x)) - 0.5 * jnp.log(K)


def softmax_centered():
    """
    Create a `JaxBijector` instance for the sum-to-zero softmax mapping. Commonly used to transform from the unit simplex to unconstrained vectors
    """
    return JaxBijector(_softmax_centered_forward, _softmax_centered_inverse, _softmax_centered_log_det_jac)


########################################################################################
# Softmax-centered ILR
########################################################################################


def _ilr_forward_project(z):
    """
    Project from CLR (sum-to-zero) space to ILR coordinates (orthonormal Helmert basis).
    Computes y_k = (sum_{i=0}^k z_i - (k+1)*z_{k+1}) / sqrt((k+1)(k+2)).
    """
    if z.ndim != 1:
        raise ValueError(f"ILR projection expects 1D input, got shape {z.shape}")
    K = z.shape[0]
    if K == 1:
        return jnp.array([])  # 0-dimensional output for K=1 simplex

    # k = 1, 2, ..., K-1  (corresponding to output indices 0..K-2)
    k = jnp.arange(1, K)
    cumsum_z = jnp.cumsum(z[:-1])  # [z0, z0+z1, ..., z0+...+z_{K-2}]
    numerators = cumsum_z - k * z[1:]
    return numerators / jnp.sqrt(k * (k + 1))


def _ilr_inverse_project(y):
    """
    Project from ILR coordinates back to CLR (sum-to-zero) space.
    Reconstructs z where sum(z)=0.
    """
    if y.ndim != 1:
        raise ValueError(f"ILR inverse expects 1D input, got shape {y.shape}")
    K_minus_1 = y.shape[0]
    K = K_minus_1 + 1
    if K == 1:
        return jnp.zeros((1,))

    # Coefficients a_k = y_k / sqrt((k+1)(k+2)) for k=0..K-2
    k = jnp.arange(K_minus_1)
    coeffs = y / jnp.sqrt((k + 1) * (k + 2))

    total = jnp.sum(coeffs)
    cumsum_coeffs = jnp.cumsum(coeffs)

    # Negative contributions for z[1..K-1]: -sqrt(i/(i+1)) * y[i-1]
    i = jnp.arange(1, K)  # 1, 2, ..., K-1
    neg_parts = jnp.sqrt(i / (i + 1)) * y

    # z[0] = total; z[i] = total - cumsum(coeffs)[i-1] - neg_parts[i-1]
    z_tail = total - cumsum_coeffs - neg_parts
    return jnp.concatenate([jnp.array([total]), z_tail])


def _softmax_centered_ilr_forward(x):
    """
    Forward: K-simplex -> R^(K-1) via CLR then ILR projection.
    **No batching.** Raises error if x.ndim != 1.
    """
    if x.ndim != 1:
        raise ValueError(
            f"Forward transform expects 1D simplex (K,), got {x.shape}. "
            f"Remove batch dimensions or use vmap explicitly."
        )
    log_x = jnp.log(x)
    z = log_x - jnp.mean(log_x)
    return _ilr_forward_project(z)


def _softmax_centered_ilr_inverse(y):
    """
    Inverse: R^(K-1) -> K-simplex via ILR then softmax.
    **No batching.** Raises error if y.ndim != 1.
    """
    if y.ndim != 1:
        raise ValueError(
            f"Inverse transform expects 1D unconstrained (K-1,), got {y.shape}. "
            f"Remove batch dimensions or use vmap explicitly."
        )
    if y.shape[0] == 0:
        return jnp.ones((1,))  # K=1 case
    z = _ilr_inverse_project(y)
    return jax.nn.softmax(z)


def _softmax_centered_ilr_log_det_jac(x, y):
    """
    Log |det J| for the FORWARD transformation (x -> y).

    For Stan ILR: log|det J| = -sum(log(x)) - 0.5*log(K)
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"LDJ expects 1D inputs, got x: {x.shape}, y: {y.shape}")

    K = x.shape[0]
    if y.shape[0] != K - 1:
        raise ValueError(f"Dimension mismatch in LDJ: x has {K} elements, " f"y has {y.shape[0]} (expected {K-1})")

    # ILR volume scaling: 1/(sqrt(K) * prod(x_i))
    return -jnp.sum(jnp.log(x)) - 0.5 * jnp.log(K)


def softmax_centered_ilr():
    """
    Create a `JaxBijector` for the Isometric Log-Ratio (ILR) simplex mapping borrowed from Stan.

    Uses the orthonormal Helmert basis.

    See also:
        `Stan: Simplex Transform <https://mc-stan.org/docs/reference-manual/transforms.html#simplex-transform.section>`_
    """
    return JaxBijector(_softmax_centered_ilr_forward, _softmax_centered_ilr_inverse, _softmax_centered_ilr_log_det_jac)


########################################################################################
# Default dict
########################################################################################


default_bijector_dict = {
    ir.Beta: lambda a, b: logit(),
    ir.Cauchy: None,
    ir.Dirichlet: lambda a: softmax_centered_ilr(),
    ir.Exponential: lambda a: log(),
    ir.Gamma: lambda a, b: log(),
    ir.Lognormal: lambda a, b: log(),
    ir.MultiNormal: None,
    ir.Normal: None,
    ir.NormalPrec: None,
    ir.StudentT: None,
    ir.Uniform: lambda a, b: scaled_logit(a, b),
    ir.Wishart: lambda a, b: spd_to_unconstrained(),
}
"""
A reasonable default bijector dictionary:

============================ ======
`Op` class                   bijector factory
============================ ======
`ir.Beta`                    `logit`
`ir.Cauchy`                  None
`ir.Dirichlet`               `softmax_centered_ilr`
`ir.Exponential`             `log`
`ir.Gamma`                   `log`
`ir.Lognormal`               `log`
`ir.MultiNormal`             None
`ir.Normal`                  None
`ir.NormalPrec`              None
`ir.StudentT`                None
`ir.Uniform`                 `scaled_logit`
`ir.Wishart`                 `spd_to_unconstrained`
============================ ======

It is easy to provide alternative bijectors: Just create a new dictionary, with functions that create new `JaxBijector` instances (if you want). (View source to see how this is defined.)
"""
