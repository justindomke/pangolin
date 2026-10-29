"""Provide standard backend-independent Transform combinators and factories."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike

from pangolin import ir

from .indexing import index_orthogonal
from .transforming import Transform, compose_transforms
from .vmapping import vmap

__all__ = [
    "exp",
    "log",
    "logit",
    "inv_logit",
    "stick_breaking",
    "softmax_centered",
    "softmax_centered_ilr",
    "scaled_logit",
    "scale_by",
    "shift_by",
    "cholesky",
    "fill_tril",
    "extract_tril",
    "exp_diagonal",
    "log_diagonal",
    "unconstrain_spd",
]


def constant_like(prototype: ir.RV, value: ArrayLike) -> ir.RV:
    """Construct a constant carrying ``prototype``'s interface."""
    return ir.RV(ir.Constant(value), interface=prototype.interface)


def apply(op: ir.Op, *args: ir.RV) -> ir.RV:
    """Apply an operation to one or more RVs without broadcasting."""
    if not args:
        raise ValueError("apply requires at least one RV")
    return ir.RV(op, *args, interface=args[0].interface)


def elementwise_apply(op: ir.Op, *args: ir.RV) -> ir.RV:
    """Apply a scalar operation at every position of equal-shaped RVs using `vmap`."""
    if not args:
        raise ValueError("elementwise_apply requires at least one RV")
    if any(arg.shape != args[0].shape for arg in args[1:]):
        raise ValueError("elementwise operands must have equal shapes")

    def scalar_fun(*values: ir.RV) -> ir.RV:
        return apply(op, *values)

    mapped_fun = scalar_fun
    for _ in args[0].shape:
        mapped_fun = vmap(mapped_fun)
    return mapped_fun(*args)


def negative(x: ir.RV) -> ir.RV:
    """Negate a scalar RV."""
    return apply(ir.Mul(), constant_like(x, -1), x)


def positive_sign() -> ir.RV:
    """Declare a strictly increasing scalar bijection."""
    return ir.RV(ir.Constant(1))


def exp_forward(x: ir.RV) -> ir.RV:
    """Apply the scalar exponential operation."""
    return apply(ir.Exp(), x)


def exp_inverse(y: ir.RV) -> ir.RV:
    """Apply the scalar logarithm operation."""
    return apply(ir.Log(), y)


def exp_log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
    """Return the exponential forward log-Jacobian determinant."""
    return apply(ir.Log(), y)


exp: Final[Transform] = Transform(
    exp_forward,
    exp_inverse,
    exp_log_det_jac,
    sign=positive_sign,
)
"""Map real scalars to positive scalars with the exponential Transform."""


log: Final[Transform] = exp.reverse
"""Map positive scalars to real scalars with the logarithm Transform."""


def logit_forward(x: ir.RV) -> ir.RV:
    """Apply the scalar logit operation."""
    return apply(ir.Logit(), x)


def logit_inverse(y: ir.RV) -> ir.RV:
    """Apply the scalar inverse-logit operation."""
    return apply(ir.InvLogit(), y)


def logit_log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
    """Return the logit forward log-Jacobian determinant."""
    one_minus_x = apply(ir.Sub(), constant_like(x, 1), x)
    log_x = apply(ir.Log(), x)
    log_one_minus_x = apply(ir.Log(), one_minus_x)
    return apply(ir.Sub(), negative(log_x), log_one_minus_x)


logit: Final[Transform] = Transform(
    logit_forward,
    logit_inverse,
    logit_log_det_jac,
    sign=positive_sign,
)
"""Map the open unit interval to the real line with the logit Transform."""


inv_logit: Final[Transform] = logit.reverse
"""Map the real line to the open unit interval with the inverse-logit Transform."""


def scale_by(factor: ir.RV) -> Transform:
    """Create the scalar Transform ``y = factor * x``.

    ``scale_by`` is a Transform factory. The returned Transform's functions capture the exact
    ``factor`` RV in their closures, so applying the Transform makes ``factor`` a parent in the
    resulting graph.

    Parameters
    ----------
    factor
        Scalar IR RV captured as a Transform parameter; its value must be finite and nonzero.

    Returns
    -------
    transform
        Scalar scaling Transform whose functions close over ``factor``.

    Raises
    ------
    TypeError
        If ``factor`` is not an `ir.RV`.
    ValueError
        If ``factor`` is not scalar, or the Transform is applied to a nonscalar RV.

    Examples
    --------
    >>> factor = ir.RV(ir.Constant(2))
    >>> transform = scale_by(factor)
    >>> x = ir.RV(ir.Constant(3))
    >>> transform.forward(x).parents == (x, factor)
    True
    """
    if not isinstance(factor, ir.RV):
        raise TypeError(f"scale_by factor must be an RV; got {type(factor).__name__}")
    if factor.shape != ():
        raise ValueError("scale_by factor must be scalar")

    def forward(x: ir.RV) -> ir.RV:
        if x.shape != ():
            raise ValueError("scale_by supports only scalar values")
        return apply(ir.Mul(), x, factor)

    def inverse(y: ir.RV) -> ir.RV:
        if y.shape != ():
            raise ValueError("scale_by supports only scalar values")
        return apply(ir.Div(), y, factor)

    def log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
        return apply(ir.Log(), apply(ir.Abs(), factor))

    def sign() -> ir.RV:
        return apply(ir.Div(), factor, apply(ir.Abs(), factor))

    return Transform(
        forward,
        inverse,
        log_det_jac,
        sign=sign,
    )


def shift_by(offset: ir.RV) -> Transform:
    """Create the elementwise Transform ``y = x + offset``.

    ``shift_by`` is a Transform factory. The returned Transform's functions capture the exact
    ``offset`` RV in their closures, so applying the Transform makes ``offset`` a parent in the
    resulting graph.

    ``x`` and ``offset`` must have identical shapes; this function performs no broadcasting.

    Parameters
    ----------
    offset
        IR RV captured as a Transform parameter; its values must be finite.

    Returns
    -------
    transform
        Elementwise shift Transform whose functions close over ``offset``.

    Raises
    ------
    TypeError
        If ``offset`` is not an `ir.RV`.
    ValueError
        If the Transform is applied to an RV with a different shape.

    Examples
    --------
    >>> offset = ir.RV(ir.Constant([1, 2]))
    >>> x = ir.RV(ir.Constant([3, 4]))
    >>> shift_by(offset).forward(x).shape
    (2,)
    """
    if not isinstance(offset, ir.RV):
        raise TypeError(f"shift_by offset must be an RV; got {type(offset).__name__}")

    def forward(x: ir.RV) -> ir.RV:
        return elementwise_apply(ir.Add(), x, offset)

    def inverse(y: ir.RV) -> ir.RV:
        return elementwise_apply(ir.Sub(), y, offset)

    def log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
        return constant_like(x, 0.0)

    return Transform(
        forward,
        inverse,
        log_det_jac,
        sign=positive_sign if offset.shape == () else None,
    )


def scaled_logit(lower: ir.RV, upper: ir.RV) -> Transform:
    """Create the scalar Transform from ``(lower, upper)`` to the real line.

    ``scaled_logit`` is a Transform factory. The returned Transform's functions capture the exact
    ``lower`` and ``upper`` RVs in their closures, so applying the Transform makes the bounds
    parents in the resulting graph.

    Parameters
    ----------
    lower
        Finite scalar lower endpoint captured as a Transform parameter.
    upper
        Finite scalar upper endpoint, strictly greater than ``lower``, captured as a parameter.

    Returns
    -------
    transform
        Scalar scaled-logit Transform whose functions close over ``lower`` and ``upper``.

    Raises
    ------
    TypeError
        If either endpoint is not an `ir.RV`.
    ValueError
        If either endpoint or the transformed value is nonscalar.

    Examples
    --------
    >>> lower = ir.RV(ir.Constant(1))
    >>> upper = ir.RV(ir.Constant(5))
    >>> x = ir.RV(ir.Constant(3))
    >>> scaled_logit(lower, upper).forward(x).shape
    ()
    """
    if not isinstance(lower, ir.RV) or not isinstance(upper, ir.RV):
        raise TypeError("scaled_logit bounds must be RVs")
    if lower.shape != () or upper.shape != ():
        raise ValueError("scaled_logit bounds must be scalar")

    def forward(x: ir.RV) -> ir.RV:
        if x.shape != ():
            raise ValueError("scaled_logit supports only scalar values")
        shifted = apply(ir.Sub(), x, lower)
        width = apply(ir.Sub(), upper, lower)
        return apply(ir.Logit(), apply(ir.Div(), shifted, width))

    def inverse(y: ir.RV) -> ir.RV:
        if y.shape != ():
            raise ValueError("scaled_logit supports only scalar values")
        width = apply(ir.Sub(), upper, lower)
        scaled = apply(ir.Mul(), width, apply(ir.InvLogit(), y))
        return apply(ir.Add(), lower, scaled)

    def log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
        width = apply(ir.Log(), apply(ir.Abs(), apply(ir.Sub(), upper, lower)))
        lower_distance = apply(ir.Log(), apply(ir.Abs(), apply(ir.Sub(), x, lower)))
        upper_distance = apply(ir.Log(), apply(ir.Abs(), apply(ir.Sub(), upper, x)))
        return apply(ir.Sub(), apply(ir.Sub(), width, lower_distance), upper_distance)

    return Transform(
        forward,
        inverse,
        log_det_jac,
        sign=positive_sign,
    )


def require_simplex_shape(value: ir.RV, name: str) -> int:
    """Validate a nondegenerate simplex-coordinate shape and return its size."""
    if len(value.shape) != 1:
        raise ValueError(f"{name} requires a vector")
    size = value.shape[0]
    if size < 2:
        raise ValueError(f"{name} requires at least two simplex components")
    return size


def require_simplex_pair(x: ir.RV, y: ir.RV, name: str) -> int:
    """Validate matching constrained and unconstrained simplex-coordinate shapes."""
    size = require_simplex_shape(x, name)
    if y.shape != (size - 1,):
        raise ValueError(f"{name} requires output shape {(size - 1,)}, got {y.shape}")
    return size


def stick_breaking_forward(x: ir.RV) -> ir.RV:
    """Map a simplex vector to unconstrained stick-breaking coordinates."""
    size = require_simplex_shape(x, "stick_breaking forward")
    unconstrained_size = size - 1
    truncation = constant_like(x, np.eye(unconstrained_size, size))
    x_truncated = apply(ir.Matmul(), truncation, x)
    exclusive_sum = constant_like(
        x, np.tril(np.ones((unconstrained_size, unconstrained_size)), k=-1)
    )
    used_stick = apply(ir.Matmul(), exclusive_sum, x_truncated)
    remainder = elementwise_apply(
        ir.Sub(), constant_like(x, np.ones(unconstrained_size)), used_stick
    )
    proportions = elementwise_apply(ir.Div(), x_truncated, remainder)
    return elementwise_apply(ir.Logit(), proportions)


def stick_breaking_inverse(y: ir.RV) -> ir.RV:
    """Map unconstrained stick-breaking coordinates to a simplex vector."""
    if len(y.shape) != 1:
        raise ValueError("stick_breaking inverse requires a vector")
    unconstrained_size = y.shape[0]
    size = unconstrained_size + 1
    proportions = elementwise_apply(ir.InvLogit(), y)
    one_minus_proportions = elementwise_apply(
        ir.Sub(), constant_like(y, np.ones(unconstrained_size)), proportions
    )
    log_one_minus_proportions = elementwise_apply(ir.Log(), one_minus_proportions)
    prefix_sum = constant_like(y, np.tril(np.ones((size, unconstrained_size)), k=-1))
    log_remainders = apply(ir.Matmul(), prefix_sum, log_one_minus_proportions)
    remainders = elementwise_apply(ir.Exp(), log_remainders)

    extension = constant_like(y, np.eye(size, unconstrained_size))
    extended_proportions = apply(ir.Matmul(), extension, proportions)
    last_one = np.zeros(size)
    last_one[-1] = 1
    extended_proportions = elementwise_apply(
        ir.Add(), extended_proportions, constant_like(y, last_one)
    )
    return elementwise_apply(ir.Mul(), extended_proportions, remainders)


def stick_breaking_log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
    """Return the forward log-Jacobian determinant of stick breaking."""
    size = require_simplex_pair(x, y, "stick_breaking log determinant")
    unconstrained_size = size - 1
    proportions = elementwise_apply(ir.InvLogit(), y)
    one_minus_proportions = elementwise_apply(
        ir.Sub(), constant_like(y, np.ones(unconstrained_size)), proportions
    )
    log_proportions = elementwise_apply(ir.Log(), proportions)
    log_one_minus_proportions = elementwise_apply(ir.Log(), one_minus_proportions)
    exclusive_sum = constant_like(
        y, np.tril(np.ones((unconstrained_size, unconstrained_size)), k=-1)
    )
    log_remainders = apply(ir.Matmul(), exclusive_sum, log_one_minus_proportions)
    terms = elementwise_apply(ir.Add(), log_proportions, log_one_minus_proportions)
    terms = elementwise_apply(ir.Add(), terms, log_remainders)
    return negative(apply(ir.Sum(0), terms))


stick_breaking: Final[Transform] = Transform(
    stick_breaking_forward,
    stick_breaking_inverse,
    stick_breaking_log_det_jac,
)
"""Map a simplex of length ``K`` to ``K - 1`` stick-breaking coordinates."""


def softmax_centered_forward(x: ir.RV) -> ir.RV:
    """Map a simplex vector to centered log coordinates with the last coordinate omitted."""
    size = require_simplex_shape(x, "softmax_centered forward")
    projection = np.eye(size) - np.ones((size, size)) / size
    log_x = elementwise_apply(ir.Log(), x)
    return apply(ir.Matmul(), constant_like(x, projection[:-1]), log_x)


def softmax_centered_inverse(y: ir.RV) -> ir.RV:
    """Map centered log coordinates to a simplex vector."""
    if len(y.shape) != 1:
        raise ValueError("softmax_centered inverse requires a vector")
    unconstrained_size = y.shape[0]
    expansion = np.vstack([np.eye(unconstrained_size), -np.ones((1, unconstrained_size))])
    logits = apply(ir.Matmul(), constant_like(y, expansion), y)
    return apply(ir.Softmax(), logits)


def softmax_centered_log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
    """Return the forward log-Jacobian determinant of the centered-softmax transform."""
    size = require_simplex_pair(x, y, "softmax_centered log determinant")
    log_x = elementwise_apply(ir.Log(), x)
    negative_log_product = negative(apply(ir.Sum(0), log_x))
    normalization = constant_like(x, np.log(size))
    return apply(ir.Sub(), negative_log_product, normalization)


softmax_centered: Final[Transform] = Transform(
    softmax_centered_forward,
    softmax_centered_inverse,
    softmax_centered_log_det_jac,
)
"""Map a simplex of length ``K`` to ``K - 1`` centered log coordinates."""


def helmert_basis(size: int) -> np.ndarray:
    """Return the orthonormal Helmert basis for a simplex of the given size."""
    basis = np.zeros((size - 1, size))
    for row in range(size - 1):
        count = row + 1
        denominator = np.sqrt(count * (count + 1))
        basis[row, :count] = 1 / denominator
        basis[row, count] = -count / denominator
    return basis


def softmax_centered_ilr_forward(x: ir.RV) -> ir.RV:
    """Map a simplex vector to orthonormal isometric log-ratio coordinates."""
    size = require_simplex_shape(x, "softmax_centered_ilr forward")
    basis = constant_like(x, helmert_basis(size))
    log_x = elementwise_apply(ir.Log(), x)
    return apply(ir.Matmul(), basis, log_x)


def softmax_centered_ilr_inverse(y: ir.RV) -> ir.RV:
    """Map orthonormal isometric log-ratio coordinates to a simplex vector."""
    if len(y.shape) != 1:
        raise ValueError("softmax_centered_ilr inverse requires a vector")
    size = y.shape[0] + 1
    basis = constant_like(y, helmert_basis(size))
    logits = apply(ir.Matmul(), apply(ir.Transpose(), basis), y)
    return apply(ir.Softmax(), logits)


def softmax_centered_ilr_log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
    """Return the forward log-Jacobian determinant of the simplex ILR transform."""
    size = require_simplex_pair(x, y, "softmax_centered_ilr log determinant")
    log_x = elementwise_apply(ir.Log(), x)
    negative_log_product = negative(apply(ir.Sum(0), log_x))
    normalization = constant_like(x, 0.5 * np.log(size))
    return apply(ir.Sub(), negative_log_product, normalization)


softmax_centered_ilr: Final[Transform] = Transform(
    softmax_centered_ilr_forward,
    softmax_centered_ilr_inverse,
    softmax_centered_ilr_log_det_jac,
)
"""Map a simplex of length ``K`` to ``K - 1`` orthonormal ILR coordinates."""


def cholesky_forward(x: ir.RV) -> ir.RV:
    """Compute a lower-triangular Cholesky factor."""
    return apply(ir.Cholesky(), x)


def cholesky_inverse(y: ir.RV) -> ir.RV:
    """Reconstruct a positive-definite matrix from its Cholesky factor."""
    transposed = apply(ir.Transpose(), y)
    return apply(ir.Matmul(), y, transposed)


def cholesky_log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
    """Return the forward log-Jacobian determinant of Cholesky decomposition."""
    if len(y.shape) != 2 or y.shape[0] != y.shape[1]:
        raise ValueError("cholesky log determinant requires a square matrix")
    size = y.shape[0]
    diagonal = apply(ir.Diag(), y)
    log_diagonal = elementwise_apply(ir.Log(), diagonal)
    powers = constant_like(y, list(range(size, 0, -1)))
    weighted_logs = apply(ir.Matmul(), powers, log_diagonal)
    log_two = apply(ir.Log(), constant_like(y, 2))
    normalization = apply(ir.Mul(), constant_like(y, -size), log_two)
    return apply(ir.Sub(), normalization, weighted_logs)


cholesky: Final[Transform] = Transform(
    cholesky_forward,
    cholesky_inverse,
    cholesky_log_det_jac,
)
"""Map positive-definite matrices to lower-triangular Cholesky factors."""


def fill_tril_forward(params: ir.RV) -> ir.RV:
    """Fill a lower-triangular matrix row by row from a vector."""
    if len(params.shape) != 1:
        raise ValueError("fill_tril requires a vector")
    num_entries = params.shape[0]
    size = int((np.sqrt(8 * num_entries + 1) - 1) / 2)
    if size * (size + 1) // 2 != num_entries:
        raise ValueError("fill_tril vector length must be triangular")

    index_mesh = np.zeros((size, size), dtype=int)
    rows, columns = np.tril_indices(size)
    index_mesh[rows, columns] = np.arange(num_entries)
    indices = constant_like(params, index_mesh)
    raw_matrix = index_orthogonal(params, indices, interface=params.interface)
    mask = constant_like(params, np.tril(np.ones((size, size))))
    return elementwise_apply(ir.Mul(), raw_matrix, mask)


def extract_tril_forward(matrix: ir.RV) -> ir.RV:
    """Extract lower-triangular matrix entries row by row into a vector."""
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("extract_tril requires a square matrix")
    size = matrix.shape[0]
    rows, columns = np.tril_indices(size)
    row_rv = constant_like(matrix, rows)
    column_rv = constant_like(matrix, columns)

    def select(value: ir.RV, row: ir.RV, column: ir.RV) -> ir.RV:
        return apply(ir.Index(), value, row, column)

    return vmap(select, in_axes=(None, 0, 0))(matrix, row_rv, column_rv)


def zero_log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
    """Return a scalar zero log-Jacobian determinant."""
    return constant_like(x, 0.0)


fill_tril: Final[Transform] = Transform(
    fill_tril_forward,
    extract_tril_forward,
    zero_log_det_jac,
)
"""Map vectors to lower-triangular matrices by filling entries row by row."""


extract_tril: Final[Transform] = fill_tril.reverse
"""Map lower-triangular matrices to vectors by extracting entries row by row."""


def exp_diagonal_forward(x: ir.RV) -> ir.RV:
    """Exponentiate a matrix diagonal while preserving other entries."""
    diagonal = apply(ir.Diag(), x)
    transformed = elementwise_apply(ir.Exp(), diagonal)
    difference = elementwise_apply(ir.Sub(), transformed, diagonal)
    adjustment = apply(ir.DiagMatrix(), difference)
    return elementwise_apply(ir.Add(), x, adjustment)


def exp_diagonal_inverse(y: ir.RV) -> ir.RV:
    """Take the logarithm of a matrix diagonal while preserving other entries."""
    diagonal = apply(ir.Diag(), y)
    transformed = elementwise_apply(ir.Log(), diagonal)
    difference = elementwise_apply(ir.Sub(), transformed, diagonal)
    adjustment = apply(ir.DiagMatrix(), difference)
    return elementwise_apply(ir.Add(), y, adjustment)


def exp_diagonal_log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
    """Return the forward log-Jacobian determinant of diagonal exponentiation."""
    return apply(ir.Sum(0), apply(ir.Diag(), x))


exp_diagonal: Final[Transform] = Transform(
    exp_diagonal_forward,
    exp_diagonal_inverse,
    exp_diagonal_log_det_jac,
)
"""Exponentiate a matrix diagonal while preserving other entries."""


log_diagonal: Final[Transform] = exp_diagonal.reverse
"""Take the logarithm of a matrix diagonal while preserving other entries."""


unconstrain_spd: Final[Transform] = compose_transforms([cholesky, log_diagonal, extract_tril])
"""Map positive-definite matrices to unconstrained vectors."""
