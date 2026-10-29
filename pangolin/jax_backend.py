"""This module compiles pangolin models into plain-old JAX functions.

An Op and its already-computed parent values lower to either a JAX value or a
:class:`LoweredDistribution`. Graph-level operations combine that local interpretation with the
generic traversal operations in :mod:`pangolin.dag`.

`ir.Bijected` is evaluated using its stored IR bijector. Choosing unconstrained coordinates
is a separate graph transformation, not part of this backend's lowering policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import singledispatch
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy import special as jspecial
from jax.typing import ArrayLike
from numpyro import distributions as dist

from pangolin import dag, ir, util
from pangolin.ir import Op, RV
from pangolin.util import PyTree

__all__ = ["ancestor_log_prob", "ancestor_sample", "ancestor_sampler"]


# --------------------------------------------------------------------------------------------------
# Lowering
# --------------------------------------------------------------------------------------------------


class LoweredDistribution:
    """A bound JAX distribution represented by sampling and log-density functions."""

    def __init__(
        self,
        sample: Callable[[Array], Array],
        log_prob: Callable[[Array], Array],
    ):
        """Store callables that accept and return JAX arrays."""
        self.sample_fun = sample
        self.log_prob_fun = log_prob

    def sample(self, key: Array) -> Array:
        """Draw one sample using a JAX PRNG key."""
        return self.sample_fun(key)

    def log_prob(self, value: ArrayLike) -> Array:
        """Evaluate the log density or mass at a value."""
        return self.log_prob_fun(jnp.asarray(value))


type LoweringResult = Array | LoweredDistribution
type Lowering = Callable[[Op, tuple[Array, ...]], LoweringResult]


def index(array: Array, *indices: Array) -> Array:
    """Select one entry using exactly one scalar index per array dimension."""
    if len(indices) != array.ndim:
        raise ValueError("Index requires one index per array dimension")
    if any(index.ndim != 0 for index in indices):
        raise ValueError("Index requires scalar indices")
    return array[indices]


deterministic_functions: dict[type[Op], Callable[..., ArrayLike]] = {
    ir.Add: lambda a, b: a + b,
    ir.Sub: lambda a, b: a - b,
    ir.Mul: lambda a, b: a * b,
    ir.Div: lambda a, b: a / b,
    ir.Pow: lambda a, b: a**b,
    ir.Abs: jnp.abs,
    ir.Minimum: jnp.minimum,
    ir.Maximum: jnp.maximum,
    ir.Arccos: jnp.arccos,
    ir.Arccosh: jnp.arccosh,
    ir.Arcsin: jnp.arcsin,
    ir.Arcsinh: jnp.arcsinh,
    ir.Arctan: jnp.arctan,
    ir.Arctanh: jnp.arctanh,
    ir.Cos: jnp.cos,
    ir.Cosh: jnp.cosh,
    ir.Exp: jnp.exp,
    ir.Identity: lambda a: a,
    ir.InvLogit: jax.nn.sigmoid,
    ir.Log: jnp.log,
    ir.Loggamma: jspecial.gammaln,
    ir.Logit: jspecial.logit,
    ir.Sin: jnp.sin,
    ir.Sinh: jnp.sinh,
    ir.Step: lambda x: jnp.heaviside(x, 0.5),
    ir.Tan: jnp.tan,
    ir.Tanh: jnp.tanh,
    ir.Matmul: jnp.matmul,
    ir.Inv: jnp.linalg.inv,
    ir.Cholesky: jnp.linalg.cholesky,
    ir.Transpose: jnp.transpose,
    ir.Diag: jnp.diag,
    ir.DiagMatrix: jnp.diag,
    ir.Softmax: jax.nn.softmax,
    ir.Index: index,
}


def categorical_distribution(probs: ArrayLike) -> dist.Distribution:
    """Construct a NumPyro categorical distribution from probabilities."""
    distribution = dist.Categorical(probs=probs)
    assert distribution is not None
    return distribution


distribution_constructors: dict[type[Op], Callable[..., dist.Distribution]] = {
    ir.Normal: dist.Normal,
    ir.NormalPrec: lambda loc, prec: dist.Normal(loc, 1 / jnp.sqrt(prec)),
    ir.Bernoulli: dist.Bernoulli,
    ir.BernoulliLogit: dist.BernoulliLogits,
    ir.Beta: dist.Beta,
    ir.BetaBinomial: lambda n, a, b: dist.BetaBinomial(a, b, n),
    ir.Binomial: dist.Binomial,
    ir.Categorical: categorical_distribution,
    ir.Cauchy: dist.Cauchy,
    ir.Exponential: dist.Exponential,
    ir.Dirichlet: dist.Dirichlet,
    ir.Gamma: dist.Gamma,
    ir.Lognormal: dist.LogNormal,
    ir.Multinomial: dist.Multinomial,
    ir.MultiNormal: dist.MultivariateNormal,
    ir.Poisson: dist.Poisson,
    ir.StudentT: dist.StudentT,
    ir.Uniform: dist.Uniform,
    ir.Wishart: dist.Wishart,
}


def make_deterministic_lowering(function: Callable[..., ArrayLike]) -> Lowering:
    """Adapt a JAX function into a deterministic Op lowering."""

    def lowering(op: Op, parent_values: tuple[Array, ...]) -> Array:
        return jnp.asarray(function(*parent_values))

    return lowering


def make_distribution_lowering(constructor: Callable[..., dist.Distribution]) -> Lowering:
    """Adapt a NumPyro distribution constructor into a random Op lowering.

    Log density is minus infinity outside the distribution's support. Out-of-support values
    are replaced by a feasible value before evaluating the density, keeping inactive branches
    safe under differentiation and vectorization. NaN observations remain NaN.
    """

    def lowering(op: Op, parent_values: tuple[Array, ...]) -> LoweredDistribution:
        distribution = constructor(*parent_values)

        def sample(key: Array) -> Array:
            return jnp.asarray(distribution.sample(key))

        def log_prob(value: Array) -> Array:
            if distribution.support is None:
                return jnp.asarray(distribution.log_prob(value))
            supported = jnp.asarray(distribution.support(value))
            feasible_value = jnp.asarray(distribution.support.feasible_like(value))
            safe_value = jnp.where(supported, value, feasible_value)
            log_density = jnp.asarray(distribution.log_prob(safe_value))
            return jnp.where(
                jnp.any(jnp.isnan(value)),
                jnp.nan,
                jnp.where(supported, log_density, -jnp.inf),
            )

        return LoweredDistribution(sample, log_prob)

    return lowering


@singledispatch
def lower_unchecked(op: Op, parent_values: tuple[Array, ...]) -> LoweringResult:
    """Dispatch an Op lowering without checking its shape or randomness contract."""
    raise NotImplementedError(f"No JAX lowering is defined for {type(op).__name__}")


for op_type, function in deterministic_functions.items():
    lower_unchecked.register(op_type)(make_deterministic_lowering(function))

for op_type, constructor in distribution_constructors.items():
    lower_unchecked.register(op_type)(make_distribution_lowering(constructor))


def shape(value: ArrayLike) -> tuple[int, ...]:
    """Return the static JAX shape of an array-like value."""
    return jnp.shape(value)


def lower(op: Op, parent_values: Sequence[ArrayLike]) -> LoweringResult:
    """Bind an Op to its parent values as a JAX value or distribution."""
    parents = tuple(jnp.asarray(value) for value in parent_values)
    expected_shape = op.get_shape(*(value.shape for value in parents))
    lowered = lower_unchecked(op, parents)
    is_distribution = isinstance(lowered, LoweredDistribution)
    if is_distribution != op.random:
        raise TypeError(
            f"Lowering {type(op).__name__} returned "
            f"{'a distribution' if is_distribution else 'a value'}, contrary to the Op contract"
        )
    if isinstance(lowered, LoweredDistribution):
        distribution = lowered

        def sample(key: Array) -> Array:
            value = distribution.sample(key)
            assert value.shape == expected_shape, "Error: shape was not as expected"
            return value

        def log_prob(value: Array) -> Array:
            assert value.shape == expected_shape, "Error: shape was not as expected"
            return distribution.log_prob(value)

        return LoweredDistribution(sample, log_prob)

    assert lowered.shape == expected_shape, "Error: shape was not as expected"
    return lowered


def require_array(lowered: LoweringResult) -> Array:
    """Return a lowered JAX array, rejecting a distribution."""
    if isinstance(lowered, LoweredDistribution):
        raise TypeError("Expected a lowered array, got a distribution")
    return lowered


def require_distribution(lowered: LoweringResult) -> LoweredDistribution:
    """Return a lowered distribution, rejecting a JAX array."""
    if not isinstance(lowered, LoweredDistribution):
        raise TypeError("Expected a lowered distribution, got an array")
    return lowered


def lower_deterministic(op: Op, parents: Sequence[ArrayLike]) -> Array:
    """Lower an Op to a JAX array, rejecting a distribution result."""
    return require_array(lower(op, parents))


def lower_random(op: Op, parents: Sequence[ArrayLike]) -> LoweredDistribution:
    """Lower an Op to a distribution, rejecting an array result."""
    return require_distribution(lower(op, parents))


# --------------------------------------------------------------------------------------------------
# Constant
# --------------------------------------------------------------------------------------------------


@lower_unchecked.register
def lower_constant(op: ir.Constant, parent_values: tuple[Array, ...]) -> Array:
    """Lower a constant Op to its stored value."""
    return jnp.asarray(op.value)


# --------------------------------------------------------------------------------------------------
# Sum
# --------------------------------------------------------------------------------------------------


@lower_unchecked.register
def lower_sum(op: ir.Sum, parent_values: tuple[Array, ...]) -> Array:
    """Lower a sum Op along its configured axis."""
    (parent_value,) = parent_values
    return jnp.sum(parent_value, axis=op.axis)


# --------------------------------------------------------------------------------------------------
# Concatenate
# --------------------------------------------------------------------------------------------------


@lower_unchecked.register
def lower_concatenate(
    op: ir.Concatenate,
    parent_values: tuple[Array, ...],
) -> Array:
    """Lower concatenation along the configured axis."""
    return jnp.concatenate(parent_values, axis=op.axis)


# --------------------------------------------------------------------------------------------------
# LogProb
# --------------------------------------------------------------------------------------------------


@lower_unchecked.register
def lower_log_prob(op: ir.LogProb, parent_values: tuple[Array, ...]) -> Array:
    """Evaluate the base distribution's log density or mass at the first parent value."""
    value, *parents = parent_values
    distribution = lower_random(op.base_op, parents)
    return distribution.log_prob(value)


# --------------------------------------------------------------------------------------------------
# CDF
# --------------------------------------------------------------------------------------------------


@lower_unchecked.register
def lower_cdf(op: ir.CDF, parent_values: tuple[Array, ...]) -> Array:
    """Evaluate a joint CDF with all distribution parameters fixed.

    Supported primitives are Normal, NormalPrec, Beta, Cauchy, Exponential, Gamma, Lognormal,
    Poisson, StudentT, and Uniform. Composite evaluates its deterministic intermediates; VMap
    multiplies the CDFs of its conditionally independent draws. These rules compose recursively.
    Scalar Bijected uses its bijector's sign and inverse, reversing the probability for
    decreasing bijections. Evaluation is guaranteed only inside the transformed interval.
    Missing sign and non-scalar bijections are unsupported. Truncated normalizes the base CDF
    between its supplied bounds, which must be strictly interior.
    Other base Ops raise NotImplementedError, including Scan (whose joint CDF generally needs
    integration).
    """
    value, *parents = parent_values
    base = op.base_op
    if isinstance(base, ir.Composite):
        # Insert the evaluation value before the original inputs, shifting every old reference.
        par_nums = [tuple(n + 1 for n in nums) for nums in base.par_nums]
        par_nums[-1] = (0, *par_nums[-1])
        cdf_op = ir.Composite(base.num_inputs + 1, (*base.ops[:-1], ir.CDF(base.ops[-1])), par_nums)
        return lower_deterministic(cdf_op, parent_values)

    if isinstance(base, ir.VMap):
        mapped_cdf = ir.VMap(ir.CDF(base.base_op), (0, *base.in_axes), base.axis_size)
        return jnp.prod(lower_deterministic(mapped_cdf, parent_values))

    if isinstance(base, ir.Bijected):
        bijector = base.bijector
        if bijector.sign is None:
            raise NotImplementedError("CDF is not implemented for Bijected without sign")
        n = bijector.n_biject_params
        params = tuple(parents[:n])
        base_parents = tuple(parents[n:])
        input_shape = base.base_op.get_shape(*(p.shape for p in base_parents))
        if input_shape != () or value.shape != ():
            raise NotImplementedError("CDF is not implemented for non-scalar Bijected")

        sign = lower_deterministic(bijector.sign, params)
        x = lower_deterministic(bijector.inverse, (value, *params))
        probability = lower_deterministic(ir.CDF(base.base_op), (x, *base_parents))
        return jnp.where(sign > 0, probability, 1.0 - probability)

    if isinstance(base, ir.Truncated):
        if not base.lower and not base.upper:
            return lower_deterministic(ir.CDF(base.base_op), parent_values)
        lo, hi, base_parents, a, b, valid = truncation_parameters(base, tuple(parents))
        outside = (value <= lo) | (value >= hi)
        safe_value = jnp.where(outside, lo if base.lower else hi, value)
        probability = lower_deterministic(ir.CDF(base.base_op), (safe_value, *base_parents))
        probability = (probability - a) / jnp.where(valid, b - a, 1.0)
        probability = jnp.where(value <= lo, 0.0, jnp.where(value >= hi, 1.0, probability))
        return jnp.where(valid & ~jnp.isnan(value), probability, jnp.nan)

    numpyro_constructor = distribution_constructors.get(type(base))
    if numpyro_constructor is None:
        raise NotImplementedError(f"CDF is not implemented for {type(base).__name__}")
    numpyro_distribution = numpyro_constructor(*parents)
    numpyro_support = numpyro_distribution.support
    if (
        numpyro_support is None
        or numpyro_distribution.event_shape
        or type(numpyro_distribution).cdf is dist.Distribution.cdf
    ):
        raise NotImplementedError(f"CDF is not implemented for {type(base).__name__}")

    numpyro_lower_bound = getattr(numpyro_support, "lower_bound", -jnp.inf)
    numpyro_upper_bound = getattr(numpyro_support, "upper_bound", jnp.inf)
    if base.discrete:
        outside = (value < numpyro_lower_bound) | (value >= numpyro_upper_bound)
    else:
        outside = (value <= numpyro_lower_bound) | (value >= numpyro_upper_bound)
    safe_value = jnp.where(outside, jnp.asarray(numpyro_support.feasible_like(value)), value)
    numpyro_probability = numpyro_distribution.cdf(safe_value)
    probability = jnp.asarray(numpyro_probability)
    return jnp.where(
        value < numpyro_lower_bound if base.discrete else value <= numpyro_lower_bound,
        0.0,
        jnp.where(value >= numpyro_upper_bound, 1.0, probability),
    )


# --------------------------------------------------------------------------------------------------
# Quantile
# --------------------------------------------------------------------------------------------------


@lower_unchecked.register
def lower_quantile(op: ir.Quantile, parent_values: tuple[Array, ...]) -> Array:
    """Evaluate a scalar quantile with all distribution parameters fixed.

    Primitive distributions use NumPyro's inverse CDF: Normal, NormalPrec, Cauchy,
    Exponential, Lognormal, and Uniform are supported. Beta, Gamma, and StudentT quantiles
    are unsupported.
    Discrete distributions are currently unsupported by this backend, although the IR permits
    their quantiles; they raise NotImplementedError.

    Composite evaluates its deterministic intermediates. Scalar Bijected uses its bijector's
    sign and forward operation, reversing the probability for decreasing bijections.
    Truncated rescales the probability to the retained base-CDF interval. These rules compose
    recursively. Quantiles of array-valued distributions are rejected by the IR; use VMap
    around Quantile for independent evaluations. Other base Ops raise NotImplementedError.

    Probabilities outside [0, 1] return NaN, including under JIT. At zero and one, primitive
    quantiles use support endpoints. Bijections need not handle endpoint limits: transformed
    quantiles are guaranteed only for probabilities strictly between zero and one.
    """
    probability, *parents = parent_values
    base = op.base_op
    if jnp.iscomplexobj(probability):
        raise TypeError("Quantile requires a real probability")

    if isinstance(base, ir.Composite):
        par_nums = [tuple(n + 1 for n in nums) for nums in base.par_nums]
        par_nums[-1] = (0, *par_nums[-1])
        quantile_op = ir.Composite(
            base.num_inputs + 1, (*base.ops[:-1], ir.Quantile(base.ops[-1])), par_nums
        )
        return lower_deterministic(quantile_op, parent_values)

    valid = (probability >= 0) & (probability <= 1)
    safe_probability = jnp.where(valid, probability, 0.5)

    if isinstance(base, ir.Bijected):
        bijector = base.bijector
        if bijector.sign is None:
            raise NotImplementedError("Quantile is not implemented for Bijected without sign")
        n = bijector.n_biject_params
        params, base_parents = tuple(parents[:n]), tuple(parents[n:])
        if base.base_op.get_shape(*(p.shape for p in base_parents)) != ():
            raise NotImplementedError("Quantile is not implemented for non-scalar Bijected")
        sign = lower_deterministic(bijector.sign, params)
        base_probability = jnp.where(sign > 0, safe_probability, 1.0 - safe_probability)
        x = lower_deterministic(ir.Quantile(base.base_op), (base_probability, *base_parents))
        value = lower_deterministic(bijector.forward, (x, *params))
        return jnp.where(valid, value, jnp.nan)

    if isinstance(base, ir.Truncated):
        if not base.lower and not base.upper:
            return lower_deterministic(ir.Quantile(base.base_op), parent_values)
        _, _, base_parents, a, b, valid_bounds = truncation_parameters(base, tuple(parents))
        base_probability = jnp.where(valid_bounds, a + safe_probability * (b - a), 0.5)
        value = lower_deterministic(ir.Quantile(base.base_op), (base_probability, *base_parents))
        return jnp.where(valid & valid_bounds, value, jnp.nan)

    numpyro_constructor = distribution_constructors.get(type(base))
    if numpyro_constructor is None or isinstance(base, (ir.Beta, ir.Gamma, ir.StudentT)):
        raise NotImplementedError(f"Quantile is not implemented for {type(base).__name__}")
    numpyro_distribution = numpyro_constructor(*parents)
    numpyro_support = numpyro_distribution.support
    if (
        numpyro_support is None
        or numpyro_distribution.event_shape
        or type(numpyro_distribution).icdf is dist.Distribution.icdf
    ):
        raise NotImplementedError(f"Quantile is not implemented for {type(base).__name__}")
    interior_probability = jnp.where(
        (safe_probability > 0) & (safe_probability < 1), safe_probability, 0.5
    )
    value = jnp.asarray(numpyro_distribution.icdf(interior_probability))
    value = jnp.where(probability == 0, getattr(numpyro_support, "lower_bound", -jnp.inf), value)
    value = jnp.where(probability == 1, getattr(numpyro_support, "upper_bound", jnp.inf), value)
    return jnp.where(valid, value, jnp.nan)


# --------------------------------------------------------------------------------------------------
# VJP
# --------------------------------------------------------------------------------------------------


@lower_unchecked.register
def lower_vjp(op: ir.VJP, parent_values: tuple[Array, ...]) -> Array:
    """Differentiate the existing base lowering in one real floating-point input.

    The base output must also be real floating-point. The real cotangent is converted to its
    dtype; other inputs are held fixed and may have nondifferentiable dtypes, such as indices.
    """
    cotangent, *parents = parent_values
    selected = parents[op.argnum]
    if not jnp.issubdtype(selected.dtype, jnp.floating):
        raise TypeError("VJP requires a real floating-point differentiation input")

    def evaluate(value: Array) -> Array:
        inputs = parents.copy()
        inputs[op.argnum] = value
        return lower_deterministic(op.base_op, inputs)

    output, pullback = jax.vjp(evaluate, selected)
    if not jnp.issubdtype(output.dtype, jnp.floating):
        raise TypeError("VJP requires a real floating-point base output")
    if jnp.iscomplexobj(cotangent):
        raise TypeError("VJP requires a real cotangent")
    return pullback(cotangent.astype(output.dtype))[0]


# --------------------------------------------------------------------------------------------------
# Bijected
# --------------------------------------------------------------------------------------------------


@lower_unchecked.register
def lower_transformed(
    op: ir.Bijected,
    parent_values: tuple[Array, ...],
) -> LoweredDistribution:
    """Lower a transformed distribution using its base distribution and IR bijector.

    Sampling applies the forward map. Log density applies the inverse and the
    change-of-variables formula. Bijections operate between open domains: evaluation at
    endpoints or outside the inverse's domain is not guaranteed. Neither operation uses sign.
    """
    bijector = op.bijector
    n = bijector.n_biject_params
    params = parent_values[:n]
    base = lower_random(op.base_op, parent_values[n:])

    def sample(key: Array) -> Array:
        x = base.sample(key)
        return lower_deterministic(bijector.forward, (x, *params))

    def log_prob(y: Array) -> Array:
        x = lower_deterministic(bijector.inverse, (y, *params))
        ldj = lower_deterministic(bijector.log_det_jac, (x, y, *params))
        return base.log_prob(x) - ldj

    return LoweredDistribution(sample, log_prob)


# --------------------------------------------------------------------------------------------------
# Truncated
# --------------------------------------------------------------------------------------------------


def truncation_parameters(
    op: ir.Truncated, parent_values: tuple[Array, ...]
) -> tuple[Array, Array, tuple[Array, ...], Array, Array, Array]:
    """Read optional bounds and evaluate their base CDFs and validity.

    Return the lower and upper bounds, base parents, lower and upper probabilities, and a
    validity flag. Omitted bounds use infinities and probabilities zero or one without
    evaluating a CDF. Each supplied bound must be finite with CDF strictly between zero and
    one; both bounds and probabilities must be strictly ordered. All four interpretations
    of Truncated use this check, including when nested in another distribution.
    """
    index = 0
    lo = parent_values[index] if op.lower else jnp.asarray(-jnp.inf)
    index += int(op.lower)
    hi = parent_values[index] if op.upper else jnp.asarray(jnp.inf)
    index += int(op.upper)
    parents = parent_values[index:]
    a = lower_deterministic(ir.CDF(op.base_op), (lo, *parents)) if op.lower else jnp.asarray(0.0)
    b = lower_deterministic(ir.CDF(op.base_op), (hi, *parents)) if op.upper else jnp.asarray(1.0)
    valid = (lo < hi) & (a < b)
    if op.lower:
        valid = valid & jnp.isfinite(lo) & (a > 0) & (a < 1)
    if op.upper:
        valid = valid & jnp.isfinite(hi) & (b > 0) & (b < 1)
    return lo, hi, parents, a, b, valid


@lower_unchecked.register
def lower_truncated(op: ir.Truncated, parent_values: tuple[Array, ...]) -> LoweredDistribution:
    """Lower a continuous scalar distribution restricted by optional interior bounds.

    Each supplied bound must be finite and strictly inside the base distribution's interval.
    Log density subtracts the log retained probability; sampling evaluates the base Quantile
    at a uniform probability between the bound CDFs. Composite, Bijected, and nested Truncated
    bases use these same rules recursively. Each nested truncation validates its own bounds.

    Sampling requires Quantile support; log density requires only the base CDF and density.
    With neither bound supplied, delegate to the base without requiring either CDF or Quantile.
    Unsupported operations raise NotImplementedError.

    Invalid bounds produce NaN samples and log densities, including under JIT. Density is
    zero outside the retained interval. Direct CDF subtraction can lose precision in extreme
    tails or narrow intervals; rounded endpoint probabilities are rejected as non-interior.
    No specialized tail-stable normalization or sampler is provided.
    """
    if not op.lower and not op.upper:
        return lower_random(op.base_op, parent_values)
    lo, hi, parents, a, b, valid = truncation_parameters(op, parent_values)
    distribution = lower_random(op.base_op, parents)
    log_mass = jnp.log(jnp.where(valid, b - a, 1.0))

    def sample(key: Array) -> Array:
        probability = jax.random.uniform(key, minval=a, maxval=b)
        probability = jnp.where(valid, probability, 0.5)
        value = lower_deterministic(ir.Quantile(op.base_op), (probability, *parents))
        return jnp.where(valid, value, jnp.nan)

    def log_prob(value: Array) -> Array:
        inside = (value >= lo) & (value <= hi)
        safe_value = jnp.where(inside, value, lo if op.lower else hi)
        log_density = distribution.log_prob(safe_value) - log_mass
        return jnp.where(
            valid & ~jnp.isnan(value),
            jnp.where(inside, log_density, -jnp.inf),
            jnp.nan,
        )

    return LoweredDistribution(sample, log_prob)


# --------------------------------------------------------------------------------------------------
# Composite
# --------------------------------------------------------------------------------------------------


@lower_unchecked.register
def lower_composite(op: ir.Composite, parent_values: tuple[Array, ...]) -> LoweringResult:
    """Lower a Composite by recursively lowering its inner Ops in order."""
    values = list(parent_values)
    for op_number, (inner_op, parent_numbers) in enumerate(zip(op.ops, op.par_nums, strict=True)):
        inner_parents = [values[number] for number in parent_numbers]
        lowered = lower(inner_op, inner_parents)
        if op_number == len(op.ops) - 1:
            return lowered
        if isinstance(lowered, LoweredDistribution):
            raise TypeError("Only the final Op in a Composite may be random")
        values.append(lowered)
    raise AssertionError("Composite must contain an output Op")


# --------------------------------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------------------------------


def prepare_scan_inputs(
    op: ir.Scan, parent_values: tuple[Array, ...]
) -> tuple[Array, tuple[Array, ...], Callable[[Sequence[Array]], tuple[Array, ...]]]:
    """Move mapped scan axes forward and return a function that restores argument order."""
    init = parent_values[0]
    rest = parent_values[1:]
    mapped = tuple(
        jnp.moveaxis(value, axis, 0)
        for value, axis in zip(rest, op.in_axes, strict=True)
        if axis is not None
    )
    unmapped = tuple(value for value, axis in zip(rest, op.in_axes, strict=True) if axis is None)

    def merge(mapped_values: Sequence[Array]) -> tuple[Array, ...]:
        mapped_iterator = iter(mapped_values)
        unmapped_iterator = iter(unmapped)
        return tuple(
            next(mapped_iterator) if axis is not None else next(unmapped_iterator)
            for axis in op.in_axes
        )

    return init, mapped, merge


@lower_unchecked.register
def lower_scan(op: ir.Scan, parent_values: tuple[Array, ...]) -> LoweringResult:
    """Lower a deterministic or random Scan through ``jax.lax.scan``."""
    init, mapped, merge = prepare_scan_inputs(op, parent_values)

    if not op.random:

        def step(carry: Array, mapped_values: Sequence[Array]) -> tuple[Array, Array]:
            value = lower_deterministic(op.base_op, (carry, *merge(mapped_values)))
            return value, value

        _, values = jax.lax.scan(step, init, mapped, length=op.length)
        return values

    def sample(key: Array) -> Array:
        keys = jax.random.split(key, op.length)

        def step(carry: Array, key_and_values: Sequence[Array]) -> tuple[Array, Array]:
            subkey = key_and_values[0]
            mapped_values = key_and_values[1:]
            distribution = lower_random(op.base_op, (carry, *merge(mapped_values)))
            value = distribution.sample(subkey)
            return value, value

        _, values = jax.lax.scan(step, init, (keys, *mapped), length=op.length)
        return values

    def log_prob(values: Array) -> Array:

        def step(carry: Array, value_and_mapped: Sequence[Array]) -> tuple[Array, Array]:
            value = value_and_mapped[0]
            mapped_values = value_and_mapped[1:]
            distribution = lower_random(op.base_op, (carry, *merge(mapped_values)))
            term = distribution.log_prob(value)
            return value, term

        _, terms = jax.lax.scan(step, init, (values, *mapped), length=op.length)
        return jnp.sum(terms)

    return LoweredDistribution(sample, log_prob)


# --------------------------------------------------------------------------------------------------
# VMap
# --------------------------------------------------------------------------------------------------


@lower_unchecked.register
def lower_vmap(op: ir.VMap, parent_values: tuple[Array, ...]) -> LoweringResult:
    """Lower a deterministic or random VMap through ``jax.vmap``."""
    if not op.random:

        def evaluate(*parents: Array) -> Array:
            return lower_deterministic(op.base_op, parents)

        return jax.vmap(evaluate, in_axes=op.in_axes, axis_size=op.axis_size)(*parent_values)

    output_axis_size = op.get_shape(*(parent.shape for parent in parent_values))[0]

    def sample(key: Array) -> Array:
        keys = jax.random.split(key, output_axis_size)

        def sample_base(subkey: Array, *parents: Array) -> Array:
            distribution = lower_random(op.base_op, parents)
            return distribution.sample(subkey)

        return jax.vmap(sample_base, in_axes=(0, *op.in_axes), axis_size=op.axis_size)(
            keys, *parent_values
        )

    def log_prob(values: Array) -> Array:
        def log_prob_base(value: Array, *parents: Array) -> Array:
            distribution = lower_random(op.base_op, parents)
            return distribution.log_prob(value)

        terms = jax.vmap(log_prob_base, in_axes=(0, *op.in_axes), axis_size=op.axis_size)(
            values, *parent_values
        )
        return jnp.sum(terms)

    return LoweredDistribution(sample, log_prob)


# --------------------------------------------------------------------------------------------------
# Graph operations
# --------------------------------------------------------------------------------------------------


def ancestor_sample_graph(
    vars: PyTree[RV], key: Array | None, boundary: Mapping[RV, Array] | None = None
) -> PyTree[Array]:
    """Evaluate a graph, sampling random nodes not supplied by the boundary."""
    boundary_values = {}
    for var, value in (boundary or {}).items():
        value = jnp.asarray(value)
        if shape(value) != var.shape:
            raise ValueError(
                f"Boundary value for {var} has shape {shape(value)}, expected {var.shape}"
            )
        boundary_values[var] = value

    def step(var: RV, parent_values: tuple[Array, ...]):
        nonlocal key
        lowered = lower(var.op, parent_values)
        if isinstance(lowered, LoweredDistribution):
            if key is None:
                raise ValueError(
                    f"Random variable {var} has no given value; pass key= to sample it"
                )
            key, subkey = jax.random.split(key)
            value = lowered.sample(subkey)
        else:
            value = lowered
        return value, None

    values, _ = dag.graph_scan(vars, step, boundary=boundary_values)
    return values


def ancestor_sample_flat(
    vars: list[RV], key: Array | None = None, size: int | None = None
) -> list[Array]:
    """Compatibility wrapper for flat backend tests and consumers."""
    return ancestor_sample(vars, key, size)


def ancestor_log_prob_flat(vars: Sequence[RV], values: Sequence[ArrayLike]) -> ArrayLike:
    """Compatibility wrapper for flat backend tests and consumers."""
    return ancestor_log_prob(*vars)(*values)


def fill_in(
    random_vars: Sequence[RV],
    random_values: Sequence[ArrayLike],
    desired_vars: Sequence[RV],
    key: Array | None = None,
) -> list[Array]:
    """Compute desired values from supplied random values, optionally sampling missing ones."""
    if len(random_vars) != len(random_values):
        raise ValueError("random_vars and random_values must have same length")
    if any(not var.op.random for var in random_vars):
        raise ValueError("random_vars must contain only random variables")

    boundary = {
        var: jnp.asarray(value) for var, value in zip(random_vars, random_values, strict=True)
    }
    return ancestor_sample_graph(list(desired_vars), key, boundary)


# --------------------------------------------------------------------------------------------------
# Public PyTree API
# --------------------------------------------------------------------------------------------------


def ancestor_sample(
    vars: PyTree[RV], key: Array | None = None, size: int | None = None
) -> PyTree[Array]:
    """Draw ancestral samples while preserving the requested RV PyTree structure.

    Parameters
    ----------
    vars
        PyTree of RVs to sample.
    key
        JAX random key, or None to generate a key automatically.
    size
        Number of samples, or None to draw a single sample.

    Returns
    -------
    out
        PyTree matching ``vars``, with JAX arrays in place of RVs. When ``size`` is
        provided, each array has an additional leading dimension of that size.

    Examples
    --------
    Sample a constant RV.

    >>> x = RV(ir.Constant(1.5))
    >>> ancestor_sample(x)
    Array(1.5, dtype=float...)

    Sample a PyTree with the RV inside it.

    >>> ancestor_sample({'sup': [[x]]})
    {'sup': [[Array(1.5, dtype=float...)]]}

    Draw several samples.

    >>> ancestor_sample(x, size=3)
    Array([1.5, 1.5, 1.5], dtype=float...)

    Draw several samples from a PyTree with an RV inside it.

    >>> ancestor_sample({'sup': x}, size=3)
    {'sup': Array([1.5, 1.5, 1.5], dtype=float...)}

    Sample from several random variables at once.

    >>> y = RV(ir.Add(), x, x)
    >>> z = RV(ir.Mul(), x, y)
    >>> print(ancestor_sample({'cat': x, 'dog': [y, z]}))
    {'cat': Array(1.5, dtype=float...), 'dog': [Array(3., dtype=float...), Array(4.5, dtype=float...)]}
    """
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(0, 2**32 - 1))
    if size is None:
        return ancestor_sample_graph(vars, key)
    keys = jax.random.split(key, size)
    return jax.vmap(lambda sample_key: ancestor_sample_graph(vars, sample_key))(keys)


def ancestor_sampler(vars: PyTree[RV]) -> Callable[[Array], PyTree[Array]]:
    """Create a keyed ancestral sampler for an RV PyTree.

    Parameters
    ----------
    vars
        PyTree of RVs to sample.

    Returns
    -------
    sampler
        Function taking a JAX random key and returning a joint sample as a PyTree
        of JAX arrays matching the structure and shapes of ``vars``.

    Examples
    --------
    >>> x = RV(ir.Constant(1.5))
    >>> y = RV(ir.Add(), x, x)
    >>> fun = ancestor_sampler([{'cat': x}, y])

    You now have a plain-old JAX function. You can evaluate it.

    >>> key = jax.random.PRNGKey(0)
    >>> fun(key)
    [{'cat': Array(1.5, dtype=float...)}, Array(3., dtype=float...)]

    You can do normal JAX stuff with it, e.g. vmap it.

    >>> print(jax.vmap(fun)(jax.random.split(key, 3)))
    [{'cat': Array([1.5, 1.5, 1.5], dtype=float...)}, Array([3., 3., 3.], dtype=float...)]
    """

    def sampler(key: Array) -> PyTree[Array]:
        return ancestor_sample(vars, key)

    return sampler


def ancestor_log_prob(*vars: PyTree[RV], **kwvars: PyTree[RV]) -> Callable[..., ArrayLike]:
    """Create a joint log-probability function for a PyTree of random variables.

    Parameters
    ----------
    *vars
        PyTrees of random RVs whose values will be supplied positionally.
    **kwvars
        PyTrees of random RVs whose values will be supplied by keyword.
        Together with ``vars``, these must include every upstream random RV.

    Returns
    -------
    log_prob
        Function accepting values matching the positional and keyword RV PyTrees
        and returning their scalar joint log probability as a JAX array.

    Examples
    --------
    >>> loc = ir.RV(ir.Constant(0.0))
    >>> scale = ir.RV(ir.Constant(1.0))
    >>> x = RV(ir.Normal(), loc, scale)
    >>> fun = ancestor_log_prob(x)

    You now have a plain JAX function. You can evaluate it.

    >>> fun(0.0)
    Array(-0.9189385..., dtype=float...)

    Or you can vmap it.

    >>> jax.vmap(fun)(jnp.array([0.0, 0.5]))
    Array([-0.9189385..., -1.0439385...], dtype=float...)

    Here's a more complex example:

    >>> op = ir.VMap(ir.Normal(), [None, None], 3)
    >>> y = RV(op, loc, scale)
    >>> fun = ancestor_log_prob({'x': x, 'y': y})
    >>> fun({'x': 0.0, 'y': [0.0, 0.5, 0.1]})
    Array(-3.805..., dtype=float...)

    You can also create a function that uses positional and/or keyword arguments:

    >>> fun = ancestor_log_prob(x, cat=y)
    >>> fun(0.0, cat=[0.0, 0.5, 0.1])
    Array(-3.805..., dtype=float...)
    """
    all_vars = (vars, kwvars)

    def log_prob(*values: PyTree[ArrayLike], **kwvalues: PyTree[ArrayLike]) -> ArrayLike:
        all_values = util.assimilate_vals(all_vars, (values, kwvalues))
        rv_leaves, value_leaves = util.dual_flatten(all_vars, all_values, allow_prefix=False)
        given = dict(zip(rv_leaves, value_leaves, strict=True))

        if not given:
            return jnp.asarray(0.0)

        def step(var: RV, parent_values: tuple[ArrayLike, ...]):
            lowered = lower(var.op, parent_values)
            if isinstance(lowered, LoweredDistribution):
                if var not in given:
                    raise ValueError(f"Random variable {var} has no supplied value")
                value = given[var]
                emission = lowered.log_prob(value)
            else:
                if var in given:
                    raise ValueError("Can't provide a value for a non-random variable")
                value = lowered
                emission = 0.0
            return value, emission

        return dag.graph_fold(all_vars, step)

    return log_prob
