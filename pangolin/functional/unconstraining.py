"""Provide backend-independent unconstraining of inference-problem IR graphs."""

from collections.abc import Callable

import numpy as np

from pangolin import dag, ir
from pangolin.util import PyTree

from . import tforms
from .indexing import index_orthogonal
from .scanning import make_abstract_rvs as make_scan_abstract_rvs
from .scanning import scan
from .tracing import AbstractOp, graph_scan, require_rv_leaves, trace_call
from .transforming import Transform, apply_transform, compose_transforms
from .vmapping import make_abstract_rvs, vmap

__all__ = ["transform_for", "unconstrain"]


# --------------------------------------------------------------------------------------------------
# Primitive Op transforms
# --------------------------------------------------------------------------------------------------


def transform_for_primitive(
    op: ir.Op,
    parents: tuple[ir.RV, ...],
    /,
) -> Transform | None:
    """Return Pangolin's default Transform for one primitive random Op."""
    op_type = type(op)
    if op_type in (ir.Cauchy, ir.MultiNormal, ir.Normal, ir.NormalPrec, ir.StudentT):
        return None
    if op_type in (ir.Exponential, ir.Gamma, ir.Lognormal):
        return tforms.log
    if op_type is ir.Beta:
        return tforms.logit
    if op_type is ir.Dirichlet:
        return tforms.softmax_centered_ilr
    if op_type is ir.Uniform:
        lower, upper = parents
        return tforms.scaled_logit(lower, upper)
    if op_type is ir.Wishart:
        return tforms.unconstrain_spd

    raise NotImplementedError(f"No default transform is implemented for {op_type.__name__}")


# --------------------------------------------------------------------------------------------------
# Composite Op transforms
# --------------------------------------------------------------------------------------------------


def transform_for_composite(
    op: ir.Composite,
    parents: tuple[ir.RV, ...],
    /,
) -> Transform | None:
    """Return a Transform for a Composite Op using the default recursive policy."""
    values = list(parents)
    for inner_op, parent_numbers in zip(op.ops[:-1], op.par_nums[:-1], strict=True):
        inner_parents = tuple(values[number] for number in parent_numbers)
        if parents:
            values.append(ir.RV(inner_op, *inner_parents, interface=parents[0].interface))
        else:
            values.append(ir.RV(inner_op, *inner_parents))

    final_op = op.ops[-1]
    final_parents = tuple(values[number] for number in op.par_nums[-1])
    return transform_for(final_op, final_parents)


# --------------------------------------------------------------------------------------------------
# VMap Op transforms
# --------------------------------------------------------------------------------------------------


def transform_for_vmap(
    op: ir.VMap,
    parents: tuple[ir.RV, ...],
    /,
) -> Transform | None:
    """Lift the recursively selected base Transform pointwise through a VMap Op."""
    abstract_parents, axis_size = make_abstract_rvs(
        list(parents),
        list(op.in_axes),
        op.axis_size,
    )
    base_transform = transform_for(op.base_op, tuple(abstract_parents))
    if base_transform is None:
        return None

    def transform_for_base(base_parents: tuple[ir.RV, ...]) -> Transform:
        transform = transform_for(op.base_op, base_parents)
        if transform is None:
            raise ValueError("VMap base transform policy returned inconsistent results")
        return transform

    def forward(x: ir.RV) -> ir.RV:
        def base_forward(value: ir.RV, *base_parents: ir.RV) -> ir.RV:
            return transform_for_base(base_parents).forward(value)

        mapped_forward = vmap(base_forward, (0, *op.in_axes), axis_size)
        return mapped_forward(x, *parents)

    def inverse(y: ir.RV) -> ir.RV:
        def base_inverse(value: ir.RV, *base_parents: ir.RV) -> ir.RV:
            return transform_for_base(base_parents).inverse(value)

        mapped_inverse = vmap(base_inverse, (0, *op.in_axes), axis_size)
        return mapped_inverse(y, *parents)

    def log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
        def base_log_det(
            base_x: ir.RV,
            base_y: ir.RV,
            *base_parents: ir.RV,
        ) -> ir.RV:
            return transform_for_base(base_parents).log_det_jac(base_x, base_y)

        mapped = vmap(base_log_det, (0, 0, *op.in_axes), axis_size)
        mapped_log_det = mapped(x, y, *parents)
        return ir.RV(ir.Sum(0), mapped_log_det, interface=mapped_log_det.interface)

    return Transform(forward, inverse, log_det_jac)


# --------------------------------------------------------------------------------------------------
# Scan Op transforms
# --------------------------------------------------------------------------------------------------


def lagged_carries(start: ir.RV, values: ir.RV, length: int, /) -> ir.RV:
    """Prepend a Scan's initial carry and remove its final output."""
    expected_shape = (length, *start.shape)
    if values.shape != expected_shape:
        raise ValueError(
            f"Scan values must have shape {expected_shape} to construct lagged carries; "
            f"got {values.shape}"
        )

    expanded_start = ir.RV(
        ir.VMap(ir.Identity(), (None,), axis_size=1),
        start,
        interface=start.interface,
    )
    if length == 1:
        return expanded_start

    index_values = (np.arange(length - 1), *(np.arange(size) for size in start.shape))
    indices = tuple(ir.RV(ir.Constant(value), interface=values.interface) for value in index_values)
    prefix = index_orthogonal(values, *indices, interface=values.interface)
    return ir.RV(ir.Concatenate(), expanded_start, prefix, interface=values.interface)


def transform_for_scan(
    op: ir.Scan,
    parents: tuple[ir.RV, ...],
    /,
) -> Transform | None:
    """Lift the recursively selected base Transform through a Scan Op.

    The forward transform and its log-Jacobian use a lagged sequence containing the initial carry
    followed by every constrained output except the last. Their per-step graphs are lifted with
    `vmap`, and the log-Jacobians are summed. The inverse uses a deterministic `scan` to reconstruct
    the constrained carry recursively.
    """
    start, *inputs = parents
    abstract_carry = ir.RV(AbstractOp(start.shape), interface=start.interface)
    abstract_inputs = make_scan_abstract_rvs(list(inputs), list(op.in_axes), op.length)
    base_transform = transform_for(op.base_op, (abstract_carry, *abstract_inputs))
    if base_transform is None:
        return None

    abstract_value = ir.RV(AbstractOp(start.shape), interface=start.interface)
    _, _, forward_boundary = trace_call(base_transform.forward, abstract_value)
    captured = [value for value in forward_boundary if value is not abstract_value]
    carry_dependent = abstract_carry in dag.upstream_nodes(captured)

    def step_transform(carry: ir.RV, base_inputs: tuple[ir.RV, ...]) -> Transform:
        transform = transform_for(op.base_op, (carry, *base_inputs))
        if transform is None:
            raise ValueError("Scan base transform policy returned inconsistent results")
        return transform

    def forward(x: ir.RV) -> ir.RV:
        previous = lagged_carries(start, x, op.length)

        def base_forward(value: ir.RV, carry: ir.RV, *base_inputs: ir.RV) -> ir.RV:
            return step_transform(carry, base_inputs).forward(value)

        mapped_forward = vmap(base_forward, (0, 0, *op.in_axes), op.length)
        return mapped_forward(x, previous, *inputs)

    def inverse(y: ir.RV) -> ir.RV:
        if carry_dependent:

            def recursive_inverse(carry: ir.RV, arguments: tuple[ir.RV, ...]) -> ir.RV:
                value, *base_inputs = arguments
                return step_transform(carry, tuple(base_inputs)).inverse(value)

            scanned_inverse = scan(recursive_inverse, op.length, (0, *op.in_axes))
            return scanned_inverse(start, (y, *inputs))

        def pointwise_inverse(arguments: tuple[ir.RV, ...]) -> ir.RV:
            value, *base_inputs = arguments
            return step_transform(start, tuple(base_inputs)).inverse(value)

        mapped_inverse = vmap(pointwise_inverse, (0, *op.in_axes), op.length)
        return mapped_inverse((y, *inputs))

    def log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
        def base_log_det(
            base_x: ir.RV,
            base_y: ir.RV,
            carry: ir.RV,
            *base_inputs: ir.RV,
        ) -> ir.RV:
            return step_transform(carry, base_inputs).log_det_jac(base_x, base_y)

        previous = lagged_carries(start, x, op.length)
        mapped = vmap(base_log_det, (0, 0, 0, *op.in_axes), op.length)
        mapped_log_det = mapped(x, y, previous, *inputs)
        return ir.RV(ir.Sum(0), mapped_log_det, interface=mapped_log_det.interface)

    return Transform(forward, inverse, log_det_jac)


# --------------------------------------------------------------------------------------------------
# Bijected Op transforms
# --------------------------------------------------------------------------------------------------


def transform_for_transformed(
    op: ir.Bijected,
    parents: tuple[ir.RV, ...],
    /,
) -> Transform | None:
    """Unconstrain a Bijected Op by undoing its bijection and handling its base Op.

    If the stored bijection maps a base value ``x`` to the Bijected value ``y = T(x)``, first
    applying ``T.inverse`` returns to the known support of ``op.base_op``. The recursive
    ``transform_for`` policy can then map ``x`` to unconstrained coordinates. Thus the resulting
    Transform is ``base_transform ∘ T.inverse``; no knowledge of the support of ``y`` is needed.

    The first ``op.bijector.n_biject_params`` parents parameterize ``T`` and the remaining parents
    parameterize ``op.base_op``. If the base needs no Transform, ``T.inverse`` alone maps ``y``
    back to the base's already-unconstrained coordinates.
    """
    num_bijector_parents = op.bijector.n_biject_params
    bijector_parents = parents[:num_bijector_parents]
    distribution_parents = parents[num_bijector_parents:]

    def forward(x: ir.RV) -> ir.RV:
        return ir.RV(op.bijector.forward, x, *bijector_parents, interface=x.interface)

    def inverse(y: ir.RV) -> ir.RV:
        return ir.RV(op.bijector.inverse, y, *bijector_parents, interface=y.interface)

    def log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
        return ir.RV(
            op.bijector.log_det_jac,
            x,
            y,
            *bijector_parents,
            interface=x.interface,
        )

    def sign() -> ir.RV:
        assert op.bijector.sign is not None
        if bijector_parents:
            return ir.RV(
                op.bijector.sign, *bijector_parents, interface=bijector_parents[0].interface
            )
        return ir.RV(op.bijector.sign)

    stored_transform = Transform(
        forward, inverse, log_det_jac, sign=sign if op.bijector.sign is not None else None
    )
    base_transform = transform_for(op.base_op, distribution_parents)
    if base_transform is None:
        return stored_transform.reverse
    return compose_transforms([stored_transform.reverse, base_transform])


# --------------------------------------------------------------------------------------------------
# Truncated Op transforms
# --------------------------------------------------------------------------------------------------


def transform_for_truncated(
    op: ir.Truncated,
    parents: tuple[ir.RV, ...],
    /,
) -> Transform | None:
    """Unconstrain a Truncated Op using its strictly interior bounds.

    Two supplied bounds define the whole support, so a scaled logit suffices without a base
    transform. With neither bound, use the base transform unchanged. With one bound, first
    unconstrain the base and then take the logarithm of the positive distance from the
    transformed bound. The base transform's sign determines which direction is positive;
    one-sided truncation therefore requires a scalar monotonic base transform with a known sign.

    This uses Truncated's contract that every supplied bound is finite and strictly interior
    to the base support. The backend validates that contract when evaluating the distribution;
    this function neither evaluates CDFs nor derives support endpoints.
    """
    if op.lower and op.upper:
        return tforms.scaled_logit(parents[0], parents[1])

    base_parents = parents[int(op.lower) + int(op.upper) :]
    base_transform = transform_for(op.base_op, base_parents)
    if not op.lower and not op.upper:
        return base_transform

    if base_transform is None:
        base_transform = compose_transforms([])
    if base_transform.sign is None:
        raise NotImplementedError(
            "One-sided Truncated unconstraining requires a base transform with a known sign"
        )

    bound = base_transform.forward(parents[0])
    direction = base_transform.sign()
    if op.upper:
        direction = tforms.negative(direction)
    return compose_transforms(
        [
            base_transform,
            tforms.shift_by(tforms.negative(bound)),
            tforms.scale_by(direction),
            tforms.log,
        ]
    )


# --------------------------------------------------------------------------------------------------
# Default transform selection
# --------------------------------------------------------------------------------------------------


def transform_for(
    op: ir.Op,
    parents: tuple[ir.RV, ...],
    /,
) -> Transform | None:
    """Return Pangolin's default constrained-to-unconstrained Transform.

    The primitive Op policy is:

    - ``Normal``, ``NormalPrec``, ``Cauchy``, ``StudentT``, and ``MultiNormal`` are unchanged.
    - ``Exponential``, ``Gamma``, and ``Lognormal`` use ``tforms.log``.
    - ``Beta`` uses ``tforms.logit``.
    - ``Uniform`` uses ``tforms.scaled_logit`` with its lower and upper parents.
    - ``Dirichlet`` uses ``tforms.softmax_centered_ilr``.
    - ``Wishart`` uses ``tforms.unconstrain_spd``.
    - ``Bernoulli``, ``BernoulliLogit``, ``Binomial``, ``Poisson``, ``BetaBinomial``,
      ``Categorical``, and ``Multinomial`` raise ``NotImplementedError`` because they are discrete.

    Container Ops recursively use the same policy:

    - ``Composite`` uses the policy for its final Op.
    - ``VMap`` applies the base transform independently at each mapped position.
    - ``Scan`` applies the base transform at each step using the previous constrained carry. If
      inversion depends on that carry, a deterministic Scan reconstructs the trajectory in order.
    - For ``Bijected``, its stored bijection ``T`` maps a base value ``x`` to the represented
      value ``y``. Unconstraining first recovers ``x = T.inverse(y)``, then unconstrains ``x`` using
      the policy for the base Op. If the base Op is already unconstrained, the transform is simply
      ``T.inverse``.
    - ``Truncated`` with two bounds uses a scaled logit. With one bound, it unconstrains the
      base and takes the log of the positive distance from the transformed bound, using the
      base transform's sign. With neither bound, it uses the base transform unchanged.

    These rules recurse through arbitrary nested combinations of ``Composite``, ``VMap``, ``Scan``,
    ``Bijected``, and ``Truncated``.

    An unsupported continuous Op raises ``NotImplementedError``, and a deterministic Op raises
    ``ValueError``.

    Parameters
    ----------
    op
        Random Op whose distribution is to be unconstrained.
    parents
        RVs supplying the Op's parameters, in their usual order.

    Returns
    -------
    Transform or None
        A Transform mapping the distribution to unconstrained coordinates, or None if it
        is already unconstrained. The transform may capture parent RVs.

    Notes
    -----
    The helpers recursively call this default selector. Passing a custom selector to
    `unconstrain` does not change their recursive choices.
    """
    op.get_shape(*(parent.shape for parent in parents))
    if not op.random:
        raise ValueError(f"Cannot unconstrain deterministic Op {op}")
    if op.discrete:
        raise NotImplementedError(f"Cannot unconstrain discrete Op {op}")
    if isinstance(op, ir.Composite):
        return transform_for_composite(op, parents)
    if isinstance(op, ir.VMap):
        return transform_for_vmap(op, parents)
    if isinstance(op, ir.Scan):
        return transform_for_scan(op, parents)
    if isinstance(op, ir.Bijected):
        return transform_for_transformed(op, parents)
    if isinstance(op, ir.Truncated):
        return transform_for_truncated(op, parents)
    return transform_for_primitive(op, parents)


# --------------------------------------------------------------------------------------------------
# Inference-problem rewriting
# --------------------------------------------------------------------------------------------------


def unconstrain(
    vars: PyTree[ir.RV],
    given: PyTree[ir.RV],
    *,
    transform_for: Callable[[ir.Op, tuple[ir.RV, ...]], Transform | None] = transform_for,
) -> tuple[PyTree[ir.RV], PyTree[ir.RV]]:
    """Rewrite an inference problem with unconstrained latent RVs.

    The inference problem is the joint ancestor graph of ``vars`` and ``given``. The graph is
    rebuilt parent-first without modifying the input graph. Deterministic nodes and random nodes
    appearing as leaves of ``given`` retain their Ops. For each other random node,
    ``transform_for(op, parents)`` selects a constrained-to-unconstrained Transform using its
    rewritten parents. None leaves the node unchanged; otherwise ``apply_transform`` creates
    unconstrained randomness and a deterministic reconstruction of the original value.
    Shared nodes remain shared, and the two input PyTree structures are preserved.

    Parameters
    ----------
    vars
        PyTree of RVs whose joint distribution is of interest.
    given
        PyTree of observed RVs; these exact nodes are not passed to the selector.
    transform_for
        Function accepting an Op and its parent RVs and returning a `Transform` or None.
        The default is `pangolin.functional.transform_for`. A custom selector may delegate
        an Op to the default selector or implement its own handling of composed Ops.
        Delegating a composed Op uses the default choices throughout that Op.

    Returns
    -------
    new_vars
        Rewritten ``vars`` with the same PyTree structure and conditional distribution.
    new_given
        Rewritten ``given`` with the same PyTree structure. Observed random nodes retain
        their original Ops while receiving reconstructed latent parents.

    Raises
    ------
    TypeError
        If either input has a non-RV leaf or the selector returns something other than a
        Transform or None.
    NotImplementedError
        If the default selector encounters a discrete or unsupported continuous Op.
    Exception
        Exceptions from the selector or transform construction are propagated.

    Notes
    -----
    A supplied Transform must be a valid bijection from the distribution's domain onto
    unconstrained coordinates, with the correct log-Jacobian. It may change the coordinate
    shape, but its inverse must reconstruct the original shape. None preserves the
    distribution without asserting that its domain is unconstrained.

    Examples
    --------
    Consider a model with a positive latent scale and an observed Normal value:

    >>> from pangolin import functional, ir
    >>> rate = ir.RV(ir.Constant(1.0))
    >>> location = ir.RV(ir.Constant(0.0))
    >>> scale = ir.RV(ir.Exponential(), rate)
    >>> observation = ir.RV(ir.Normal(), location, scale)
    >>> ir.print_upstream(observation, {"scale": scale, "observation": observation})
    shape | statement
    ----- | ---------
    ()    | a = 0.
    ()    | b = 1.
    ()    | scale ~ exponential(b)
    ()    | observation ~ normal(a, scale)

    Rewrite ``scale`` with ``observation`` given, then display both returned values:

    >>> new_scale, new_observation = functional.unconstrain(scale, observation)
    >>> ir.print_upstream(new_observation, {"scale": new_scale, "observation": new_observation})
    shape | statement
    ----- | ---------
    ()    | a = 0.
    ()    | b = 1.
    ()    | c ~ bijected(exponential, bijector(log, exp, composite(2, [log, -1, mul], [[0], [], [3, 2]]), sign=1))(b)
    ()    | scale = exp(c)
    ()    | observation ~ normal(a, scale)

    The rewritten ``scale`` and ``observation`` occupy the same constrained spaces as before. The
    random ``scale`` node has become a deterministic reconstruction from the unconstrained random
    node ``c``, while the given ``observation`` retains its original Normal Op.
    """
    require_rv_leaves(vars, "unconstrain vars")
    given_rvs = require_rv_leaves(given, "unconstrain given")
    observed = set(given_rvs)

    def rewrite(var: ir.RV, parents: tuple[ir.RV, ...]) -> tuple[ir.RV, None]:
        rebuilt = ir.RV(var.op, *parents, interface=var.interface)
        if not var.op.random or var in observed:
            return rebuilt, None

        transform = transform_for(rebuilt.op, rebuilt.parents)
        if transform is not None:
            if not isinstance(transform, Transform):
                raise TypeError(
                    "transform_for must return a Transform or None; "
                    f"got {type(transform).__name__}"
                )
            _, rebuilt = apply_transform(transform, rebuilt)
        return rebuilt, None

    (new_vars, new_given), _ = graph_scan((vars, given), rewrite)
    return new_vars, new_given
