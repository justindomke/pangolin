"""Construct truncated scalar distributions and independent arrays of them."""

from collections.abc import Callable
from functools import wraps

from pangolin import ir

from .tracing import trace_call

__all__ = ["truncated"]


def truncate_op(op: ir.Op, *, lower: bool = True, upper: bool = True) -> ir.Op:
    """Add leading bound parents, lifting truncation through VMap and Composite.

    The returned Op receives the lower bound if ``lower``, then the upper bound if ``upper``,
    followed by the original parents. Both flags default to ``True``. Recursion changes only
    VMap base Ops and Composite final Ops; every other Op is wrapped whole in `ir.Truncated`.
    No RVs are constructed. Scalar bounds and scalar leaf outputs are checked later by
    ``get_shape``, when the parent shapes are available.

    Examples
    --------
    A scalar distribution gains two leading parents, before its original parameters:

    >>> op = truncate_op(ir.Normal())
    >>> op
    Truncated(Normal())
    >>> op.get_shape((), (), (), ())  # lo, hi, mean, scale
    ()

    A one-sided truncation adds only its supplied bound:

    >>> op = truncate_op(ir.Normal(), lower=False)
    >>> op
    Truncated(Normal(), lower=False)
    >>> op.get_shape((), (), ())  # hi, mean, scale
    ()

    VMap preserves the original axes and size, with both bounds unmapped:

    >>> op = truncate_op(ir.VMap(ir.Normal(), (0, None), 3))
    >>> op
    VMap(Truncated(Normal()), [None, None, 0, None], 3)
    >>> op.get_shape((), (), (3,), ())
    (3,)

    With one bound, only one unmapped axis is added:

    >>> truncate_op(ir.VMap(ir.Normal(), (0, None), 3), upper=False)
    VMap(Truncated(Normal(), upper=False), [None, 0, None], 3)

    Nested VMaps share the same bounds at every level. Non-leading mapped axes are retained:

    >>> base = ir.VMap(ir.VMap(ir.Normal(), (0, None), 2), (1, None), 3)
    >>> op = truncate_op(base)
    >>> op
    VMap(VMap(Truncated(Normal()), [None, None, 0, None], 2), [None, None, 1, None], 3)
    >>> op.get_shape((), (), (2, 3), ())
    (3, 2)

    A Composite receives the bounds as inputs 0 and 1. All old input and intermediate
    indices shift by two; only the final Op receives the new bounds. Here the old inputs
    are mean and log scale, and Exp computes the scale:

    >>> base = ir.Composite(2, (ir.Exp(), ir.Normal()), ((1,), (0, 2)))
    >>> op = truncate_op(base)
    >>> op
    Composite(4, [Exp(), Truncated(Normal())], [[3], [0, 1, 2, 4]])
    >>> op.ops[0] is base.ops[0]
    True
    >>> op.get_shape((), (), (), ())
    ()

    Even a Composite with no original inputs gains two bound inputs. Its constants remain
    internal, at indices 2 and 3:

    >>> base = ir.Composite(0, (ir.Constant(1.1), ir.Constant(2.2), ir.Normal()), ((), (), (0, 1)))
    >>> truncate_op(base)
    Composite(2, [Constant(1.1), Constant(2.2), Truncated(Normal())], [[], [], [0, 1, 2, 3]])

    VMap and Composite can alternate. Each Composite routes the bounds to its final Op,
    and each VMap passes them through unchanged:

    >>> scalar = ir.Composite(2, (ir.Normal(),), ((0, 1),))
    >>> mapped = ir.VMap(scalar, (0, None), 3)
    >>> base = ir.Composite(2, (mapped,), ((0, 1),))
    >>> op = truncate_op(base)
    >>> op
    Composite(4, [VMap(Composite(4, [Truncated(Normal())], [[0, 1, 2, 3]]), [None, None, 0, None], 3)], [[0, 1, 2, 3]])
    >>> op.get_shape((), (), (3,), ())
    (3,)

    Recursion stops at a Bijected Op: bounds apply to the transformed value, not to its
    underlying distribution. Its entire structure is retained:

    >>> bijector = ir.Bijector(ir.Exp(), ir.Log(), ir.Composite(2, (ir.Identity(),), ((0,),)))
    >>> base = ir.Bijected(ir.Normal(), bijector)
    >>> op = truncate_op(base)
    >>> op == ir.Truncated(base)
    True
    >>> op.base_op is base
    True
    >>> op.get_shape((), (), (), ())
    ()

    An array-valued leaf is rejected when shapes are supplied, rather than being decomposed
    into scalar distributions:

    >>> op = truncate_op(ir.MultiNormal())
    >>> op.get_shape((), (), (2,), (2, 2))
    Traceback (most recent call last):
        ...
    ValueError: base_op must have scalar output
    """
    if not isinstance(lower, bool) or not isinstance(upper, bool):
        raise TypeError("lower and upper must be booleans")
    num_bounds = int(lower) + int(upper)
    if isinstance(op, ir.VMap):
        return ir.VMap(
            truncate_op(op.base_op, lower=lower, upper=upper),
            (None,) * num_bounds + op.in_axes,
            op.axis_size,
        )
    if isinstance(op, ir.Composite):
        ops = (*op.ops[:-1], truncate_op(op.ops[-1], lower=lower, upper=upper))
        # Leading bounds shift both original inputs and intermediate results.
        par_nums = tuple(tuple(index + num_bounds for index in parents) for parents in op.par_nums)
        par_nums = (*par_nums[:-1], (*range(num_bounds), *par_nums[-1]))
        return ir.Composite(op.num_inputs + num_bounds, ops, par_nums)
    return ir.Truncated(op, lower=lower, upper=upper)


def truncated[**P](
    fun: Callable[P, ir.RV], lo: ir.RV | None = None, hi: ir.RV | None = None
) -> Callable[P, ir.RV]:
    """A combinator that restricts scalar distributions to a shared interval.

    Calling ``truncated(fun, lo, hi)(*args, **kwargs)`` calls ``fun`` once and
    constructs a new random RV representing its output distribution restricted to
    the supplied bounds, with the appropriate normalization. Each bound must be a scalar RV
    or ``None`` to omit that bound. Supplied bounds may be random or deterministic; their
    realized values must be finite and strictly inside the base distribution's support, with
    ``lo < hi`` when both are supplied. Backends enforce these value-dependent requirements
    during evaluation, not when the graph is constructed.

    The function must return exactly one newly created continuous random RV. As with `composite`,
    no other random RVs created by the function may contribute to the output. Multivariate
    distributions are unsupported, except when formed by vectorizing scalar distributions.

    The result preserves the output shape, applying the same bounds to every vectorized draw.
    With both bounds omitted, the original output RV is returned, subject to the same requirements.

    Parameters
    ----------
    fun
        Function creating one continuous random RV, scalar or an independent array of scalar
        draws expressed through VMap and Composite. Its arguments may have any Python or
        PyTree structure accepted by the function itself.
    lo
        Scalar RV supplying the lower bound, or ``None`` (the default) to omit it.
    hi
        Scalar RV supplying the upper bound, or ``None`` (the default) to omit it.

    Returns
    -------
    truncated_fun
        Function accepting the same arguments as ``fun`` and constructing its truncated output.

    Raises
    ------
    TypeError
        If either bound is neither an RV nor ``None``, or ``fun`` does not return exactly one RV.
    ValueError
        If a bound is not scalar, the output is pre-existing, deterministic, or discrete,
        a contributing intermediate RV is random, or a remaining base distribution is not scalar.

    Notes
    -----
    Arguments are supplied to ``fun`` unchanged. This combinator performs no coercion or
    broadcasting. Any coercion or broadcasting performed by ``fun`` is preserved. Bounds are
    checked when the combinator is constructed and are never broadcast. The result carries the
    interface of ``fun``'s output, regardless of the interfaces of its bounds or parameters.

    Examples
    --------
    Truncate a Normal using scalar RV bounds. The result's parents are the bounds followed
    by the original distribution parameters:

    >>> def normal(mean, scale):
    ...     return ir.RV(ir.Normal(), mean, scale)
    >>> zero = ir.RV(ir.Constant(0.0))
    >>> one = ir.RV(ir.Constant(1.0))
    >>> mean = ir.RV(ir.Constant(1.1))
    >>> scale = ir.RV(ir.Constant(2.2))
    >>> positive_normal = truncated(normal, lo=zero)
    >>> result = positive_normal(mean, scale)
    >>> result.op
    Truncated(Normal(), upper=False)
    >>> result.parents == (zero, mean, scale)
    True
    >>> result.shape, result.op.random
    ((), True)

    Vectorization remains outside the scalar truncation. The bounds are shared, not mapped:

    >>> from pangolin.functional import composite, vmap
    >>> means = ir.RV(ir.Constant([1.1, 1.2, 1.3]))
    >>> vector_normal = vmap(normal, (0, None))
    >>> result = truncated(vector_normal, lo=zero)(means, scale)
    >>> result.op
    VMap(Truncated(Normal(), upper=False), [None, 0, None], 3)
    >>> result.shape
    (3,)

    Deterministic parameter calculations inside a Composite are retained:

    >>> def normal_with_log_scale(mean, log_scale):
    ...     scale = ir.RV(ir.Exp(), log_scale)
    ...     return normal(mean, scale)
    >>> log_scale = ir.RV(ir.Constant(3.3))
    >>> result = truncated(composite(normal_with_log_scale), zero, one)(mean, log_scale)
    >>> result.op
    Composite(4, [Exp(), Truncated(Normal())], [[3], [0, 1, 2, 4]])

    Scalar bijections are truncated in their output coordinates, not underneath the bijection:

    >>> from pangolin.functional import tforms
    >>> result = truncated(tforms.exp(normal), lo=one)(mean, scale)
    >>> result.op
    Truncated(Bijected(Normal(), Bijector(Exp(), Log(), Composite(2, [Log()], [[1]]), sign=Constant(1))), upper=False)

    Even a length-one array is not a scalar bound:

    >>> truncated(normal, ir.RV(ir.Constant([0.0])))
    Traceback (most recent call last):
        ...
    ValueError: truncation bounds must be scalar
    """
    bounds = tuple(bound for bound in (lo, hi) if bound is not None)
    if any(not isinstance(bound, ir.RV) for bound in bounds):
        raise TypeError("truncation bounds must be RVs or None")
    if any(bound.shape != () for bound in bounds):
        raise ValueError("truncation bounds must be scalar")

    @wraps(fun)
    def call(*args: P.args, **kwargs: P.kwargs) -> ir.RV:
        output, generated, _ = trace_call(fun, *args, **kwargs)
        if not isinstance(output, ir.RV):
            raise TypeError("truncated function must return exactly one newly created RV")
        if not generated or generated[-1] is not output:
            raise ValueError("truncated function output must be created during the call")
        if not output.op.random:
            raise ValueError("truncated function output must be random")
        if output.op.discrete:
            raise ValueError("truncated function output must be continuous")
        if any(node.op.random for node in generated[:-1]):
            raise ValueError(
                "truncated function must not create contributing random intermediate RVs"
            )
        op = truncate_op(output.op, lower=lo is not None, upper=hi is not None)
        if not bounds:
            op.get_shape(*(parent.shape for parent in output.parents))
            return output
        return ir.RV(op, *bounds, *output.parents, interface=output.interface)

    original_name = getattr(fun, "__name__", None)
    if not original_name or original_name == "<lambda>":
        original_name = "function"
    call.__name__ = f"truncated_{original_name}"
    call.__qualname__ = call.__name__
    return call
