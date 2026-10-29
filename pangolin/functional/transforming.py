"""Provide callable `Transform` combinators for transformed distributions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import FunctionType
from typing import Final, Literal

from pangolin import ir

from .compositing import composite_from_trace
from .tracing import trace_call

__all__ = ["Transform", "compose_transforms"]


class Transform:
    """Package an invertible pair of RV functions with distribution combinators.

    ``forward(x)`` computes ``y``, ``inverse(y)`` reconstructs ``x``, and
    ``log_det_jac(x, y)`` computes the scalar forward log absolute Jacobian determinant. Thus, a
    Transform contains two inverse value functions plus the density correction that relates them.
    `Transform.__call__` and `Transform.apply_and_invert` are combinators that accept a function
    constructing ``x`` and return functions based on one random RV with an `ir.Bijected` Op.

    The three functions are traced when the transformed function is called. Their deterministic
    graphs are compiled into an `ir.Bijector`; RVs they capture become bijector parameters. The
    forward and inverse functions must capture the same RV identities, while ``log_det_jac`` may
    use any subset of them.

    Optional ``sign()`` declares a continuous, strictly monotone scalar bijection, returning
    ``+1`` for increasing or ``-1`` for decreasing. It is traced at application time and may
    capture any subset of the forward parameters; its Op accepts all parameters in the same
    order. It may not depend on the value being transformed or introduce additional captured RVs.

    Parameters
    ----------
    forward
        Unary function mapping a base RV to its transformed value.
    inverse
        Unary function mapping the transformed value back to the base value.
    log_det_jac
        Binary function computing ``log(abs(det(J_forward(x))))`` from ``x`` and ``y``.
    sign
        Optional zero-argument function returning a deterministic scalar RV, ``+1`` for an
        increasing scalar bijection or ``-1`` for a decreasing one.

    Examples
    --------
    Construct an exponential Transform and apply it to a normal distribution function:

    >>> exp_transform = Transform(
    ...     lambda x: ir.RV(ir.Exp(), x),
    ...     lambda y: ir.RV(ir.Log(), y),
    ...     lambda x, y: ir.RV(ir.Log(), y),
    ... )
    >>> def normal(mean, scale):
    ...     return ir.RV(ir.Normal(), mean, scale)
    >>> mean = ir.RV(ir.Constant(0))
    >>> scale = ir.RV(ir.Constant(1))
    >>> result = exp_transform(normal)(mean, scale)
    >>> result.op
    Bijected(Normal(), Bijector(Exp(), Log(), Composite(2, [Log()], [[1]])))
    >>> result.parents == (mean, scale)
    True

    The forward and inverse functions determine the transformed and reconstructed interfaces:

    >>> forward_interface = ir.RVInterface()
    >>> inverse_interface = ir.RVInterface()
    >>> special_exp = Transform(
    ...     lambda x: ir.RV(ir.Exp(), x, interface=forward_interface),
    ...     lambda y: ir.RV(ir.Log(), y, interface=inverse_interface),
    ...     lambda x, y: ir.RV(ir.Log(), y),
    ... )
    >>> def special_normal(mean, scale):
    ...     return ir.RV(ir.Normal(), mean, scale)
    >>> special_mean = ir.RV(ir.Constant(0))
    >>> special_scale = ir.RV(ir.Constant(1))
    >>> transformed, reconstructed = special_exp.apply_and_invert(special_normal)(
    ...     special_mean, special_scale
    ... )
    >>> transformed.interface is forward_interface
    True
    >>> reconstructed.interface is inverse_interface
    True

    The sign is supplied by a graph-building callback too:

    >>> increasing_exp = Transform(
    ...     exp_transform.forward, exp_transform.inverse, exp_transform.log_det_jac,
    ...     sign=lambda: ir.RV(ir.Constant(1)),
    ... )
    >>> increasing_exp(normal)(mean, scale).op.bijector.sign
    Constant(1)

    """

    def __init__(
        self,
        forward: Callable[[ir.RV], ir.RV],
        inverse: Callable[[ir.RV], ir.RV],
        log_det_jac: Callable[[ir.RV, ir.RV], ir.RV],
        *,
        sign: Callable[[], ir.RV] | None = None,
    ) -> None:
        self.forward: Final = forward
        self.inverse: Final = inverse
        self.log_det_jac: Final = log_det_jac
        self.sign: Final = sign

    def __call__[**P](self, fun: Callable[P, ir.RV], /) -> Callable[P, ir.RV]:
        """Apply this Transform combinator to a continuous distribution function.

        The returned function preserves ``fun``'s complete call signature and passes all arguments
        through unchanged. The result carries the interface selected by ``Transform.forward``.

        Parameters
        ----------
        fun
            Function that creates one continuous random RV accepted by ``forward``.

        Returns
        -------
        transformed_fun
            Function with the same signature as ``fun`` that creates the transformed RV.
        """

        def transformed_fun(*args: P.args, **kwargs: P.kwargs) -> ir.RV:
            x = fun(*args, **kwargs)
            if not isinstance(x, ir.RV):
                raise TypeError(
                    "a transformed function must return exactly one RV; " f"got {type(x).__name__}"
                )
            transformed, _ = apply_transform(self, x)
            return transformed

        set_transformed_name(transformed_fun, fun)
        return transformed_fun

    def apply_and_invert[**P](self, fun: Callable[P, ir.RV], /) -> Callable[P, tuple[ir.RV, ir.RV]]:
        """Apply this Transform combinator and also reconstruct the base value.

        Like `Transform.__call__`, this method is a combinator: it accepts a
        distribution-producing function and returns another function.

        The reconstructed base is deterministic and uses the compiled inverse operation; it is
        not a second draw from ``fun``.

        Parameters
        ----------
        fun
            Function that creates one continuous random RV accepted by ``forward``.

        Returns
        -------
        transformed_and_base_fun
            Function with the same signature as ``fun`` that returns ``(y, x)``.
        """

        def transformed_and_base_fun(*args: P.args, **kwargs: P.kwargs) -> tuple[ir.RV, ir.RV]:
            x = fun(*args, **kwargs)
            if not isinstance(x, ir.RV):
                raise TypeError(
                    "a transformed function must return exactly one RV; " f"got {type(x).__name__}"
                )
            return apply_transform(self, x)

        set_transformed_name(transformed_and_base_fun, fun, suffix="_and_base")
        return transformed_and_base_fun

    @property
    def reverse(self) -> Transform:
        """Reverse the value functions, negate the LDJ, and preserve the sign."""

        def reversed_log_det(y: ir.RV, x: ir.RV) -> ir.RV:
            log_det = self.log_det_jac(x, y)
            minus_one = ir.RV(ir.Constant(-1), interface=log_det.interface)
            return ir.RV(ir.Mul(), minus_one, log_det, interface=log_det.interface)

        return Transform(
            self.inverse,
            self.forward,
            reversed_log_det,
            sign=self.sign,
        )

    def __repr__(self) -> str:
        """Return a constructor-shaped representation of the Transform."""
        args = [repr(self.forward), repr(self.inverse), repr(self.log_det_jac)]
        if self.sign is not None:
            args.append(f"sign={self.sign!r}")
        return f"Transform({', '.join(args)})"


def apply_transform(
    transform: Transform,
    x: ir.RV,
    /,
) -> tuple[ir.RV, ir.RV]:
    """Compile and apply ``transform`` to one base random RV."""
    traced_y, forward_generated, forward_boundary = trace_call(transform.forward, x)
    if not isinstance(traced_y, ir.RV):
        raise TypeError("forward must return exactly one RV")
    if not forward_boundary or forward_boundary[0] is not x:
        raise ValueError("forward must depend on its input RV")
    forward_captured = forward_boundary[1:]
    forward_op = composite_from_trace(
        traced_y,
        forward_generated,
        [x, *forward_captured],
        allow_existing_output=True,
    )
    y = ir.RV(forward_op, x, *forward_captured, interface=traced_y.interface)

    traced_x, inverse_generated, inverse_boundary = trace_call(transform.inverse, y)
    if not isinstance(traced_x, ir.RV):
        raise TypeError("inverse must return exactly one RV")
    if not inverse_boundary or inverse_boundary[0] is not y:
        raise ValueError("inverse must depend on its input RV")
    inverse_captured = inverse_boundary[1:]
    if set(inverse_captured) != set(forward_captured):
        raise ValueError("forward and inverse must capture exactly the same RV objects")
    inverse_op = composite_from_trace(
        traced_x,
        inverse_generated,
        [y, *forward_captured],
        allow_existing_output=True,
    )

    traced_log_det, log_det_generated, log_det_boundary = trace_call(transform.log_det_jac, x, y)
    if not isinstance(traced_log_det, ir.RV):
        raise TypeError("log_det_jac must return exactly one RV")
    log_det_captured = [
        parent for parent in log_det_boundary if parent is not x and parent is not y
    ]
    if not set(log_det_captured) <= set(forward_captured):
        raise ValueError("log_det_jac cannot capture an RV that forward does not capture")
    log_det_op = composite_from_trace(
        traced_log_det,
        log_det_generated,
        [x, y, *forward_captured],
        allow_existing_output=True,
    )

    sign_op = None
    if transform.sign is not None:
        result, generated, captured = trace_call(transform.sign)
        if not isinstance(result, ir.RV):
            raise TypeError("sign must return exactly one RV")
        if not set(captured) <= set(forward_captured):
            raise ValueError("sign cannot capture an RV that forward does not capture")
        sign_op = composite_from_trace(
            result,
            generated,
            forward_captured,
            allow_existing_output=True,
        )

    bijector = ir.Bijector(
        forward_op,
        inverse_op,
        log_det_op,
        n_biject_params=len(forward_captured),
        sign=sign_op,
    )
    transformed_op = ir.Bijected(x.op, bijector)
    transformed_rv = ir.RV(
        transformed_op,
        *forward_captured,
        *x.parents,
        interface=traced_y.interface,
    )
    reconstructed_rv = ir.RV(
        inverse_op,
        transformed_rv,
        *forward_captured,
        interface=traced_x.interface,
    )

    return transformed_rv, reconstructed_rv


def set_transformed_name(
    transformed_fun: FunctionType,
    original_fun: Callable[..., object],
    *,
    suffix: str = "",
) -> None:
    """Give a transformed function a stable name derived from its input function."""
    original_name = getattr(original_fun, "__name__", None)
    if not original_name or original_name == "<lambda>":
        original_name = "function"
    transformed_fun.__name__ = f"transformed_{original_name}{suffix}"
    transformed_fun.__qualname__ = transformed_fun.__name__


def compose_transforms(
    transforms: Sequence[Transform],
    log_det_direction: Literal["forward", "inverse"] = "forward",
) -> Transform:
    """Compose Transform combinators in left-to-right application order.

    Monotonic signs are multiplied if every Transform supplies one; otherwise the result's
    sign is absent. The empty composition has sign ``+1``.

    Parameters
    ----------
    transforms
        Transforms to apply in sequence; the first item is applied first.
    log_det_direction
        Whether to reconstruct intermediate values from ``x`` or from ``y`` while computing the
        combined log determinant.

    Returns
    -------
    transform
        One Transform equivalent to applying every input Transform in order.

    Raises
    ------
    TypeError
        If an item is not a `Transform`.
    ValueError
        If ``log_det_direction`` is not ``"forward"`` or ``"inverse"``.
    """
    transforms = tuple(transforms)
    for transform in transforms:
        if not isinstance(transform, Transform):
            raise TypeError("every item in transforms must be a Transform")
    if log_det_direction not in ("forward", "inverse"):
        raise ValueError("log_det_direction must be 'forward' or 'inverse'")

    def forward(x: ir.RV) -> ir.RV:
        for transform in transforms:
            x = transform.forward(x)
        return x

    def inverse(y: ir.RV) -> ir.RV:
        for transform in reversed(transforms):
            y = transform.inverse(y)
        return y

    def add_log_det(total: ir.RV, term: ir.RV) -> ir.RV:
        return ir.RV(ir.Add(), total, term, interface=total.interface)

    def forward_log_det(x: ir.RV, y: ir.RV) -> ir.RV:
        total = ir.RV(ir.Constant(0.0), interface=x.interface)
        for transform in transforms:
            next_x = transform.forward(x)
            total = add_log_det(total, transform.log_det_jac(x, next_x))
            x = next_x
        return total

    def inverse_log_det(x: ir.RV, y: ir.RV) -> ir.RV:
        total = ir.RV(ir.Constant(0.0), interface=x.interface)
        for transform in reversed(transforms):
            previous_y = transform.inverse(y)
            total = add_log_det(total, transform.log_det_jac(previous_y, y))
            y = previous_y
        return total

    log_det_jac = forward_log_det if log_det_direction == "forward" else inverse_log_det

    signs = [transform.sign for transform in transforms if transform.sign is not None]

    def composed_sign() -> ir.RV:
        total = ir.RV(ir.Constant(1))
        for sign in signs:
            term = sign()
            total = ir.RV(ir.Mul(), total, term, interface=term.interface)
        return total

    return Transform(
        forward,
        inverse,
        log_det_jac,
        sign=composed_sign if len(signs) == len(transforms) else None,
    )
