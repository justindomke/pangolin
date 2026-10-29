"""Build quantile graphs from scalar-distribution-producing functions."""

from collections.abc import Callable
from typing import Concatenate

from pangolin import ir

from .tracing import trace_call

__all__ = ["quantile"]


def quantile[**P](fun: Callable[P, ir.RV]) -> Callable[Concatenate[ir.RV, P], ir.RV]:
    """A combinator that returns a function building a scalar distribution's quantile.

    Calling ``quantile(fun)(probability, *args, **kwargs)`` calls ``fun(*args, **kwargs)``
    once, retains its contributing deterministic graph, and creates an `ir.Quantile` RV
    using the output's Op and parents. It does not introduce a Composite or retain the
    random output as a parent. The result is a deterministic scalar RV.

    The function must return exactly one newly created scalar random RV. Every other
    contributing newly created RV must be deterministic. Existing random inputs and captures
    are allowed. The quantile evaluates the output distribution using its parent values;
    it does not integrate over their upstream distributions. Newly created nodes that do not
    contribute to the output are ignored. Discrete and continuous distributions are allowed
    by the IR, but numerical support depends on the backend.

    Parameters
    ----------
    fun
        Function creating one scalar random RV. Its arguments may have any Python or
        PyTree structure accepted by the function itself.

    Returns
    -------
    quantile_fun
        Function taking a positional-only scalar probability RV followed by ``fun``'s
        arguments. For a probability strictly between zero and one, the result is the
        smallest value at which the CDF reaches that probability. At zero and one, it uses
        the corresponding one-sided limits, which may be infinite. Probability values and
        numerical support are checked by the backend, not when constructing the graph.

    Raises
    ------
    TypeError
        If the probability or the function output is not a single RV.
    ValueError
        If the probability or distribution output is not scalar, the output is pre-existing
        or deterministic, or a contributing intermediate RV is random.

    Notes
    -----
    Arguments are supplied to ``fun`` unchanged. This combinator performs no coercion or
    broadcasting. Any coercion performed by ``fun`` is preserved, but its output must still
    be scalar. Use `vmap` around the quantile function for independent evaluations; there is
    no joint quantile for an array-valued distribution. The result carries the interface of
    ``fun``'s output, regardless of the probability RV's interface.

    Examples
    --------
    Deterministic parameter calculations remain ordinary graph nodes:

    >>> def normal_with_log_scale(mean, log_scale):
    ...     scale = ir.RV(ir.Exp(), log_scale)
    ...     return ir.RV(ir.Normal(), mean, scale)
    >>> probability = ir.RV(ir.Constant(0.5))
    >>> zero = ir.RV(ir.Constant(0.0))
    >>> result = quantile(normal_with_log_scale)(probability, zero, zero)
    >>> result.op
    Quantile(Normal())
    >>> result.parents[0] is probability
    True
    >>> result.parents[2].op
    Exp()
    >>> result.shape, result.op.random
    ((), False)

    Vectorize the quantile function, rather than the distribution inside it:

    >>> from pangolin.functional import vmap
    >>> probabilities = ir.RV(ir.Constant([0.1, 0.5, 0.9]))
    >>> result = vmap(quantile(normal_with_log_scale), (0, None, None))(probabilities, zero, zero)
    >>> result.op
    VMap(Quantile(Normal()), [0, None, None], 3)
    >>> result.shape
    (3,)
    """

    def call(probability: ir.RV, /, *args: P.args, **kwargs: P.kwargs) -> ir.RV:
        if not isinstance(probability, ir.RV):
            raise TypeError("quantile probability must be an RV")
        output, generated, _ = trace_call(fun, *args, **kwargs)
        if not isinstance(output, ir.RV):
            raise TypeError("quantile function must return exactly one newly created RV")
        if not generated or generated[-1] is not output:
            raise ValueError("quantile function output must be created during the call")
        if not output.op.random:
            raise ValueError("quantile function output must be random")
        if any(node.op.random for node in generated[:-1]):
            raise ValueError(
                "quantile function must not create contributing random intermediate RVs"
            )
        return ir.RV(
            ir.Quantile(output.op), probability, *output.parents, interface=output.interface
        )

    original_name = getattr(fun, "__name__", None)
    if not original_name or original_name == "<lambda>":
        original_name = "function"
    call.__name__ = f"quantile_{original_name}"
    call.__qualname__ = call.__name__
    return call
