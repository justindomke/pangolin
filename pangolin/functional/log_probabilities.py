"""Build log-probability graphs from distribution-producing functions."""

from collections.abc import Callable
from typing import Concatenate

from pangolin import ir

from .tracing import trace_call

__all__ = ["log_prob"]


def log_prob[**P](fun: Callable[P, ir.RV]) -> Callable[Concatenate[ir.RV, P], ir.RV]:
    """A combinator that returns a function building a distribution's log probability.

    Calling ``log_prob(fun)(value, *args, **kwargs)`` calls ``fun(*args, **kwargs)`` once,
    retains its contributing deterministic graph, and creates an `ir.LogProb` RV using the
    output's Op and parents. It does not introduce a Composite or retain the random output
    as a parent. The result is a deterministic scalar RV carrying that output's interface.

    The function must return exactly one newly created random RV. Every other contributing
    newly created RV must be deterministic. Existing random inputs and captures are allowed.
    The resulting LogProb node evaluates the output distribution at the evaluation value and
    its parent values. It does not compute a marginal density by integrating over random parents.
    Newly created nodes that do not contribute to the output are ignored. Discrete and continuous
    distributions are both allowed.

    Parameters
    ----------
    fun
        Function creating one random RV. Arguments may have any Python or PyTree structure
        accepted by the function itself and are supplied unchanged, without coercion.

    Returns
    -------
    log_prob_fun
        Function taking a positional-only evaluation RV followed by ``fun``'s arguments.
        The evaluation RV must have exactly the distribution output's shape; no coercion or
        broadcasting is performed. For array-valued distributions the result is the joint
        log density or mass of the whole array.

    Raises
    ------
    TypeError
        If the evaluation value or the function output is not a single RV.
    ValueError
        If the output is pre-existing or deterministic, a contributing intermediate RV is
        random, or the evaluation value has the wrong shape.

    Examples
    --------
    Deterministic parameter calculations remain ordinary graph nodes:

    >>> def normal_with_log_scale(mean, log_scale):
    ...     scale = ir.RV(ir.Exp(), log_scale)
    ...     return ir.RV(ir.Normal(), mean, scale)
    >>> normal_log_prob = log_prob(normal_with_log_scale)
    >>> mean = ir.RV(ir.Constant(0.0))
    >>> log_scale = ir.RV(ir.Constant(0.0))
    >>> value = ir.RV(ir.Constant(2.0))
    >>> result = normal_log_prob(value, mean, log_scale)
    >>> result.op
    LogProb(Normal())
    >>> result.parents[2].op
    Exp()
    >>> result.parents[0] is value
    True
    >>> result.shape, result.op.random
    ((), False)
    """

    def call(value: ir.RV, /, *args: P.args, **kwargs: P.kwargs) -> ir.RV:
        if not isinstance(value, ir.RV):
            raise TypeError("log_prob evaluation value must be an RV")
        output, generated, _ = trace_call(fun, *args, **kwargs)
        if not isinstance(output, ir.RV):
            raise TypeError("log_prob function must return exactly one newly created RV")
        if not generated or generated[-1] is not output:
            raise ValueError("log_prob function output must be created during the call")
        if not output.op.random:
            raise ValueError("log_prob function output must be random")
        if any(node.op.random for node in generated[:-1]):
            raise ValueError(
                "log_prob function must not create contributing random intermediate RVs"
            )
        return ir.RV(ir.LogProb(output.op), value, *output.parents, interface=output.interface)

    original_name = getattr(fun, "__name__", None)
    if not original_name or original_name == "<lambda>":
        original_name = "function"
    call.__name__ = f"log_prob_{original_name}"
    call.__qualname__ = call.__name__
    return call
