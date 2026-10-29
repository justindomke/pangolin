"""Provide the `composite` functional combinator."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

from pangolin import ir

from .tracing import trace_call

__all__ = ["composite"]


def composite_from_trace(
    output: ir.RV,
    generated: list[ir.RV],
    parents: list[ir.RV],
    *,
    allow_existing_output: bool = False,
    simplify: bool = True,
) -> ir.Op:
    """Construct an Op from a traced graph and an explicit parent order.

    ``generated`` must contain every newly created RV contributing to ``output``, once each in
    creation order. When ``output`` is new, it must be the final item in ``generated``. ``parents``
    supplies every pre-existing parent of the generated graph in the exact order the resulting
    Composite should use. It may include unused RVs.

    When ``allow_existing_output`` is true, ``output`` may instead be an existing RV in
    ``parents``. The resulting Composite ends in an `ir.Identity` selecting that parent.

    Parameters
    ----------
    output
        Final RV represented by the Composite.
    generated
        Contributing generated RVs in creation order.
    parents
        Ordered external inputs to the Composite.
    allow_existing_output
        Whether an existing output in ``parents`` may be represented with an Identity operation.
    simplify
        If true, replace a Composite containing a single Op with that Op when they are
        equivalent (the default). Otherwise, retain the Composite.

    Returns
    -------
    composite_op
        Operation representing ``output`` in terms of ``parents``, a Composite unless simplified.

    Raises
    ------
    ValueError
        If the trace is inconsistent with ``output`` or omits a required external parent.
    """
    if not generated or generated[-1] is not output:
        if not allow_existing_output:
            raise ValueError("output must be the final generated RV")
        if generated:
            raise ValueError("an existing output cannot have contributing generated RVs")
        if output not in parents:
            raise ValueError("an existing output must be included in parents")
        parent_number = parents.index(output)
        op = ir.Composite(len(parents), (ir.Identity(),), ((parent_number,),))
    else:
        positions = {parent: position for position, parent in enumerate(parents)}
        parent_numbers = []
        for generated_position, rv in enumerate(generated):
            try:
                parent_numbers.append(tuple(positions[parent] for parent in rv.parents))
            except KeyError as error:
                raise ValueError(
                    "parents must include every external parent of the generated graph"
                ) from error
            positions[rv] = len(parents) + generated_position

        preceding_ops = tuple(rv.op for rv in generated[:-1])
        ops: tuple[*tuple[ir.Op, ...], ir.Op] = (*preceding_ops, output.op)
        op = ir.Composite(len(parents), ops, tuple(parent_numbers))

    if simplify and len(op.ops) == 1 and op.par_nums[0] == tuple(range(op.num_inputs)):
        return op.ops[0]
    return op


def composite[**P](fun: Callable[P, ir.RV], *, simplify: bool = True) -> Callable[P, ir.RV]:
    """A combinator that returns a function compressing ``fun``'s graph into one RV.

    Each call invokes ``fun`` normally, traces the newly created IR graph contributing to its
    output, and compresses that graph into one `ir.Composite` Op. Pre-existing RVs become parents
    of the resulting RV. New RVs that do not contribute to the output are omitted.

    Every contributing RV other than the output must have a deterministic Op. The output may be
    deterministic or random. Arguments need not be RVs: they are passed to ``fun`` unchanged and
    may control ordinary Python execution or be used by ``fun`` to construct RVs. ``composite``
    performs no argument coercion. The result is a plain `ir.RV` carrying the traced output's
    interface.

    Parameters
    ----------
    fun
        Function that creates and returns exactly one new `ir.RV`. Its arguments may have any
        Python or PyTree structure accepted by the function itself.
    simplify
        If true, replace a Composite containing a single Op with that Op when they are
        equivalent (the default). Otherwise, retain the Composite.

    Returns
    -------
    composite_fun
        Function accepting the same runtime arguments as ``fun`` and returning an RV containing
        the compressed Op, which is an `ir.Composite` unless simplified.

    Raises
    ------
    TypeError
        If ``fun`` does not return exactly one `ir.RV`.
    ValueError
        If ``fun`` returns an RV that existed before the call, or a contributing intermediate RV
        has a random Op.

    Examples
    --------
    The combinator first returns a function. Calling that function constructs a Composite RV whose
    parents are the existing arguments:

    >>> def normal_with_sum(a, b):
    ...     total = ir.RV(ir.Add(), a, b)
    ...     return ir.RV(ir.Normal(), total, b)
    >>> composite_normal_with_sum = composite(normal_with_sum)
    >>> composite_normal_with_sum  # doctest: +ELLIPSIS
    <function composite_normal_with_sum at 0x...>
    >>> x = ir.RV(ir.Constant(1))
    >>> y = ir.RV(ir.Constant(2))
    >>> result = composite_normal_with_sum(x, y)
    >>> result.op
    Composite(2, [Add(), Normal()], [[0, 1], [2, 1]])
    >>> result.parents == (x, y)
    True

    RVs captured by ``fun`` follow contributing argument RVs in the parent list:

    >>> offset = ir.RV(ir.Constant(3))
    >>> add_offset = composite(lambda value: ir.RV(ir.Add(), offset, value))
    >>> result = add_offset(x)
    >>> result.parents == (x, offset)
    True

    Non-RV arguments are passed through and do not themselves become parents:

    >>> def choose(value, take_log):
    ...     op = ir.Log() if take_log else ir.Exp()
    ...     return ir.RV(op, value)
    >>> choose_composite = composite(choose, simplify=False)
    >>> result = choose_composite(x, True)
    >>> result.op
    Composite(1, [Log()], [[0]])
    >>> result.parents == (x,)
    True

    By default, the single-Op Composite is simplified:

    >>> composite(choose)(x, True).op
    Log()

    The Composite RV carries the interface selected by ``fun``:

    >>> interface = ir.RVInterface()
    >>> value = ir.RV(ir.Constant(1))
    >>> special_exp = composite(
    ...     lambda value: ir.RV(ir.Exp(), value, interface=interface)
    ... )
    >>> result = special_exp(value)
    >>> result.interface is interface
    True
    """

    @wraps(fun)
    def call(*args: P.args, **kwargs: P.kwargs) -> ir.RV:
        output, generated, boundary = trace_call(fun, *args, **kwargs)
        if not isinstance(output, ir.RV):
            raise TypeError("composite function must return exactly one newly created RV")
        if not generated or generated[-1] is not output:
            raise ValueError("every output must be created during the function call")

        op = composite_from_trace(output, generated, boundary, simplify=simplify)
        return ir.RV(op, *boundary, interface=output.interface)

    original_name = getattr(fun, "__name__", None)
    if not original_name or original_name == "<lambda>":
        original_name = "function"
    call.__name__ = f"composite_{original_name}"
    call.__qualname__ = call.__name__
    return call
