"""Inspect graph-building functions and emit specialized IR source."""

import inspect
import keyword
from collections.abc import Callable, Mapping
from functools import wraps

import jax

from pangolin import ir, util
from pangolin.util import PyTree

from .tracing import AbstractOp, require_rv_leaves, trace_call

__all__ = ["ir_code", "signature"]


def rv_tree_code(tree: object, names: Mapping[ir.RV, str]) -> str:
    """Display an RV-only tree with references to named graph nodes."""
    if isinstance(tree, ir.RV):
        return names[tree]
    if type(tree) is dict:
        return "{" + ", ".join(
            f"{key!r}: {rv_tree_code(value, names)}" for key, value in tree.items()
        ) + "}"
    if type(tree) is tuple or type(tree) is list:
        items = ", ".join(rv_tree_code(value, names) for value in tree)
        if isinstance(tree, list):
            return f"[{items}]"
        return "(" + items + ("," if len(tree) == 1 else "") + ")"
    raise NotImplementedError("ir_code supports only RVs and built-in list/tuple/dict trees")


def ir_code[**P](
    fun: Callable[P, PyTree[ir.RV]], namespace: Mapping[str, object] | None = None
) -> Callable[P, str]:
    """Return a function that displays a specialized IR graph as Python-like code.

    Every supplied input leaf must be an RV. Each occurrence becomes a fresh abstract input
    with the same shape and interface before calling ``fun`` once. The returned string shows
    a function building the resulting graph using bare IR names, with no imports or prefixes.
    Intermediate assignments preserve node sharing, and the return statement preserves the
    output's list, tuple, or dictionary structure.

    Parameters
    ----------
    fun
        Graph-building function with RV-only input and output trees. Bind fixed Python
        arguments through a closure or ``functools.partial`` before inspection.
    namespace
        Optional mapping such as ``locals()`` containing every captured RV. Captures appear
        as free references, such as ``scale`` or ``parameters['scale']``, not function
        parameters. The first reference whose root name does not conflict with a parameter,
        the displayed function name, or an IR name is used.

    Returns
    -------
    code_for
        Function returning the displayed code as a string without printing it.

    Raises
    ------
    TypeError
        If any supplied input or returned leaf is not an RV.
    ValueError
        If a captured RV has no unambiguous reference in ``namespace``.
    NotImplementedError
        If an input or output uses a custom PyTree rather than built-in lists, tuples, or dicts.

    Notes
    -----
    This describes one traced specialization, not arbitrary Python control flow. Omitted
    defaults remain fixed during tracing. Captured RVs, including RV defaults, remain free
    references. Input ancestry is not copied. Interfaces are used during tracing but omitted from
    the displayed code.

    Input names come from the function signature when available; otherwise ``*args`` and
    ``**kwargs`` are used. Lambdas use the name ``model``. Ops use their existing ``repr``,
    which may abbreviate constants. This is an inspection tool, not an executable serialization
    of the graph.

    Examples
    --------
    Inspect the precise broadcasting behavior of built-in functions:

    >>> from pangolin import interface as pi
    >>> print(ir_code(pi.normal)(pi.constant(0.0), pi.constant(1.0)))
    def normal(mu, sigma):
        a = RV(Normal(), mu, sigma)
        return a
    >>> print(ir_code(pi.normal)(pi.constant([0.0, 1.0]), pi.constant(1.0)))
    def normal(mu, sigma):
        a = RV(VMap(Normal(), [0, None], 2), mu, sigma)
        return a

    Inspect a pairwise squared-distance matrix built with two named ``vfor`` indices:

    >>> def distances(x):
    ...     return pi.vfor(
    ...         lambda i, j: pi.sum((x[i, :] - x[j, :]) ** 2, axis=0),
    ...         i=x.shape[0],
    ...         j=x.shape[0],
    ...     )
    >>> print(ir_code(distances)(pi.constant([[0., 0.], [1., 2.], [3., 1.]])))
    def distances(x):
        a = RV(VMap(VMap(VMap(Sub(), [0, 0], 2), [None, 0], 3), [0, None], 3), x, x)
        b = RV(Constant(2))
        c = RV(VMap(VMap(VMap(Pow(), [0, None], 2), [0, None], 3), [0, None], 3), a, b)
        d = RV(VMap(VMap(Sum(axis=0), [0], 3), [0], 3), c)
        return d

    Input and output PyTree structure remains visible, including repeated references to an RV:

    >>> def model(parameters):
    ...     x = pi.normal(parameters['mean'], parameters['scales'][0])
    ...     return {'draw': x, 'derived': (pi.exp(x), x)}
    >>> print(ir_code(model)({'mean': pi.constant(0.0), 'scales': [pi.constant(1.0)]}))
    def model(parameters):
        a = RV(Normal(), parameters['mean'], parameters['scales'][0])
        b = RV(Exp(), a)
        return {'draw': a, 'derived': (b, a)}

    Captured RVs are referenced through the supplied namespace:

    >>> scale = pi.exponential(1.0)
    >>> def model(mean):
    ...     return pi.normal(mean, scale)
    >>> print(ir_code(model, locals())(pi.constant(0.0)))
    def model(mean):
        a = RV(Normal(), mean, scale)
        return a

    Non-RV inputs are rejected even if the original function accepts them: RV inputs are
    required so they can be replaced with symbolic inputs during tracing.

    >>> pi.normal(0.0, 1.0)
    RV(Normal(), RV(Constant(0.)), RV(Constant(1.)))
    >>> ir_code(pi.normal)(0.0, 1.0)
    Traceback (most recent call last):
        ...
    TypeError: ir_code inputs must be a PyTree whose leaves are RV; found float
    """
    try:
        fun_signature = inspect.signature(fun)
    except (TypeError, ValueError):
        fun_signature = None

    @wraps(fun)
    def code_for(*args: P.args, **kwargs: P.kwargs) -> str:
        inputs = require_rv_leaves((args, kwargs), "ir_code inputs")
        rv_tree_code((args, kwargs), dict.fromkeys(inputs, "input"))
        abstract_inputs: list[ir.RV] = []

        def abstract(value: ir.RV) -> ir.RV:
            result = ir.RV(AbstractOp(value.shape), interface=value.interface)
            abstract_inputs.append(result)
            return result

        abstract_args, abstract_kwargs = jax.tree_util.tree_map(abstract, (args, kwargs))
        if fun_signature is None:
            parameters = [
                inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
                inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
            ]
            arguments = {"args": abstract_args, "kwargs": abstract_kwargs}
        else:
            arguments = fun_signature.bind(*abstract_args, **abstract_kwargs).arguments
            parameters = [
                parameter.replace(default=inspect.Parameter.empty, annotation=inspect.Parameter.empty)
                for name, parameter in fun_signature.parameters.items()
                if name in arguments
            ]

        reserved = set(ir.__all__)
        function_name = getattr(fun, "__name__", "model")
        if not function_name.isidentifier() or keyword.iskeyword(function_name) or function_name in reserved:
            function_name = "model"
        used = reserved | {p.name for p in parameters} | {function_name}

        def new_name() -> str:
            index = 0
            while True:
                name = chr(ord("a") + index % 26) + (str(index // 26) if index >= 26 else "")
                index += 1
                if name not in used:
                    used.add(name)
                    return name

        names = {rv: paths[0] for rv, paths in util.node_names(abstract_inputs, arguments).items()}
        output, generated, boundary = trace_call(fun, *abstract_args, **abstract_kwargs)
        captures = [rv for rv in boundary if rv not in names]
        capture_names = util.node_names(captures, namespace if namespace is not None else {})
        unavailable = used.copy()
        for rv in captures:
            name = next((
                name for name in capture_names.get(rv, [])
                if (root := name.split("[", 1)[0]).isidentifier()
                and not keyword.iskeyword(root)
                and root not in unavailable
                and namespace is not None and root in namespace
            ), None)
            if name is None:
                raise ValueError(
                    "Captured RV has no unambiguous reference in namespace; "
                    "provide a name that does not conflict with the function's parameters or IR names"
                )
            used.add(name.split("[", 1)[0])
            names[rv] = name

        statements = []
        for rv in generated:
            names[rv] = new_name()
            parents = "".join(f", {names[parent]}" for parent in rv.parents)
            statements.append(f"    {names[rv]} = RV({rv.op!r}{parents})")
        statements.append(f"    return {rv_tree_code(output, names)}")
        return f"def {function_name}{inspect.Signature(parameters)}:\n" + "\n".join(statements)

    return code_for


def signature[**P](
    fun: Callable[P, object],
    *,
    in_ops: bool | None = None,
    out_ops: bool | None = None,
    ops: bool = False,
) -> Callable[P, str]:
    """Return a function that describes the inputs and output of each call as a string.

    Calling ``signature(fun)(*args, **kwargs)`` calls ``fun`` once with the supplied
    arguments unchanged and returns the string ``(inputs) -> output``. No coercion, abstract
    substitution, sampling, or numerical evaluation is added by this combinator.
    Any such behavior in ``fun`` itself still occurs.

    RVs are displayed as ``RV()`` for scalars, ``RV(3,)`` for vectors, and
    ``RV(2, 3)`` for matrices. If enabled, the Op's str is appended after the dimensions,
    as in ``RV(3, vmap(normal, [0, None], 3))``. Lists, tuples, and dictionaries are described
    recursively; dictionary keys and other values use their repr. Positional parameter
    names are omitted, while supplied keyword names are retained.

    Parameters
    ----------
    fun
        Function whose calls should be summarized.
    in_ops
        Whether to show the Op of each input RV. None uses ``ops``.
    out_ops
        Whether to show the Op of each output RV. None uses ``ops``.
    ops
        Show both input and output Ops. Default: False. Setting this to True conflicts
        with an explicit ``in_ops=False`` or ``out_ops=False`` and raises ValueError.

    Returns
    -------
    describe_fun
        Function with the same accepted arguments as ``fun``. It returns a summary string,
        rather than ``fun``'s result, without printing it.

    Notes
    -----
    This describes one observed call, not a general static type signature. Ordinary Python
    values can change the output structure or shape, so their values are shown rather than
    just their types. Omitted defaults are not listed. Input descriptions are recorded before
    the call, and Python side effects and exceptions from ``fun`` are preserved.

    An RV's displayed Op is only its own Op, not its parents' graphs. Use `ir.print_upstream`
    to inspect the full graph. Interface properties are neither changed nor displayed.

    Examples
    --------
    Inspect how ``pi.normal`` coerces Python values into RVs and broadcasts scalar and
    array inputs. Inputs are shown as supplied, before coercion:

    >>> from pangolin import interface as pi
    >>> print(signature(pi.normal)(pi.constant(0.0), pi.constant(1.0)))
    (RV(), RV()) -> RV()
    >>> print(signature(pi.normal)(pi.constant([0.0, 1.0]), pi.constant(1.0)))
    (RV(2,), RV()) -> RV(2,)
    >>> print(signature(pi.normal)(0.0, 1.0))
    (0.0, 1.0) -> RV()
    >>> print(signature(pi.normal)([0.0, 1.0], 1.0))
    ([0.0, 1.0], 1.0) -> RV(2,)

    Optionally include Ops alongside shapes, for outputs alone or for both inputs and outputs:

    >>> print(signature(pi.normal, out_ops=True)(pi.constant([0.0, 1.0]), pi.constant(1.0)))
    (RV(2,), RV()) -> RV(2, vmap(normal, [0, None], 2))
    >>> x = pi.normal(0., 1.)
    >>> print(signature(pi.exp, in_ops=True)(x))
    (RV(normal)) -> RV()
    >>> print(signature(pi.exp, ops=True)(x))
    (RV(normal)) -> RV(exp)

    Inspect a pairwise squared-distance matrix built with two named ``vfor`` indices:

    >>> def distances(x):
    ...     return pi.vfor(
    ...         lambda i, j: pi.sum((x[i, :] - x[j, :]) ** 2, axis=0),
    ...         i=x.shape[0],
    ...         j=x.shape[0],
    ...     )
    >>> print(signature(distances)(pi.constant([[0., 0.], [1., 2.], [3., 1.]])))
    (RV(3, 2)) -> RV(3, 3)

    Input and output PyTree structure remains visible:

    >>> def model(parameters):
    ...     x = pi.normal(parameters['mean'], parameters['scales'][0])
    ...     return {'draw': x, 'derived': (pi.exp(x), x)}
    >>> print(signature(model)({'mean': pi.constant(0.0), 'scales': [pi.constant(1.0)]}))
    ({'mean': RV(), 'scales': [RV()]}) -> {'draw': RV(), 'derived': (RV(), RV())}

    Captured RVs are not call arguments and do not appear in the signature:

    >>> scale = pi.exponential(1.0)
    >>> def model(mean):
    ...     return pi.normal(mean, scale)
    >>> print(signature(model)(pi.constant(0.0)))
    (RV()) -> RV()

    PyTree structure and keyword names remain visible, even when the output structure
    depends on a Python argument:

    >>> def model(parameters, *, squared=False):
    ...     x = pi.normal(parameters['mean'], parameters['scale'])
    ...     if squared:
    ...         return {'x': x, 'squared': x * x}
    ...     else:
    ...         return x
    >>> print(signature(model)({'mean': 0.0, 'scale': 1.0}, squared=True))
    ({'mean': 0.0, 'scale': 1.0}, squared=True) -> {'x': RV(), 'squared': RV()}
    >>> print(signature(model)({'mean': 0.0, 'scale': 1.0}, squared=False))
    ({'mean': 0.0, 'scale': 1.0}, squared=False) -> RV()

    """

    if ops and (in_ops is False or out_ops is False):
        raise ValueError("ops=True conflicts with in_ops=False or out_ops=False")
    show_in_ops = ops if in_ops is None else in_ops
    show_out_ops = ops if out_ops is None else out_ops

    def describe(value: object, ops: bool) -> str:
        if isinstance(value, ir.RV):
            fields = [str(size) for size in value.shape]
            if ops:
                fields.append(str(value.op))
            contents = ", ".join(fields)
            if not ops and len(value.shape) == 1:
                contents += ","
            return "RV(" + contents + ")"
        if isinstance(value, dict):
            return "{" + ", ".join(
                f"{key!r}: {describe(child, ops)}" for key, child in value.items()
            ) + "}"
        if isinstance(value, (list, tuple)):
            contents = ", ".join(describe(child, ops) for child in value)
            if isinstance(value, list):
                return "[" + contents + "]"
            return "(" + contents + ("," if len(value) == 1 else "") + ")"
        return repr(value)

    @wraps(fun)
    def describe_fun(*args: P.args, **kwargs: P.kwargs) -> str:
        inputs = [describe(value, show_in_ops) for value in args]
        inputs.extend(
            f"{key}={describe(value, show_in_ops)}" for key, value in kwargs.items()
        )
        output = fun(*args, **kwargs)
        return f"({', '.join(inputs)}) -> {describe(output, show_out_ops)}"

    return describe_fun
