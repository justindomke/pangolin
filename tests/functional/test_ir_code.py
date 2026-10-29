"""Check displayed IR code using the existing Op repr, without testing execution."""

from functools import partial
from typing import NamedTuple

import numpy as np
import pytest

from pangolin import functional, ir
from pangolin import interface as pi
from pangolin.functional.inspecting import ir_code


def test_exports():
    assert functional.ir_code is pi.ir_code is ir_code
    assert pi.Interface().ir_code is ir_code
    assert "ir_code" in functional.__all__
    assert "ir_code" in pi.__all__


def test_scalar_and_broadcasting(capsys):
    def normal(mean: ir.RV, scale: ir.RV) -> ir.RV:
        return pi.normal(mean, scale)

    code_for = ir_code(normal)
    assert code_for(pi.constant(0.0), pi.constant(1.0)) == (
        "def normal(mean, scale):\n"
        "    a = RV(Normal(), mean, scale)\n"
        "    return a"
    )
    assert code_for(pi.constant([0.0, 1.0]), pi.constant(1.0)) == (
        "def normal(mean, scale):\n"
        "    a = RV(VMap(Normal(), [0, None], 2), mean, scale)\n"
        "    return a"
    )
    assert capsys.readouterr().out == ""


def test_constants_created_inside_function():
    assert ir_code(lambda x: pi.normal(x, 1.0))(pi.constant(0.0)) == (
        "def model(x):\n"
        "    a = RV(Constant(1.))\n"
        "    b = RV(Normal(), x, a)\n"
        "    return b"
    )


def test_nested_trees_and_shared_outputs():
    def model(parameters):
        x = parameters["x"][0] + parameters["x"][1]
        return {"both": [x, (x,)], "empty": ({}, [], ())}

    parameters = {"x": [pi.constant(1), pi.constant(2)]}
    assert ir_code(model)(parameters) == (
        "def model(parameters):\n"
        "    a = RV(Add(), parameters['x'][0], parameters['x'][1])\n"
        "    return {'both': [a, (a,)], 'empty': ({}, [], ())}"
    )


def test_nested_positional_and_keyword_trees():
    def model(parameters, *, adjustments):
        mean = parameters[0]["mean"] + adjustments["offsets"][0]
        x = pi.normal(mean, parameters[1][0])
        return ([x, {"mean": mean}], (adjustments["offsets"][1],))

    parameters = ({"mean": pi.constant(0.0)}, [pi.constant(1.0)])
    adjustments = {"offsets": (pi.constant(2.0), pi.constant(3.0))}
    assert ir_code(model)(parameters, adjustments=adjustments) == (
        "def model(parameters, *, adjustments):\n"
        "    a = RV(Add(), parameters[0]['mean'], adjustments['offsets'][0])\n"
        "    b = RV(Normal(), a, parameters[1][0])\n"
        "    return ([b, {'mean': a}], (adjustments['offsets'][1],))"
    )


def test_rearrange_input_tree_without_creating_nodes():
    def rearrange(values):
        return {"last": [values[1]["x"]], "first": (values[0], values[0])}

    values = (pi.constant(0.0), {"x": pi.constant([1.0, 2.0])})
    assert ir_code(rearrange)(values) == (
        "def rearrange(values):\n"
        "    return {'last': [values[1]['x']], 'first': (values[0], values[0])}"
    )


def test_shared_input_container_keeps_distinct_argument_paths():
    seen = []

    def model(left, right):
        seen.extend([left["x"][0], right["x"][0]])
        return {"difference": left["x"][0] - right["x"][0], "inputs": [left, right]}

    shared = {"x": (pi.constant(0.0),)}
    assert ir_code(model)(shared, shared) == (
        "def model(left, right):\n"
        "    a = RV(Sub(), left['x'][0], right['x'][0])\n"
        "    return {'difference': a, 'inputs': "
        "[{'x': (left['x'][0],)}, {'x': (right['x'][0],)}]}"
    )
    assert seen[0] is not seen[1]


def test_shared_output_container_reuses_node_names():
    def model(x):
        shared = {"pair": (pi.exp(x), pi.sin(x))}
        return [shared, (shared,)]

    assert ir_code(model)(pi.constant(0.0)) == (
        "def model(x):\n"
        "    a = RV(Exp(), x)\n"
        "    b = RV(Sin(), x)\n"
        "    return [{'pair': (a, b)}, ({'pair': (a, b)},)]"
    )


def test_dictionary_key_repr_and_output_order():
    def model(values):
        return {"z-last-alphabetically": values[(1, 2)], "a-first": [values[(3,)]]}

    values = {(1, 2): pi.constant(0.0), (3,): pi.constant(1.0)}
    assert ir_code(model)(values) == (
        "def model(values):\n"
        "    return {'z-last-alphabetically': values[(1, 2)], 'a-first': [values[(3,)]]}"
    )


def test_empty_return_trees_and_empty_input_containers():
    def model(empty):
        assert empty == ({}, [], ())
        return {"empty": [(), {}, []]}

    assert ir_code(model)(({}, [], ())) == (
        "def model(empty):\n    return {'empty': [(), {}, []]}"
    )
    for value, expected in [([], "[]"), ({}, "{}"), ((), "()")]:
        assert ir_code(lambda: value)() == f"def model():\n    return {expected}"


def test_inputs_new_nodes_and_nested_captures_in_same_return_tree():
    scale = pi.exponential(1.0)
    namespace = {"settings": [{"scale": (scale,)}]}

    def model(parameters):
        x = pi.normal(parameters["mean"], scale)
        return {"draw": [x], "parameters": (parameters, {"scale": scale})}

    assert ir_code(model, namespace)({"mean": pi.constant(0.0)}) == (
        "def model(parameters):\n"
        "    a = RV(Normal(), parameters['mean'], settings[0]['scale'][0])\n"
        "    return {'draw': [a], 'parameters': "
        "({'mean': parameters['mean']}, {'scale': settings[0]['scale'][0]})}"
    )


def test_vmap_nested_inputs_and_outputs():
    def model(parameters, scales):
        x = pi.normal(parameters["mean"], scales[0])
        return {"draw": (x,), "shifted": [x + parameters["offset"][0]]}

    batched = pi.vmap(model, in_axes=({"mean": 0, "offset": (None,)}, None))
    parameters = {"mean": pi.constant([0.0, 1.0]), "offset": (pi.constant(2.0),)}
    assert ir_code(batched)(parameters, [pi.constant(1.0)]) == (
        "def vmap_model(parameters, scales):\n"
        "    a = RV(VMap(Normal(), [0, None], 2), parameters['mean'], scales[0])\n"
        "    b = RV(VMap(Add(), [0, None], 2), a, parameters['offset'][0])\n"
        "    return {'draw': (a,), 'shifted': [b]}"
    )


def test_scan_nested_inputs_and_tree_of_path_and_summaries():
    def step(previous, inputs):
        return previous + inputs["change"]

    trajectory = pi.scan(step, length=3)

    def model(initial, forcing):
        path = trajectory(initial["state"][0], forcing)
        return {"path": path, "start": initial["state"], "totals": [pi.sum(path, axis=0)]}

    initial = {"state": (pi.constant([0.0, 1.0]),)}
    forcing = {"change": pi.constant(np.zeros((3, 2)))}
    assert ir_code(model)(initial, forcing) == (
        "def model(initial, forcing):\n"
        "    a = RV(Scan(VMap(Add(), [0, 0], 2), 3, [0]), "
        "initial['state'][0], forcing['change'])\n"
        "    b = RV(Sum(axis=0), a)\n"
        "    return {'path': a, 'start': (initial['state'][0],), 'totals': [b]}"
    )


def test_invalid_leaf_deep_inside_keyword_tree_is_rejected_before_call():
    def model(*, parameters):
        pytest.fail("The function should not be called")

    for value in [None, 1.0, True]:
        with pytest.raises(TypeError, match="ir_code inputs"):
            ir_code(model)(parameters={"outer": [(pi.constant(0.0), {"inner": value})]})


def test_distinct_random_nodes_and_repeated_references():
    def model(mean, scale):
        x = pi.normal(mean, scale)
        y = pi.normal(mean, scale)
        return (x + x, x + y, x, y)

    assert ir_code(model)(pi.constant(0.0), pi.constant(1.0)) == (
        "def model(mean, scale):\n"
        "    a = RV(Normal(), mean, scale)\n"
        "    b = RV(Normal(), mean, scale)\n"
        "    c = RV(Add(), a, a)\n"
        "    d = RV(Add(), a, b)\n"
        "    return (c, d, a, b)"
    )


def test_input_occurrences_are_distinct_during_tracing():
    received = []

    def model(a, b):
        received.extend([a, b])
        return a - b

    x = pi.normal(0, 1)
    source = ir_code(model)(x, x)
    assert received[0] is not received[1]
    assert source == "def model(a, b):\n    c = RV(Sub(), a, b)\n    return c"


def test_captured_random_variable_is_a_free_reference():
    scale = pi.exponential(1.0)

    def model(mean):
        return pi.normal(mean, scale)

    assert ir_code(model, locals())(pi.constant(0.0)) == (
        "def model(mean):\n"
        "    a = RV(Normal(), mean, scale)\n"
        "    return a"
    )


def test_capture_name_collision_and_nested_namespace():
    captured = pi.constant(2.0)

    def model(x):
        return x + captured

    source = ir_code(model, {"x": captured, "nested": [captured]})(pi.constant(0.0))
    assert source == "def model(x):\n    a = RV(Add(), x, nested[0])\n    return a"


def test_captured_output_and_omitted_rv_default():
    captured = pi.normal(0.0, 1.0)

    def model(x=captured):
        return [x, x]

    assert ir_code(model, {"captured": captured})() == (
        "def model():\n    return [captured, captured]"
    )


def test_missing_capture_name_raises():
    captured = pi.normal(0.0, 1.0)

    def model(x):
        return x + captured

    for namespace in [None, {}, {"other": pi.normal(0.0, 1.0)}]:
        with pytest.raises(ValueError, match="Captured RV.*namespace"):
            ir_code(model, namespace)(pi.constant(0.0))


def test_capture_references_shadowed_by_parameters_raise():
    captured = pi.constant(1.0)

    def model(x):
        return x + captured

    for namespace in [{"x": captured}, {"x": [captured]}, {"RV": captured}, {"model": captured}]:
        with pytest.raises(ValueError, match="unambiguous"):
            ir_code(model, namespace)(pi.constant(0.0))


def test_alternative_capture_alias_avoids_collision():
    captured = pi.constant(1.0)

    def model(x):
        return x + captured

    assert ir_code(model, {"x": captured, "offset": captured})(pi.constant(0.0)) == (
        "def model(x):\n    a = RV(Add(), x, offset)\n    return a"
    )


def test_multiple_captures_in_same_tree_and_no_shadowing_by_generated_names():
    a = {"mean": pi.normal(0.0, 1.0), "scale": pi.exponential(1.0)}

    def model():
        return pi.normal(a["mean"], a["scale"])

    assert ir_code(model, {"a": a})() == (
        "def model():\n"
        "    b = RV(Normal(), a['mean'], a['scale'])\n"
        "    return b"
    )


def test_captured_output_also_requires_a_namespace_reference():
    captured = pi.constant(1.0)
    with pytest.raises(ValueError, match="Captured RV"):
        ir_code(lambda: captured)()


def test_partial_binds_python_configuration():
    def model(x: ir.RV, scale: float, *, positive: bool) -> ir.RV:
        y = pi.normal(x, scale)
        return pi.exp(y) if positive else y

    fun = partial(model, scale=2.0, positive=True)
    assert ir_code(fun)(pi.constant(0.0)) == (
        "def model(x):\n"
        "    a = RV(Constant(2.))\n"
        "    b = RV(Normal(), x, a)\n"
        "    c = RV(Exp(), b)\n"
        "    return c"
    )


def test_positional_only_keyword_only_and_variadic_inputs():
    def model(x, /, *values, offset, **options):
        return x + values[0] + offset + options["extra"]

    x = pi.constant(0.0)
    assert ir_code(model)(x, x, offset=x, extra=x) == (
        "def model(x, /, *values, offset, **options):\n"
        "    a = RV(Add(), x, values[0])\n"
        "    b = RV(Add(), a, offset)\n"
        "    c = RV(Add(), b, options['extra'])\n"
        "    return c"
    )


def test_unavailable_python_signature():
    class Model:
        @property
        def __signature__(self):
            raise ValueError("unavailable")

        def __call__(self, x, *, offset):
            return x + offset

    x = pi.constant(0.0)
    assert ir_code(Model())(x, offset=x) == (
        "def model(*args, **kwargs):\n"
        "    a = RV(Add(), args[0], kwargs['offset'])\n"
        "    return a"
    )


def test_unused_inputs_and_existing_outputs():
    x = pi.constant(0.0)
    assert ir_code(lambda x, unused: x)(x, x) == "def model(x, unused):\n    return x"


def test_interfaces_are_used_during_tracing_but_not_displayed():
    producer = pi.Interface()
    seen = []

    def model(x):
        seen.append(x.interface)
        return x + 1.0

    source = ir_code(model)(producer.constant(0.0))
    assert seen == [producer]
    assert source == (
        "def model(x):\n"
        "    a = RV(Constant(1.))\n"
        "    b = RV(Add(), x, a)\n"
        "    return b"
    )


def test_invalid_inputs_fail_before_calling_function():
    def model(*args, **kwargs):
        pytest.fail("must not call")

    for value in [0, 1.0, None, True, {"x": [pi.constant(0), 1]}]:
        with pytest.raises(TypeError, match="ir_code inputs"):
            ir_code(model)(value)


def test_custom_trees_are_rejected_instead_of_changing_type():
    class Pair(NamedTuple):
        x: ir.RV
        y: ir.RV

    x = pi.constant(0.0)
    with pytest.raises(NotImplementedError, match="built-in"):
        ir_code(lambda p: p)(Pair(x, x))
    with pytest.raises(NotImplementedError, match="built-in"):
        ir_code(lambda x: Pair(x, x))(x)


def test_invalid_output():
    with pytest.raises(TypeError, match="RV"):
        ir_code(lambda: 1)()  # type: ignore  # Exercise the runtime output contract.


def test_side_effects_occur_once_and_errors_propagate():
    calls = []
    error = ValueError("from model")

    def model(x):
        calls.append(x)
        raise error

    code_for = ir_code(model)
    assert not calls
    with pytest.raises(ValueError) as exc:
        code_for(pi.constant(0.0))
    assert exc.value is error
    assert len(calls) == 1


def test_large_constants_use_existing_abbreviated_repr():
    values = np.arange(2000)

    def model():
        return pi.constant(values)

    with np.printoptions(threshold=10):
        source = ir_code(model)()
        assert "..." in source
        assert source == f"def model():\n    a = RV({ir.Constant(values)!r})\n    return a"


def test_nested_vmap_and_scan_use_existing_op_repr():
    def step(previous: ir.RV, change: ir.RV) -> ir.RV:
        return pi.normal(previous + change, 0.2)

    trajectory = pi.scan(step, length=4)
    batched = pi.vmap(trajectory)
    start = pi.constant([0.0, 1.0])
    changes = pi.constant(np.zeros((2, 4)))
    assert ir_code(batched)(start, changes) == (
        "def vmap_scan_step(previous, change):\n"
        "    a = RV(VMap(Scan(Composite(2, [Add(), Constant(0.2), Normal()], "
        "[[0, 1], [], [2, 3]]), 4, [0]), [0, 0], 2), previous, change)\n"
        "    return a"
    )


def test_nested_vmap_output_tree():
    def model(x):
        return {"exp": pi.exp(x), "sin": (pi.sin(x),)}

    row = pi.vmap(model, simplify=False)
    batched = pi.vmap(row, simplify=False)
    assert ir_code(batched)(pi.constant(np.zeros((2, 3)))) == (
        "def vmap_vmap_model(x):\n"
        "    a = RV(VMap(VMap(Exp(), [0], 3), [0], 2), x)\n"
        "    b = RV(VMap(VMap(Sin(), [0], 3), [0], 2), x)\n"
        "    return {'exp': a, 'sin': (b,)}"
    )


def test_bijected_distribution():
    assert ir_code(lambda x: pi.tforms.exp(pi.normal)(x, 1.0))(pi.constant(0.0)) == (
        "def model(x):\n"
        "    a = RV(Constant(1.))\n"
        "    b = RV(Bijected(Normal(), Bijector(Exp(), Log(), Composite(2, [Log()], [[1]]), "
        "sign=Constant(1))), x, a)\n"
        "    return b"
    )


def test_log_probability():
    assert ir_code(lambda x: pi.log_prob(pi.normal)(x, 0.0, 1.0))(pi.constant(0.0)) == (
        "def model(x):\n"
        "    a = RV(Constant(0.))\n"
        "    b = RV(Constant(1.))\n"
        "    c = RV(LogProb(Normal()), x, a, b)\n"
        "    return c"
    )


def test_cumulative_probability():
    assert ir_code(lambda x: pi.cdf(pi.normal)(x, 0.0, 1.0))(pi.constant(0.0)) == (
        "def model(x):\n"
        "    a = RV(Constant(0.))\n"
        "    b = RV(Constant(1.))\n"
        "    c = RV(CDF(Normal()), x, a, b)\n"
        "    return c"
    )


def test_gradient_accumulates_both_uses_of_input():
    derivative = pi.grad(lambda x: x * x)
    assert ir_code(derivative)(pi.constant(0.0)) == (
        "def grad_function(*args):\n"
        "    a = RV(Constant(1.))\n"
        "    b = RV(VJP(Mul(), 0), a, args[0], args[0])\n"
        "    c = RV(VJP(Mul(), 1), a, args[0], args[0])\n"
        "    d = RV(Add(), b, c)\n"
        "    return d"
    )


def test_custom_op_repr_needs_no_special_handler():
    class Custom(ir.Identity):
        def __repr__(self):
            return "custom display (not executable)"

    def model(x):
        return ir.RV(Custom(), x)

    assert ir_code(model)(pi.constant(0)) == (
        "def model(x):\n"
        "    a = RV(custom display (not executable), x)\n"
        "    return a"
    )
