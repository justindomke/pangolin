"""Test call summaries and their Op-display options."""

from functools import partial
from types import FunctionType

import numpy as np
import pytest

from pangolin import interface as pi, ir
from pangolin.functional.inspecting import signature


@pytest.mark.parametrize("in_ops", [None, False, True])
@pytest.mark.parametrize("out_ops", [None, False, True])
@pytest.mark.parametrize("ops", [False, True])
def test_op_options(in_ops: bool | None, out_ops: bool | None, ops: bool):
    if ops and (in_ops is False or out_ops is False):
        with pytest.raises(ValueError, match="conflicts"):
            signature(pi.exp, in_ops=in_ops, out_ops=out_ops, ops=ops)
    else:
        x = pi.normal(0.0, 1.0)
        describe = signature(pi.exp, in_ops=in_ops, out_ops=out_ops, ops=ops)
        expected_input = "RV(normal)" if in_ops or ops else "RV()"
        expected_output = "RV(exp)" if out_ops or ops else "RV()"
        assert describe(x) == f"({expected_input}) -> {expected_output}"


def test_shapes_are_always_shown():
    x = pi.constant([1.0, 2.0, 3.0])
    assert signature(lambda x: x)(x) == "(RV(3,)) -> RV(3,)"
    assert signature(lambda x: x, ops=True)(x) == (
        f"(RV(3, {x.op})) -> RV(3, {x.op})"
    )


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((), "RV()"),
        ((1,), "RV(1,)"),
        ((3,), "RV(3,)"),
        ((2, 3), "RV(2, 3)"),
        ((2, 1, 3), "RV(2, 1, 3)"),
    ],
)
def test_shape_formatting(shape: tuple[int, ...], expected: str):
    x = ir.RV(ir.Constant(np.zeros(shape)))
    assert signature(lambda x: x)(x) == f"({expected}) -> {expected}"


def test_default_options_hide_ops():
    x = pi.normal(0.0, 1.0)
    assert signature(pi.exp)(x) == "(RV()) -> RV()"


def test_ops_use_str_not_repr():
    x = pi.normal(0.0, 1.0)
    assert str(x.op) != repr(x.op)
    assert signature(pi.exp, ops=True)(x) == "(RV(normal)) -> RV(exp)"


def test_vector_op_follows_dimensions_without_trailing_comma():
    x = pi.normal([0.0, 1.0, 2.0], 1.0)
    assert signature(lambda x: x, ops=True)(x) == (
        "(RV(3, vmap(normal, [0, None], 3))) -> RV(3, vmap(normal, [0, None], 3))"
    )


def test_only_final_op_is_shown():
    x = pi.exp(pi.normal(0.0, 1.0))
    assert signature(lambda: x, ops=True)() == "() -> RV(exp)"


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "None"), (True, "True"), (4, "4"), (1.5, "1.5"),
        ("text", "'text'"), (b"text", "b'text'"), ([], "[]"), ({}, "{}"),
        ((), "()"), ((1,), "(1,)"), ((1, 2), "(1, 2)"), ([1], "[1]"),
    ],
)
def test_python_values_and_container_punctuation(value: object, expected: str):
    assert signature(lambda x: x)(value) == f"({expected}) -> {expected}"


def test_nested_mixed_containers():
    x = pi.normal(0.0, 1.0)
    y = pi.constant([1, 2])
    values = {"a": [x, (y,)], "b": (None, {}, [])}
    expected = "{'a': [RV(), (RV(2,),)], 'b': (None, {}, [])}"
    assert signature(lambda values: values)(values) == f"({expected}) -> {expected}"


def test_op_flags_apply_to_every_nested_rv():
    x = pi.normal(0.0, 1.0)
    y = pi.exp(x)
    values = {"a": [x, (y,)]}
    with_ops = "{'a': [RV(normal), (RV(exp),)]}"
    without_ops = "{'a': [RV(), (RV(),)]}"
    assert signature(lambda values: values, in_ops=True)(values) == (
        f"({with_ops}) -> {without_ops}"
    )
    assert signature(lambda values: values, out_ops=True)(values) == (
        f"({without_ops}) -> {with_ops}"
    )


def test_dictionary_key_repr_and_insertion_order():
    values = {(1, 2): pi.constant(0), 3: pi.constant(1), "a": pi.constant(2)}
    expected = "{(1, 2): RV(), 3: RV(), 'a': RV()}"
    assert signature(lambda: values)() == f"() -> {expected}"


def test_other_objects_use_repr():
    class Token:
        def __repr__(self) -> str:
            return "Token()"

        def __str__(self) -> str:
            return "not the repr"

    token = Token()
    assert signature(lambda x: x)(token) == "(Token()) -> Token()"


def test_numpy_arrays_are_not_coerced():
    x = np.array([1, 2])

    def fun(value: np.ndarray) -> np.ndarray:
        assert value is x
        return value

    assert signature(fun)(x) == f"({x!r}) -> {x!r}"


def test_no_arguments():
    assert signature(lambda: pi.normal(0.0, 1.0))() == "() -> RV()"


def test_positional_and_keyword_arguments():
    def fun(a: int, /, b: int, *, c: int) -> int:
        return a + b + c

    describe = signature(fun)
    assert describe(1, 2, c=3) == "(1, 2, c=3) -> 6"
    assert describe(1, b=2, c=3) == "(1, b=2, c=3) -> 6"


def test_variadic_arguments_and_keyword_order():
    def fun(*args: int, **kwargs: int) -> int:
        return sum(args) + sum(kwargs.values())

    assert signature(fun)(1, 2, z=3, a=4) == "(1, 2, z=3, a=4) -> 10"


def test_omitted_and_explicit_defaults():
    def fun(x: int = 3) -> int:
        return x

    describe = signature(fun)
    assert describe() == "() -> 3"
    assert describe(3) == "(3) -> 3"
    assert describe(x=3) == "(x=3) -> 3"


def test_rv_default_is_not_added_to_inputs():
    x = pi.normal(0.0, 1.0)

    def fun(value: ir.RV = x) -> ir.RV:
        assert value is x
        return value

    assert signature(fun, ops=True)() == "() -> RV(normal)"


def test_wrapping_does_not_call_function():
    calls = []

    def fun() -> None:
        calls.append(1)

    describe = signature(fun)
    assert calls == []
    assert describe() == "() -> None"
    assert calls == [1]
    assert describe() == "() -> None"
    assert calls == [1, 1]


def test_conflicting_options_fail_before_calling_function():
    def fun() -> None:
        pytest.fail("The function should not be called")

    with pytest.raises(ValueError, match="conflicts"):
        signature(fun, ops=True, out_ops=False)


def test_inputs_are_described_before_mutation():
    values = [1]

    def fun(x: list[int]) -> list[int]:
        assert x is values
        x.append(2)
        return x

    assert signature(fun)(values) == "([1]) -> [1, 2]"
    assert values == [1, 2]


def test_rv_and_container_identity_are_preserved():
    x = pi.normal(0.0, 1.0)
    shared = [x]

    def fun(a: list[ir.RV], b: list[ir.RV], *, value: ir.RV) -> ir.RV:
        assert a is b is shared
        assert a[0] is value is x
        return value

    assert signature(fun)(shared, shared, value=x) == "([RV()], [RV()], value=RV()) -> RV()"


def test_interfaces_are_preserved():
    producer = pi.Interface(broadcasting="off")
    consumer = pi.Interface(broadcasting="numpy")
    x = producer.normal(0.0, 1.0)
    outputs: list[ir.RV] = []

    def fun(value: ir.RV) -> ir.RV:
        assert value is x
        assert value.interface is producer
        result = consumer.exp(value)
        outputs.append(result)
        return result

    assert signature(fun)(x) == "(RV()) -> RV()"
    assert x.interface is producer
    assert outputs[0].interface is consumer


def test_plain_ir_requires_no_interface():
    x = ir.RV(ir.Constant(1))

    def fun(value: ir.RV) -> ir.RV:
        assert value is x
        return ir.RV(ir.Exp(), value)

    assert signature(fun, out_ops=True)(x) == "(RV()) -> RV(exp)"


def test_returns_string_without_printing(capsys: pytest.CaptureFixture[str]):
    result = signature(lambda: pi.constant(0))()
    assert isinstance(result, str)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_function_printing_is_preserved(capsys: pytest.CaptureFixture[str]):
    def fun() -> int:
        print("called")
        return 1

    assert signature(fun)() == "() -> 1"
    assert capsys.readouterr().out == "called\n"


def test_original_exception_and_prior_side_effects_are_preserved():
    error = ValueError("from the function")
    calls = []

    def fun() -> None:
        calls.append(1)
        raise error

    with pytest.raises(ValueError) as exc:
        signature(fun)()
    assert exc.value is error
    assert calls == [1]


def test_preserves_function_metadata():
    def fun(x: int) -> int:
        """An example function."""
        return x

    describe = signature(fun)
    assert isinstance(describe, FunctionType)
    assert describe.__name__ == fun.__name__
    assert describe.__doc__ == fun.__doc__


def test_callable_object():
    class Model:
        def __call__(self, x: int) -> int:
            return x + 1

    assert signature(Model())(2) == "(2) -> 3"


def test_bound_method():
    class Model:
        def fun(self, x: int) -> int:
            return x + 1

    assert signature(Model().fun)(2) == "(2) -> 3"


def test_partial_function():
    def fun(a: int, b: int) -> int:
        return a + b

    assert signature(partial(fun, 2))(3) == "(3) -> 5"


def test_builtin_function():
    assert signature(len)([1, 2]) == "([1, 2]) -> 2"


def test_function_controls_coercion_and_broadcasting():
    describe = signature(pi.normal)
    assert describe(0.0, 1.0) == "(0.0, 1.0) -> RV()"
    assert describe([0.0, 1.0], 1.0) == "([0.0, 1.0], 1.0) -> RV(2,)"


def test_function_controls_output_structure_on_each_call():
    def fun(vector: bool) -> object:
        if vector:
            return {"x": pi.normal([0.0, 1.0], 1.0)}
        else:
            return pi.normal(0.0, 1.0)

    describe = signature(fun)
    assert describe(False) == "(False) -> RV()"
    assert describe(True) == "(True) -> {'x': RV(2,)}"


def test_function_validation_errors_are_preserved():
    strict = pi.Interface(broadcasting="off")
    with pytest.raises(ValueError):
        signature(strict.normal)([0.0, 1.0], 1.0)


def test_vmapped_function():
    fun = pi.vmap(lambda x: pi.normal(x, 1.0))
    assert signature(fun)(pi.constant([0.0, 1.0])) == "(RV(2,)) -> RV(2,)"


def test_composited_function():
    fun = pi.composite(lambda x: pi.exp(x), simplify=False)
    x = pi.constant(1.0)
    result = fun(x)
    assert isinstance(result.op, ir.Composite)
    assert signature(fun, out_ops=True)(x) == f"(RV()) -> RV({result.op})"


def test_probability_combinator():
    assert signature(pi.log_prob(pi.normal))(0.0, 0.0, 1.0) == "(0.0, 0.0, 1.0) -> RV()"


def test_cdf_combinator():
    assert signature(pi.cdf(pi.normal))(0.0, 0.0, 1.0) == "(0.0, 0.0, 1.0) -> RV()"


def test_gradient_combinator():
    fun = pi.grad(lambda x: x * x)
    assert signature(fun)(pi.constant(2.0)) == "(RV()) -> RV()"


def test_scan_combinator():
    fun = pi.scan(lambda previous, x: previous + x, length=2)
    assert signature(fun)(pi.constant(0.0), pi.constant([1.0, 2.0])) == (
        "(RV(), RV(2,)) -> RV(2,)"
    )


def test_each_call_builds_fresh_nodes():
    outputs: list[ir.RV] = []

    def fun() -> ir.RV:
        x = pi.normal(0.0, 1.0)
        outputs.append(x)
        return x

    describe = signature(fun)
    assert describe() == describe() == "() -> RV()"
    assert len(outputs) == 2
    assert outputs[0] is not outputs[1]


def test_unused_inputs_are_still_shown():
    def fun(x: ir.RV, unused: int) -> ir.RV:
        return x

    assert signature(fun)(pi.constant(0.0), 42) == "(RV(), 42) -> RV()"


def test_rv_dictionary_keys_use_repr_not_rv_summary():
    x = pi.constant(0.0)
    assert signature(lambda: {x: 1})() == f"() -> {{{x!r}: 1}}"


def test_custom_dictionary_keys_use_repr():
    class Key:
        def __repr__(self) -> str:
            return "Key()"

    assert signature(lambda: {Key(): pi.constant(0.0)})() == "() -> {Key(): RV()}"


def test_output_repr_errors_propagate():
    error = RuntimeError("cannot represent this object")

    class Result:
        def __repr__(self) -> str:
            raise error

    with pytest.raises(RuntimeError) as exc:
        signature(lambda: Result())()
    assert exc.value is error


def test_vmap_nested_parameters_with_different_mapped_axes():
    def model(parameters: dict[str, ir.RV], scale: ir.RV):
        x = pi.normal(parameters["mean"], scale)
        y = x + parameters["offset"]
        return {"draw": x, "shifted": (y, [pi.sum(y, axis=0)])}

    batched = pi.vmap(
        model,
        in_axes=({"mean": 0, "offset": 1}, None),
    )
    parameters = {
        "mean": pi.constant(np.zeros((3, 2))),
        "offset": pi.constant(np.ones((2, 3))),
    }
    assert signature(batched)(parameters, pi.constant(1.0)) == (
        "({'mean': RV(3, 2), 'offset': RV(2, 3)}, RV()) -> "
        "{'draw': RV(3, 2), 'shifted': (RV(3, 2), [RV(3,)])}"
    )


def test_nested_vmap_preserves_nested_output_structure():
    def model(mean: ir.RV):
        x = pi.normal(mean, 1.0)
        return {"draw": x, "transforms": [pi.exp(x), (x * x,)]}

    row = pi.vmap(model)
    batched = pi.vmap(row)
    assert signature(batched)(pi.constant(np.zeros((2, 3)))) == (
        "(RV(2, 3)) -> {'draw': RV(2, 3), 'transforms': [RV(2, 3), (RV(2, 3),)]}"
    )


def test_vmap_static_argument_selects_output_tree():
    def model(parameters: tuple[ir.RV, ir.RV], transformed: bool):
        mean, scale = parameters
        x = pi.normal(mean, scale)
        if transformed:
            return {"draw": x, "transformed": [pi.exp(x)]}
        else:
            return (x,)

    describe = signature(pi.vmap(model, in_axes=((0, None), None)))
    parameters = (pi.constant([0.0, 1.0, 2.0]), pi.constant(1.0))
    assert describe(parameters, False) == "((RV(3,), RV()), False) -> (RV(3,),)"
    assert describe(parameters, True) == (
        "((RV(3,), RV()), True) -> {'draw': RV(3,), 'transformed': [RV(3,)]}"
    )


def test_scan_with_nested_inputs_and_vector_carry():
    def step(
        previous: ir.RV,
        inputs: dict[str, tuple[ir.RV, ir.RV]],
        scale: ir.RV,
    ) -> ir.RV:
        change, bias = inputs["forcing"]
        return pi.normal(previous + change + bias, scale)

    trajectory = pi.scan(step, length=4, in_axes=({"forcing": (1, None)}, None))
    inputs = {"forcing": (pi.constant(np.zeros((2, 4))), pi.constant([1.0, 2.0]))}
    assert signature(trajectory)(pi.constant([0.0, 0.0]), inputs, pi.constant(0.1)) == (
        "(RV(2,), {'forcing': (RV(2, 4), RV(2,))}, RV()) -> RV(4, 2)"
    )


def test_vmap_of_scan_and_scan_of_vmap_show_axis_order():
    def step(previous: ir.RV, change: ir.RV) -> ir.RV:
        return pi.normal(previous + change, 1.0)

    trajectory = pi.scan(step, length=4)
    independent_trajectories = pi.vmap(trajectory, in_axes=(0, 0))
    parallel_step = pi.vmap(step)
    joint_trajectory = pi.scan(parallel_step, length=4, in_axes=1)
    start = pi.constant([0.0, 1.0, 2.0])
    changes = pi.constant(np.zeros((3, 4)))
    assert signature(independent_trajectories)(start, changes) == (
        "(RV(3,), RV(3, 4)) -> RV(3, 4)"
    )
    assert signature(joint_trajectory)(start, changes) == (
        "(RV(3,), RV(3, 4)) -> RV(4, 3)"
    )


def test_trajectory_and_batched_probability_summaries_in_one_tree():
    def model(start: ir.RV, changes: ir.RV):
        path = pi.scan(lambda previous, change: pi.normal(previous + change, 1.0), length=4)(
            start, changes
        )
        probabilities = pi.vmap(pi.cdf(pi.normal), in_axes=(0, None, None))(
            path, 0.0, 1.0
        )
        return {"path": path, "summary": (probabilities, [pi.sum(path, axis=0)])}

    describe = signature(pi.vmap(model))
    assert describe(pi.constant([0.0, 1.0]), pi.constant(np.zeros((2, 4)))) == (
        "(RV(2,), RV(2, 4)) -> "
        "{'path': RV(2, 4), 'summary': (RV(2, 4), [RV(2,)])}"
    )


def test_nested_vmap_ops_are_optional_in_every_output_leaf():
    def model(x: ir.RV):
        return {"exp": pi.exp(x), "sin": (pi.sin(x),)}

    row = pi.vmap(model, simplify=False)
    batched = pi.vmap(row, simplify=False)
    x = pi.constant(np.zeros((2, 3)))
    assert signature(batched)(x) == "(RV(2, 3)) -> {'exp': RV(2, 3), 'sin': (RV(2, 3),)}"
    assert signature(batched, out_ops=True)(x) == (
        "(RV(2, 3)) -> {'exp': RV(2, 3, vmap(vmap(exp, [0], 3), [0], 2)), "
        "'sin': (RV(2, 3, vmap(vmap(sin, [0], 3), [0], 2)),)}"
    )
