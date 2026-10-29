from pangolin import interface as pi, ir
from pangolin.ir import RV


import itertools

import numpy as np
import pytest

from pangolin.interface import add
from pangolin.interface import constant
from pangolin.interface import exp
from pangolin.interface import log
from pangolin.interface import normal
from pangolin.functional.tracing import trace_call


def test_trace_generated_nodes_preserves_arguments_and_output_structure():
    value = object()
    options = {"scale": object()}
    observed = []

    def fun(arg, *, settings):
        observed.append((arg, settings))
        a = constant(2)
        b = exp(a)
        return {"first": b, "again": [a, b]}

    output, generated, _ = trace_call(fun, value, settings=options)

    assert observed == [(value, options)]
    assert observed[0][0] is value
    assert observed[0][1] is options
    assert generated == [output["again"][0], output["first"]]
    assert output["first"] is output["again"][1]


def test_trace_generated_nodes_stops_at_preexisting_closure_captures():
    captured = constant(2)

    output, generated, _ = trace_call(lambda: log(exp(captured)))

    assert generated == [output.parents[0], output]
    assert captured not in generated


def test_trace_generated_nodes_rejects_non_rv_output_leaves():
    with pytest.raises(TypeError, match="trace_call output.*found int"):
        trace_call(lambda: {"bad": [constant(1), 2]})


def test_trace_call_calls_the_function_once_and_preserves_its_output():
    calls = []
    x = constant(2)

    def fun(value):
        calls.append(value)
        a = exp(value)
        b = log(a)
        return {"first": b, "again": [a, b]}

    output, generated, boundary = trace_call(fun, x)

    assert calls == [x]
    assert generated == [output["again"][0], output["first"]]
    assert boundary == [x]
    assert output["first"] is output["again"][1]


def test_trace_call_orders_argument_rvs_before_closure_captures():
    captured = constant(10)
    a = constant(1)
    b = constant(2)

    def fun(values):
        return captured + values["b"] + values["a"]

    output, generated, boundary = trace_call(fun, {"b": b, "a": a})

    assert generated == [output.parents[0], output]
    assert boundary == [a, b, captured]


def test_trace_call_orders_closure_captures_by_first_encounter():
    first = constant(1)
    second = constant(2)

    def fun():
        a = second + first
        return first + a

    _, _, boundary = trace_call(fun)

    assert boundary == [second, first]


def test_trace_call_deduplicates_repeated_boundary_references():
    x = constant(2)
    captured = constant(3)

    def fun(values):
        a = values[0] + captured
        return a + values[1] + captured

    _, _, boundary = trace_call(fun, (x, x))

    assert boundary == [x, captured]


def test_trace_call_omits_unused_argument_rvs():
    used = constant(2)
    unused = constant(3)

    _, _, boundary = trace_call(lambda x, y: exp(x), used, unused)

    assert boundary == [used]


def test_trace_call_allows_none_in_arguments():
    x = constant(2)

    output, generated, boundary = trace_call(lambda value, option: exp(value), x, None)

    assert generated == [output]
    assert boundary == [x]


def test_trace_call_treats_existing_rvs_as_boundary_not_their_ancestors():
    ancestor = constant(2)
    x = exp(ancestor)

    _, _, boundary = trace_call(log, x)

    assert boundary == [x]
    assert ancestor not in boundary


def test_trace_call_embeds_new_constants_in_the_generated_graph():
    x = constant(2)

    output, generated, boundary = trace_call(lambda value: normal(value, 1), x)

    assert generated == [output.parents[1], output]
    assert boundary == [x]


def test_trace_call_excludes_new_nodes_that_do_not_contribute():
    x = constant(2)

    def fun(value):
        exp(value)
        return log(value)

    output, generated, boundary = trace_call(fun, x)

    assert generated == [output]
    assert boundary == [x]


def test_trace_call_rejects_non_rv_output_leaves():
    with pytest.raises(TypeError, match="trace_call output.*found int"):
        trace_call(lambda: {"bad": [constant(1), 2]})


def test_trace_call_rejects_nested_none_output_leaves():
    with pytest.raises(TypeError, match="trace_call output.*found NoneType"):
        trace_call(lambda: {"bad": [constant(1), None]})


@pytest.mark.parametrize("bad_output", [None, 1, "rv"])
def test_trace_call_rejects_a_non_rv_root_output(bad_output):
    with pytest.raises(TypeError, match=f"found {type(bad_output).__name__}"):
        trace_call(lambda: bad_output)


def test_trace_call_accepts_a_preexisting_output():
    existing = constant(2)

    output, generated, boundary = trace_call(lambda: existing)

    assert output is existing
    assert generated == []
    assert boundary == [existing]


def test_trace_call_accepts_an_input_as_output():
    existing = constant(2)

    output, generated, boundary = trace_call(lambda x: x, existing)

    assert output is existing
    assert generated == []
    assert boundary == [existing]


def test_trace_call_allows_new_and_preexisting_outputs_in_one_pytree():
    x = constant(2)
    captured = constant(3)

    def fun(value):
        return {"new": exp(value), "old": [captured, captured]}

    output, generated, boundary = trace_call(fun, x)

    assert generated == [output["new"]]
    assert boundary == [x, captured]
    assert output["old"] == [captured, captured]


def test_trace_call_does_not_include_ancestors_of_a_preexisting_output():
    ancestor = constant(2)
    existing = exp(ancestor)

    output, generated, boundary = trace_call(lambda: existing)

    assert output is existing
    assert generated == []
    assert boundary == [existing]
    assert ancestor not in boundary


def test_trace_call_accepts_an_empty_output_pytree():
    output, generated, boundary = trace_call(lambda: {"empty": []})

    assert output == {"empty": []}
    assert generated == []
    assert boundary == []


def test_trace_call_preserves_output_order_while_generated_uses_creation_order():
    x = constant(2)

    def fun(value):
        shared = exp(value)
        first = log(shared)
        second = exp(shared)
        return second, first, shared

    output, generated, boundary = trace_call(fun, x)

    assert generated == [output[2], output[1], output[0]]
    assert boundary == [x]


def test_trace_call_deduplicates_a_shared_generated_subgraph():
    x = constant(2)

    def fun(value):
        shared = exp(value)
        return log(shared), exp(shared)

    output, generated, boundary = trace_call(fun, x)

    assert generated == [output[0].parents[0], output[0], output[1]]
    assert boundary == [x]


def test_trace_call_keeps_distinct_equal_boundary_rvs():
    first = constant(1)
    second = constant(1)

    output, generated, boundary = trace_call(add, first, second)

    assert generated == [output]
    assert boundary == [first, second]
    assert first is not second


def test_trace_call_keeps_distinct_rvs_with_the_same_op_and_parents():
    mean = constant(0)
    scale = constant(1)

    def fun():
        first = RV(ir.Normal(), mean, scale, interface=pi.default)
        second = RV(ir.Normal(), mean, scale, interface=pi.default)
        return first, second

    output, generated, boundary = trace_call(fun)

    assert generated == [output[0], output[1]]
    assert boundary == [mean, scale]
    assert output[0] is not output[1]


def test_trace_call_deduplicates_an_rv_that_is_both_output_and_parent():
    existing = constant(2)

    def fun():
        return existing, exp(existing)

    output, generated, boundary = trace_call(fun)

    assert generated == [output[1]]
    assert boundary == [existing]


def test_trace_call_prioritizes_positional_then_keyword_argument_rvs():
    positional = constant(1)
    first_keyword = constant(2)
    second_keyword = constant(3)
    captured = constant(4)

    def fun(x, *, a, b):
        return captured + b + x + a

    _, _, boundary = trace_call(
        fun,
        positional,
        b=second_keyword,
        a=first_keyword,
    )

    assert boundary == [positional, first_keyword, second_keyword, captured]


def test_trace_call_prioritizes_an_argument_even_when_it_is_encountered_last():
    argument = constant(1)
    captured = constant(2)

    def fun(value):
        return captured + value

    _, _, boundary = trace_call(fun, argument)

    assert boundary == [argument, captured]


def test_trace_call_uses_output_order_for_preexisting_closure_outputs():
    first = constant(1)
    second = constant(2)

    output, generated, boundary = trace_call(lambda: [second, first, second])

    assert output == [second, first, second]
    assert generated == []
    assert boundary == [second, first]


def test_trace_call_coerces_a_scalar_argument_inside_the_call():
    output, generated, boundary = trace_call(exp, 2)

    assert len(generated) == 2
    assert isinstance(generated[0].op, ir.Constant)
    assert generated[0] is output.parents[0]
    assert generated[1] is output
    assert boundary == []


def test_trace_call_coerces_a_new_parent_without_losing_an_existing_parent():
    x = constant(2)

    output, generated, boundary = trace_call(add, x, 3)

    assert generated == [output.parents[1], output]
    assert isinstance(output.parents[1].op, ir.Constant)
    assert boundary == [x]


def test_trace_call_coerces_an_array_like_argument():
    output, generated, boundary = trace_call(exp, np.array([1.0, 2.0]))

    assert generated[-1] is output
    assert output.shape == (2,)
    assert boundary == []


def test_trace_call_allows_non_rv_arguments_to_control_python_execution():
    x = constant(2)

    def fun(value, use_exp, metadata):
        assert metadata is marker
        return exp(value) if use_exp else log(value)

    marker = object()
    output, generated, boundary = trace_call(fun, x, True, marker)

    assert isinstance(output.op, ir.Exp)
    assert generated == [output]
    assert boundary == [x]


def test_trace_call_propagates_exceptions_and_does_not_corrupt_the_next_trace():
    x = constant(2)

    def fail(value):
        exp(value)
        raise RuntimeError("deliberate failure")

    with pytest.raises(RuntimeError, match="deliberate failure"):
        trace_call(fail, x)

    output, generated, boundary = trace_call(log, x)

    assert generated == [output]
    assert boundary == [x]


def test_trace_call_exhaustively_handles_small_unary_dags():
    """Check every three-node unary DAG and every nonempty subset of its possible outputs."""
    for parent_indices in itertools.product(range(2), range(3), range(4)):
        first_existing = constant(0)
        second_existing = constant(1)
        for output_mask in range(1, 1 << 5):
            created_nodes = []

            def fun():
                nodes = [first_existing, second_existing]
                for parent_index in parent_indices:
                    nodes.append(exp(nodes[parent_index]))
                created_nodes.extend(nodes[2:])
                return [node for index, node in enumerate(nodes) if output_mask & (1 << index)]

            output, generated, boundary = trace_call(fun)

            selected_indices = [index for index in range(5) if output_mask & (1 << index)]
            reachable_new_indices = []
            pending_indices = list(selected_indices)
            while pending_indices:
                index = pending_indices.pop()
                if index >= 2 and index not in reachable_new_indices:
                    reachable_new_indices.append(index)
                    pending_indices.append(parent_indices[index - 2])

            expected_generated = [
                created_nodes[index - 2] for index in range(2, 5) if index in reachable_new_indices
            ]
            expected_boundary = [
                existing
                for index, existing in enumerate((first_existing, second_existing))
                if index in selected_indices
            ]
            for index in sorted(reachable_new_indices):
                parent_index = parent_indices[index - 2]
                if parent_index < 2:
                    parent = (first_existing, second_existing)[parent_index]
                    if parent not in expected_boundary:
                        expected_boundary.append(parent)

            assert output == [
                (first_existing, second_existing, *created_nodes)[index]
                for index in selected_indices
            ]
            assert generated == expected_generated
            assert boundary == expected_boundary


def test_trace_call_exhaustively_handles_small_binary_dags():
    """Check ordered and repeated parents in every two-node binary DAG and output subset."""
    for first_parents in itertools.product(range(2), repeat=2):
        for second_parents in itertools.product(range(3), repeat=2):
            parent_indices = (first_parents, second_parents)
            first_existing = constant(0)
            second_existing = constant(1)
            for output_mask in range(1, 1 << 4):
                created_nodes = []

                def fun():
                    nodes = [first_existing, second_existing]
                    for left_index, right_index in parent_indices:
                        nodes.append(
                            RV(
                                ir.Add(),
                                nodes[left_index],
                                nodes[right_index],
                                interface=pi.default,
                            )
                        )
                    created_nodes.extend(nodes[2:])
                    return [node for index, node in enumerate(nodes) if output_mask & (1 << index)]

                output, generated, boundary = trace_call(fun)

                selected_indices = [index for index in range(4) if output_mask & (1 << index)]
                reachable_new_indices = []
                pending_indices = list(selected_indices)
                while pending_indices:
                    index = pending_indices.pop()
                    if index >= 2 and index not in reachable_new_indices:
                        reachable_new_indices.append(index)
                        pending_indices.extend(parent_indices[index - 2])

                expected_generated = [
                    created_nodes[index - 2]
                    for index in range(2, 4)
                    if index in reachable_new_indices
                ]
                expected_boundary = [
                    existing
                    for index, existing in enumerate((first_existing, second_existing))
                    if index in selected_indices
                ]
                for index in sorted(reachable_new_indices):
                    for parent_index in parent_indices[index - 2]:
                        if parent_index < 2:
                            parent = (first_existing, second_existing)[parent_index]
                            if parent not in expected_boundary:
                                expected_boundary.append(parent)

                assert output == [
                    (first_existing, second_existing, *created_nodes)[index]
                    for index in selected_indices
                ]
                assert generated == expected_generated
                assert boundary == expected_boundary
