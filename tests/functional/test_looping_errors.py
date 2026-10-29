"""Loop failures retain their cause and identify the axis being constructed."""

import pytest

from pangolin import interface as pi
from pangolin.functional import loop


def test_initialization_error_names_branch_and_function():
    def field(prev, t, s):
        if t is None:
            return pi.constant(0.0)
        if s is None:
            return prev[t - 1]
        return prev[t - 1, s] + prev[t, s - 1]

    with pytest.raises(ValueError, match="prev cannot be read during initialization") as caught:
        loop(field, interface=pi.default, s=[3], t=[2])
    assert caught.value.__notes__ == ["While initializing axis 's' of loop 'field'."]


def test_invalid_initialization_output_names_branch():
    def field(prev, t, s):
        if t is None:
            return pi.constant(0.0)
        if s is None:
            return (pi.constant(0.0), pi.constant(1.0))
        return prev[t - 1, s] + prev[t, s - 1]

    with pytest.raises(TypeError, match="must return a single RV") as caught:
        loop(field, interface=pi.default, t=[2], s=[3])
    assert caught.value.__notes__ == ["While initializing axis 's' of loop 'field'."]


def test_unused_outer_carry_identifies_only_failing_scan():
    def field(prev, t, s):
        return prev[t, s - 1] + 1.0

    with pytest.raises(ValueError, match="depend on its carry") as caught:
        pi.loop(field, s=[3, 0.0], t=[2, 0.0])
    assert caught.value.__notes__ == ["While constructing sequential axis 't' of loop 'field'."]


def test_unused_inner_carry_retains_nested_axis_context():
    def field(prev, t, s):
        return prev[t - 1, s] + 1.0

    with pytest.raises(ValueError, match="depend on its carry") as caught:
        pi.loop(field, t=[2, 0.0], s=[3, 0.0])
    assert caught.value.__notes__ == [
        "While constructing sequential axis 's' of loop 'field'.",
        "While constructing sequential axis 't' of loop 'field'.",
    ]


def test_shape_failure_names_sequential_axis():
    def trajectory(prev, t):
        return prev[t - 1] + pi.constant([1.0, 2.0])

    with pytest.raises(ValueError, match="match carry shape") as caught:
        pi.loop(trajectory, t=[3, 0.0])
    assert caught.value.__notes__ == [
        "While constructing sequential axis 't' of loop 'trajectory'."
    ]


def test_random_intermediate_failure_keeps_original_explanation():
    def trajectory(prev, t):
        noise = pi.normal(0, 1)
        return prev[t - 1] + noise

    with pytest.raises(ValueError, match="all but the last Op must be deterministic") as caught:
        pi.loop(trajectory, t=[3, 0.0])
    assert caught.value.__notes__ == [
        "While constructing sequential axis 't' of loop 'trajectory'."
    ]


def test_user_exception_and_existing_notes_are_preserved():
    error = RuntimeError("model-specific failure")
    error.add_note("original note")

    def candidate(i):
        def trajectory(prev, t):
            raise error

        return pi.loop(trajectory, t=[3, 0.0])

    with pytest.raises(RuntimeError, match="model-specific failure") as caught:
        pi.loop(candidate, i=2)
    assert caught.value is error
    assert caught.value.__notes__ == [
        "original note",
        "While constructing sequential axis 't' of loop 'trajectory'.",
        "While constructing vectorized axis 'i' of loop 'candidate'.",
    ]


def test_invalid_axis_specifications_name_axis():
    with pytest.raises(TypeError, match="axis 't': a sequential axis"):
        pi.loop(lambda prev, t: prev[t - 1], t=[])
    with pytest.raises(TypeError, match="axis 't': initializer must be an RV or None"):
        loop(lambda prev, t: prev[t - 1], interface=pi.default, t=[3, 0.0])
    with pytest.raises(TypeError, match="axis 't': initializer must be an RV, a numeric value"):
        pi.loop(lambda prev, t: prev[t - 1], t=[3, lambda: 0.0])
