"""Axis declarations preserve explicit-call graphs, initialization, and interfaces."""

import inspect

import numpy as np
import pytest

from pangolin import dag, functional, interface as pi, ir


def test_declaration_and_explicit_call_build_same_graph():
    def model(prev, i, t, s):
        if t is None:
            return pi.normal(i + s, 1.0)
        if s is None:
            return i + t
        return pi.normal(prev[i, t - 1, s] + prev[i, t, s - 1], 0.5)

    @pi.loop
    def declared(prev, i=2, t=[3], s=[4, None]):
        return model(prev, i, t, s)

    explicit = pi.loop(model, s=[4], i=2, t=[3])
    assert declared.shape == explicit.shape == (2, 3, 4)
    left, right = dag.upstream_nodes(declared), dag.upstream_nodes(explicit)
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert a.op == b.op and a.shape == b.shape
        assert [left.index(p) for p in a.parents] == [right.index(p) for p in b.parents]


def test_default_rv_initializer_remains_shared_and_specs_unchanged():
    configured, other = pi.Interface(), pi.Interface()
    start = other.normal(0.0, 1.0)
    specification = [3, start]

    def model(prev, i=2, t=specification):
        return prev[i, t - 1] + i + 1

    signature = inspect.signature(model)
    result = configured.loop(model)
    assert result.interface is other
    assert [v for v in dag.upstream_nodes(result) if v.op.random] == [start]
    assert model.__defaults__[1] is specification
    assert specification[0] == 3 and specification[1] is start
    assert inspect.signature(model) == signature


def test_numeric_default_array_is_coerced_without_mutating_default():
    configured = pi.Interface()
    initial = np.array([1.0, 2.0])
    specification = [3, initial]

    def model(prev, t=specification):
        return prev[t - 1, :] + 1.0

    result = configured.loop(model)
    assert result.shape == (3, 2)
    assert result.interface is configured
    assert specification[1] is initial
    np.testing.assert_array_equal(initial, [1.0, 2.0])


def test_nested_declarations_preserve_shared_hyperparameters():
    hyper = pi.normal(0.0, 1.0)

    @pi.loop
    def paths(i=2):
        drift = pi.normal(hyper, 0.5)

        @pi.loop
        def path(prev, t=[3]):
            if t is None:
                return 0.0
            return pi.normal(prev[t - 1] + drift, 1.0)

        return path

    # One shared hyperparameter, one drift per path, and a vectorized scan.
    random = [v for v in dag.upstream_nodes(paths) if v.op.random]
    assert [v.shape for v in random] == [(), (2,), (2, 3)]
    assert random[0] is hyper
    assert isinstance(paths.op, ir.VMap) and isinstance(paths.op.base_op, ir.Scan)


def test_functional_defaults_require_rv_initializers_and_results():
    def model(prev, t=[3, 0.0]):
        return prev[t - 1] + 1.0

    with pytest.raises(TypeError, match="initializer must be an RV or None"):
        functional.loop(model, interface=pi.default)

    def body(prev, t=[3]):
        if t is None:
            return 0.0
        return prev[t - 1] + 1.0

    with pytest.raises(TypeError, match="must return a single RV"):
        functional.loop(body, interface=pi.default)


def test_defaults_work_for_bound_methods_and_callable_objects():
    class Model:
        def __call__(self, i=2, j=3):
            return pi.normal(i + j, 1.0)

        def walk(self, prev, t=[3, 0.0]):
            return pi.normal(prev[t - 1], 1.0)

    model = Model()
    assert pi.loop(model).shape == (2, 3)
    assert pi.loop(model.walk).shape == (3,)


def test_invalid_declarations_fail_before_calling_function():
    def mixed(i=2):
        pytest.fail("must not call")

    def missing(i, j=3):
        pytest.fail("must not call")

    def missing_prev(t=[3]):
        pytest.fail("must not call")

    def vectorized_prev(prev, i=2):
        pytest.fail("must not call")

    def default_prev(prev=None, t=[3]):
        pytest.fail("must not call")

    def keyword_prev(i=2, *, prev):
        pytest.fail("must not call")

    with pytest.raises(TypeError, match="cannot mix"):
        pi.loop(mixed, i=2)
    with pytest.raises(ValueError, match="missing size for axis 'i'"):
        pi.loop(missing)
    with pytest.raises(ValueError, match="start with a parameter named 'prev'"):
        pi.loop(missing_prev)
    with pytest.raises(ValueError, match="only allowed when an axis is sequential"):
        pi.loop(vectorized_prev)
    with pytest.raises(TypeError, match="prev parameter must not have a default"):
        pi.loop(default_prev)
    with pytest.raises(TypeError, match="fixed positional signature"):
        pi.loop(keyword_prev)


def test_previous_last_and_configured_decorator_are_rejected():
    def old(t, prev):
        pytest.fail("must not call")

    with pytest.raises(ValueError, match="start with a parameter named 'prev'"):
        pi.loop(old, t=[3, 0.0])
    with pytest.raises(TypeError, match="positional argument"):
        pi.loop(i=2)
    with pytest.raises(TypeError, match="positional argument"):
        functional.loop(interface=pi.default, i=2)


def test_positional_only_defaults_and_no_axes():
    @pi.loop
    def x(prev, t=(3, 0.0), /):
        return prev[t - 1] + 1.0

    assert x.shape == (3,)

    @pi.loop
    def scalar():
        return 1.0

    assert scalar.op == ir.Constant(1.0)
