"""Previous validates a symbolic prefix and delegates the untouched suffix."""

import jax
import numpy as np
import pytest

from pangolin import interface as pi, ir, jax_backend
from pangolin.functional.looping import Previous


def test_select_each_carry_and_allow_nonsequential_prefix():
    a, b, c = (pi.constant(i) for i in range(3))
    x, y = pi.constant([10, 20]), pi.constant([[30, 40], [50, 60]])
    previous = Previous([a, b, c], [None, x, y])
    assert previous[a, b - 1] is x
    assert previous[a, b, c - 1] is y
    assert ir.rv_equal(previous[a, b - 1, 1], x[1])
    assert ir.rv_equal(previous[a, b, c - 1, 0, :], y[0, :])
    with pytest.raises(ValueError, match="not sequential"):
        previous[a - 1]


def test_single_selector_and_empty_suffix_need_no_interface():
    r = ir.RV(ir.Constant(4))
    x = ir.RV(ir.Constant([1, 2]))
    predecessor = ir.RV(ir.Sub(), r, ir.RV(ir.Constant(1)))
    previous = Previous((r,), (x,))
    assert previous[predecessor] is x
    assert previous[(predecessor,)] is x
    with pytest.raises(NotImplementedError, match="getitem"):
        previous[predecessor, 0]


def test_suffix_passed_unchanged_to_carry_interface():
    calls = []
    result = ir.RV(ir.Constant(42))

    class RecordingInterface(ir.RVInterface):
        def getitem(self, rv, *indices):
            calls.append((rv, indices))
            return result

    r, s = pi.constant(1), pi.constant(2)
    x = ir.RV(ir.Constant(0), interface=RecordingInterface())
    previous = Previous([r, s], [None, x])
    suffix = (None, Ellipsis, object(), [1, 2], slice(None), s - 1)
    assert previous[(r, s - 1, *suffix)] is result
    assert len(calls) == 1
    assert calls[0][0] is x
    assert all(a is b for a, b in zip(calls[0][1], suffix, strict=True))
    assert previous[r, s - 1] is x
    assert len(calls) == 1


def test_carry_interface_chooses_vectorized_or_orthogonal_indexing():
    class VectorizedInterface(pi.Interface):
        def getitem(self, x, *indices):
            return self.vindex(x, *indices)

    t = pi.constant(1)
    matrix = [[10, 20, 30], [40, 50, 60]]
    orthogonal = pi.constant(matrix)
    vectorized = VectorizedInterface().constant(matrix)
    left = Previous([t], [orthogonal])[t - 1, [0, 1], [1, 2]]
    right = Previous([t], [vectorized])[t - 1, [0, 1], [1, 2]]
    assert left.shape == (2, 2)
    assert right.shape == (2,)
    a, b = jax_backend.ancestor_sampler((left, right))(jax.random.key(0))
    np.testing.assert_array_equal(a, [[20, 30], [50, 60]])
    np.testing.assert_array_equal(b, [20, 60])


def test_additional_subtraction_in_suffix_is_ordinary_indexing():
    a, b = pi.constant(1), pi.constant(2)
    x = pi.constant([10, 20, 30])
    previous = Previous([a, b], [x, x])
    index = b - 1
    selected = previous[a - 1, index]
    assert selected.parents == (x, index)
    repeated = previous[a, index, index]
    assert repeated.parents == (x, index)


def test_interface_errors_propagate_unchanged():
    failure = RuntimeError("indexing failed")

    class RejectingInterface(ir.RVInterface):
        def getitem(self, rv, *indices):
            raise failure

    t = pi.constant(1)
    x = ir.RV(ir.Constant(0), interface=RejectingInterface())
    with pytest.raises(RuntimeError) as caught:
        Previous([t], [x])[t - 1, :]
    assert caught.value is failure


def test_coordinate_identity_and_exact_subtraction_structure():
    a, b = pi.constant(1), pi.constant(2)
    copy_a, copy_b = pi.constant(1), pi.constant(2)
    x = pi.constant(0)
    previous = Previous([a, b], [None, x])
    for key in (
        (copy_a, b - 1),
        (a, copy_b - 1),
        (a, b - 2),
        (a, b + (-1)),
        (a, ir.RV(ir.Sub(), b, pi.add(0, 1))),
    ):
        with pytest.raises(ValueError, match="current coordinate"):
            previous[key]
    assert previous[a, b - 1] is x


def test_no_predecessor_or_no_coordinates():
    a, b = pi.constant(1), pi.constant(2)
    previous = Previous([a, b], [None, pi.constant(0)])
    for key in ((), a, (a, b), (a, b, b - 1)):
        with pytest.raises(ValueError, match="requires a coordinate - 1"):
            previous[key]
    with pytest.raises(ValueError, match="requires a coordinate - 1"):
        Previous([], [])[()]


def test_nonsequential_predecessor_is_not_skipped():
    a, b = pi.constant(1), pi.constant(2)
    with pytest.raises(ValueError, match="not sequential"):
        Previous([a, b], [None, pi.constant(0)])[a - 1, b - 1]


def test_constructor_validation():
    scalar = pi.constant(1)
    with pytest.raises(ValueError, match="equal lengths"):
        Previous([scalar], [])
    for coordinate in (1, None, [scalar]):
        with pytest.raises(TypeError, match="coordinates must be RVs"):
            Previous([coordinate], [None])
    for shape in ([1], [[1]]):
        with pytest.raises(ValueError, match="scalar RVs"):
            Previous([pi.constant(shape)], [None])
    for carry in (0, False, [scalar], object()):
        with pytest.raises(TypeError, match="carries must be RVs or None"):
            Previous([scalar], [carry])
    with pytest.raises(TypeError, match="must be sequences"):
        Previous([scalar], {0: scalar})
    with pytest.raises(TypeError, match="must be sequences"):
        Previous(scalar, [None])


def test_constructor_copies_sequences():
    r, x = pi.constant(1), pi.constant([10, 20])
    coordinates, carries = [r], [x]
    previous = Previous(coordinates, carries)
    coordinates.clear()
    carries[0] = None
    assert previous.coordinates == (r,)
    assert previous.carries == (x,)
    assert previous[r - 1] is x
