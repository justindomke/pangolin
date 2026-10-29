import numpy as np
import pytest

from pangolin.ir import Constant, Normal, RV


def test_constant_zero_dims():
    a = Constant(1)
    assert a.get_shape() == ()


def test_constant_one_dim():
    a = Constant([1, 2, 3])
    assert a.get_shape() == (3,)


def test_constant_two_dims():
    a = Constant([[1, 2, 3], [4, 5, 6]])
    assert a.get_shape() == (2, 3)


def test_value_is_a_numpy_array_with_preserved_dtype_and_shape():
    source = np.array([[1, 2], [3, 4]], dtype=np.int32)
    constant = Constant(source)

    assert isinstance(constant.value, np.ndarray)
    assert constant.value.dtype == np.dtype(np.int32)
    assert constant.value.shape == (2, 2)
    np.testing.assert_array_equal(constant.value, source)


def test_value_does_not_change_when_source_changes():
    source = np.array([1, 2, 3])
    constant = Constant(source)

    source[0] = 100

    np.testing.assert_array_equal(constant.value, [1, 2, 3])


def test_value_and_derived_views_cannot_be_made_writable():
    constant = Constant([1, 2, 3])
    view = constant.value[1:]

    assert not constant.value.flags.writeable
    assert not view.flags.writeable
    with pytest.raises(ValueError):
        constant.value.flags.writeable = True
    with pytest.raises(ValueError):
        view.flags.writeable = True
    with pytest.raises(ValueError):
        constant.value[0] = 100


def test_equal_constants_have_equal_hashes():
    a = Constant(np.array([1, 2, 3], dtype=np.int32))
    b = Constant(np.array([1, 2, 3], dtype=np.int32))

    assert a == b
    assert hash(a) == hash(b)


def test_equality_includes_values():
    assert Constant([1, 2]) != Constant([1, 3])


def test_equality_includes_dtype():
    assert Constant(np.array([1], dtype=np.int32)) != Constant(np.array([1], dtype=np.int64))


def test_equality_includes_shape():
    assert Constant([1, 2]) != Constant([[1, 2]])


def test_equality_ignores_input_memory_order():
    c_order = np.array([[1, 2], [3, 4]], order="C")
    f_order = np.array([[1, 2], [3, 4]], order="F")

    assert Constant(c_order) == Constant(f_order)
    assert hash(Constant(c_order)) == hash(Constant(f_order))


def test_nan_equality_uses_stored_bit_representation():
    a = Constant(np.array([np.nan]))
    b = Constant(np.array([np.nan]))

    assert a == a
    assert a == b
    assert hash(a) == hash(b)

    different_nan = np.array([np.nan])
    different_nan.view(np.uint64)[0] ^= np.uint64(1)
    assert np.isnan(different_nan[0])
    assert a != Constant(different_nan)


def test_signed_zeros_are_distinct():
    assert Constant(0.0) != Constant(-0.0)


def test_equal_constants_share_set_and_dictionary_keys():
    a = Constant([1, 2])
    equivalent = Constant([1, 2])
    distinct = Constant([1, 3])

    assert len({a, equivalent, distinct}) == 2

    values = {a: "a", equivalent: "equivalent", distinct: "distinct"}
    assert len(values) == 2
    assert values[a] == "equivalent"
    assert values[distinct] == "distinct"


def test_constant_is_not_equal_to_another_op_type():
    assert Constant(1) != Normal()


@pytest.mark.parametrize("value", [[], np.empty((2, 0))])
def test_empty_constants_are_rejected(value):
    with pytest.raises(ValueError, match="does not support empty constants"):
        Constant(value)


@pytest.mark.parametrize("value", ["text", ["one", "two"], np.array([object()], dtype=object)])
def test_non_numeric_constants_are_rejected(value):
    with pytest.raises(ValueError, match="does not produce numeric dtype"):
        Constant(value)


def test_wrong_number_of_args():
    try:
        d = Constant(2)
        tmp = RV(Constant(1))
        x = RV(d, tmp)
        assert False
    except TypeError as e:
        assert (
            str(e)
            == "Constant.get_shape() takes 1 positional argument but 2 were given"
        )
