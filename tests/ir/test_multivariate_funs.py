import pytest

from pangolin.ir import *

# MatMul
# Inv
# Softmax
# Sum


def test_equality():
    assert Matmul() == Matmul()
    assert Inv() == Inv()
    assert Softmax() == Softmax()
    assert Sum(0) == Sum(0)
    assert Sum(1) == Sum(1)
    assert Matmul() != Inv()
    assert Matmul() != Softmax()
    assert Matmul() != Sum(0)
    assert Sum(0) != Sum(1)


def test_softmax_shape():
    assert Softmax().get_shape((3,)) == (3,)


@pytest.mark.parametrize("shape", [(), (2, 3)])
def test_softmax_requires_one_dimensional_input(shape):
    with pytest.raises(ValueError, match="Softmax expects a one-dimensional input"):
        Softmax().get_shape(shape)


@pytest.mark.parametrize(
    ("axis", "expected_shape"),
    [
        (0, (3, 4)),
        (1, (2, 4)),
        (2, (2, 3)),
        (-1, (2, 3)),
        (-2, (2, 4)),
        (-3, (3, 4)),
    ],
)
def test_sum_shape(axis, expected_shape):
    assert Sum(axis).get_shape((2, 3, 4)) == expected_shape


@pytest.mark.parametrize("axis", [3, -4])
def test_sum_rejects_out_of_bounds_axis(axis):
    with pytest.raises(ValueError, match="axis .* is out of bounds"):
        Sum(axis).get_shape((2, 3, 4))


def test_sum_rejects_axis_for_scalar_input():
    with pytest.raises(ValueError, match="array with 0 dimensions"):
        Sum(0).get_shape(())


@pytest.mark.parametrize("axis", [True, None, 1.5, "1"])
def test_sum_requires_integer_axis(axis):
    with pytest.raises(TypeError, match="axis for Sum must be an integer"):
        Sum(axis)
