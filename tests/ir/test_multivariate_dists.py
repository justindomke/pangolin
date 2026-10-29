import pytest

from pangolin.ir import *


def test_equality():
    assert MultiNormal() == MultiNormal()
    assert Categorical() == Categorical()
    assert Multinomial() == Multinomial()
    assert Dirichlet() == Dirichlet()
    assert MultiNormal() != Categorical()
    assert MultiNormal() != Multinomial()
    assert MultiNormal() != Dirichlet()


def test_multi_normal_shape():
    assert MultiNormal().get_shape((3,), (3, 3)) == (3,)
    assert MultiNormal().get_shape((2,), (2, 2)) == (2,)
    try:
        MultiNormal().get_shape((2,), (3, 3))
        assert False
    except ValueError:
        pass
    try:
        MultiNormal().get_shape((2,), (2, 3))
        assert False
    except ValueError:
        pass
    try:
        MultiNormal().get_shape((2,), (2,))
        assert False
    except ValueError:
        pass


def test_categorical_shape():
    assert Categorical().get_shape((1,)) == ()
    assert Categorical().get_shape((3,)) == ()
    assert Categorical().get_shape((5,)) == ()


@pytest.mark.parametrize("shape", [(), (1, 2)])
def test_categorical_requires_one_dimensional_weights(shape):
    with pytest.raises(ValueError, match="Categorical expects one-dimensional weights"):
        Categorical().get_shape(shape)


def test_multinomial_shape():
    assert Multinomial().get_shape((), (3,)) == (3,)
    bad_args = [[()], [(3,)], [(3,), ()]]
    for args in bad_args:
        try:
            Multinomial().get_shape(*args)
            assert False
        except (TypeError, ValueError):
            pass


def test_dirichlet_shape():
    assert Dirichlet().get_shape((3,)) == (3,)
    bad_args = [[()], [(3,), (3,)]]
    for args in bad_args:
        try:
            Dirichlet().get_shape(*args)
            assert False
        except (TypeError, ValueError):
            pass


def test_wishart_shape():
    assert Wishart().get_shape((), (3, 3)) == (3, 3)


def test_wishart_requires_scalar_degrees_of_freedom():
    with pytest.raises(ValueError, match="degrees of freedom for Wishart must be scalar"):
        Wishart().get_shape((1,), (3, 3))


@pytest.mark.parametrize("scale_shape", [(), (3,), (2, 3), (3, 2), (2, 3, 2)])
def test_wishart_requires_square_matrix_scale(scale_shape):
    with pytest.raises(ValueError, match="scale for Wishart must be a square matrix"):
        Wishart().get_shape((), scale_shape)
