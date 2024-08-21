from pangolin.ir.multivariate_dists import *


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
    try:
        Categorical().get_shape((1, 2))
        assert False
    except ValueError as e:
        assert str(e) == "Categorical op got input with 2 dims but expected 1."


def test_multinomial_shape():
    assert Multinomial().get_shape((), (3,)) == (3,)
    bad_args = [[()], [(3,)], [(3,),()]]
    for args in bad_args:
        try:
            Multinomial().get_shape(*args)
            assert False
        except (TypeError, ValueError):
            pass


def test_dirichlet_shape():
    assert Dirichlet().get_shape((3,)) == (3,)
    bad_args = [[()], [(3,),(3,)]]
    for args in bad_args:
        try:
            Dirichlet().get_shape(*args)
            assert False
        except (TypeError, ValueError):
            pass
