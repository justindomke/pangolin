from cleanpangolin.ir.multivariate_dists import *


def test_equality():
    assert Categorical() == Categorical()
    assert Categorical() == categorical
    assert MultiNormal() == MultiNormal()
    assert multi_normal == MultiNormal()
    assert multi_normal != categorical
    assert multi_normal != Categorical()


def test_multi_normal_shape():
    assert multi_normal.get_shape((3,), (3, 3)) == (3,)
    assert multi_normal.get_shape((2,), (2, 2)) == (2,)
    try:
        multi_normal.get_shape((2,), (3, 3))
        assert False
    except ValueError:
        pass
    try:
        multi_normal.get_shape((2,), (2, 3))
        assert False
    except ValueError:
        pass
    try:
        multi_normal.get_shape((2,), (2,))
        assert False
    except ValueError:
        pass


def test_categorical_shape():
    assert categorical.get_shape((1,)) == ()
    assert categorical.get_shape((3,)) == ()
    assert categorical.get_shape((5,)) == ()
    try:
        categorical.get_shape((1, 2))
        assert False
    except ValueError as e:
        assert str(e) == "Categorical op got input with 2 dims but expected 1."


def test_multinomial_shape():
    assert multinomial.get_shape((), (3,)) == (3,)
    bad_args = [[()], [(3,)], [(3,),()]]
    for args in bad_args:
        try:
            multinomial.get_shape(*args)
            assert False
        except (TypeError, ValueError):
            pass


def test_dirichlet_shape():
    assert dirichlet.get_shape((3,)) == (3,)
    bad_args = [[()], [(3,),(3,)]]
    for args in bad_args:
        try:
            dirichlet.get_shape(*args)
            assert False
        except (TypeError, ValueError):
            pass
