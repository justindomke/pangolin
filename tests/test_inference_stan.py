from pangolin import inference_stan, vmap, normal, makerv, dirichlet
import numpy as np

StanType = inference_stan.StanType


def test_real():
    t = StanType("real", ())
    out = t.declare("var")
    expected = "real var;"
    assert out == expected


# def test_vector():
#     t = StanType("real", (2,))
#     out = t.declare("var")
#     expected = "vector[2] var;"
#     assert out == expected


def test_vector():
    t = StanType("real", (2,))
    out = t.declare("var")
    expected = "array[2] real var;"
    assert out == expected


# def test_matrix():
#     t = StanType("real", (2, 3))
#     out = t.declare("var")
#     expected = "matrix[2,3] var;"
#     assert out == expected


def test_matrix():
    t = StanType("real", (2, 3))
    out = t.declare("var")
    expected = "array[2,3] real var;"
    assert out == expected


def test_real_lower():
    t = StanType("real", (), lower=1)
    out = t.declare("var")
    expected = "real<lower=1> var;"
    assert out == expected


def test_real_upper():
    t = StanType("real", (), upper=3)
    out = t.declare("var")
    expected = "real<upper=3> var;"
    assert out == expected


def test_real_lower_upper():
    t = StanType("real", (), lower=2, upper=3)
    out = t.declare("var")
    expected = "real<lower=2,upper=3> var;"
    assert out == expected


# def test_array_matrix_lower():
#     t = StanType("real", (2, 3, 4, 5), lower=1.5)
#     out = t.declare("var")
#     expected = "array[2,3] matrix<lower=1.5>[4,5] var;"
#     assert out == expected


def test_array_matrix_lower():
    t = StanType("real", (2, 3, 4, 5), lower=1.5)
    out = t.declare("var")
    expected = "array[2,3,4,5] real<lower=1.5> var;"
    assert out == expected


def test_int():
    t = StanType("int", ())
    out = t.declare("var")
    expected = "int var;"
    assert out == expected


def test_int_array():
    t = StanType("int", (2, 3))
    out = t.declare("var")
    expected = "array[2,3] int var;"
    assert out == expected


def test_int_array_lower_upper():
    t = StanType("int", (2, 3), lower=2, upper=3)
    out = t.declare("var")
    expected = "array[2,3] int<lower=2,upper=3> var;"
    assert out == expected


def test_simplex():
    t = StanType("simplex", (3,))
    out = t.declare("var")
    expected = "simplex[3] var;"
    assert out == expected


def test_array_simplex():
    t = StanType("simplex", (5, 3))
    out = t.declare("var")
    expected = "array[5] simplex[3] var;"
    assert out == expected


# def test_stan_type1():
#     x = vmap(normal, None, axis_size=5)(0, 1)
#     t = inference_stan.stan_type(x)
#     assert t.base_type == "real"
#     assert t.shape == (5,)
#     assert t.lower is None
#     assert t.upper is None
#     assert t.declare("var") == "vector[5] var;"


def test_stan_type1():
    x = vmap(normal, None, axis_size=5)(0, 1)
    t = inference_stan.stan_type(x)
    assert t.base_type == "real"
    assert t.shape == (5,)
    assert t.lower is None
    assert t.upper is None
    assert t.declare("var") == "array[5] real var;"


def test_stan_type2():
    a = makerv(np.random.rand(5))
    x = vmap(dirichlet, None, axis_size=4)(a)
    t = inference_stan.stan_type(x)
    assert t.base_type == "simplex"
    assert t.shape == (4, 5)
    assert t.lower is None
    assert t.upper is None
    assert t.declare("var") == "array[4] simplex[5] var;"


# def test_types2():
#     t = StanType("simplex", (2, 3, 4))
#     out = t.declare("var")
#     expected = "array[2,3] simplex[4] var;"
#     assert out == expected
