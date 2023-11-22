import pytest
from pangolin import inference_jags, inference_numpyro, inference_stan
import numpy as np
from pangolin.interface import makerv, vmap, plate, RV
from pangolin.interface import normal, beta, binomial
import jax


def run_single_test(a, b, expected_shape):
    if expected_shape is None:
        # expectd to fail
        try:
            c = a + b
        except AssertionError as e:
            return
        assert False, "failed to raise assertion error as expected"
    else:
        c = a + b
        assert c.shape == expected_shape


def run_all_tests(a, b, expected_shape):
    """
    given inputs a and b (int/float or list of int/float)
    test all combinations of adding give the right shape
    """
    for a_op in [None, makerv, np.array]:
        for b_op in [None, makerv, np.array]:
            if (a_op is makerv) or (b_op is makerv):
                print(f"{a_op=} {b_op=}")
                if a_op is None:
                    my_a = a
                else:
                    my_a = a_op(a)
                if b_op is None:
                    my_b = b
                else:
                    my_b = b_op(b)

                run_single_test(my_a, my_b, expected_shape)


def test_scalar_scalar():
    a = 1.1
    b = 2.2
    run_all_tests(a, b, ())


def test_scalar_vector():
    a = 1.1
    b = [2.2, 3.3, 4.4]
    run_all_tests(a, b, (3,))


def test_scalar_matrix():
    a = 1.1
    b = [[2.2, 3.3, 4.4], [5.5, 6.6, 7.7]]
    run_all_tests(a, b, (2, 3))


def test_vector_vector():
    a = [2.2, 3.3, 4.4]
    b = [5.5, 6.6, 7.7]
    run_all_tests(a, b, (3,))


def test_vector_matrix():
    a = [2.2, 3.3, 4.4]
    b = [[2, 3, 4], [5, 6, 7]]
    run_all_tests(a, b, None)


def test_matrix_matrix():
    a = makerv([[2, 3, 4], [5, 6, 7]])
    b = makerv([[2, 3, 4], [5, 6, 7]])
    c = a + b
    assert c.shape == (2, 3)
