import numpy as np
from pangolin.interface import makerv, normal, vmap, plate


def run_single_test(a, b, expected_shape):
    funs = [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
        lambda a, b: a**b,
    ]
    print(f"{a=}")
    print(f"{b=}")

    for fun in funs:
        if expected_shape is None:
            # expected to fail
            try:
                c = a + b
                print(f"{c=}")
            except AssertionError as e:
                return
            assert False, "failed to raise assertion error as expected"
        else:
            c = fun(a, b)
            assert c.shape == expected_shape


def run_all_tests(a, b, expected_shape):
    """
    given inputs a and b (int/float or list of int/float)
    test all combinations of adding give the right shape
    """
    # for a_op in [None, makerv, np.array]:
    #     for b_op in [None, makerv, np.array]:
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
    a = [[2, 3, 4], [5, 6, 7]]
    b = [[2, 3, 4], [5, 6, 7]]
    run_all_tests(a, b, (2, 3))


def test_vmap_over_vec():
    a = makerv(np.random.randn(5, 3))
    b = makerv(np.random.randn(5))
    c = vmap(lambda a_i, b_i: a_i + b_i)(a, b)
    assert c.shape == (5, 3)
