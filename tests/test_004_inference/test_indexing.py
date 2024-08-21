import numpy as np
from pangolin import makerv, vmap, inference
import numpyro
import pytest
import jax

def inference_numpyro(var):
    model, names = inference.numpyro.get_model_flat([var],[],[])
    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.plate("multiple_samples", 10000):
            out = model()
    return out[names[var]]

# automatically run tests on all of these
pytestmark = pytest.mark.parametrize("inference", [inference_numpyro])


# def test_indexingA(inference):
#     x_numpy = np.random.randn(3, 7)
#     y_numpy = x_numpy[:, [0, 1, 2]]
#     x = makerv(x_numpy)
#     y = x[:, [0, 1, 2]]
#     y_samp = inference.sample_flat([y], [], [], niter=1)[0]
#     assert np.allclose(y_numpy, y_samp)
#
#
# def test_indexingB(inference):
#     x_numpy = np.random.randn(3, 5, 7)
#     idx1 = np.random.randint(low=0, high=3, size=[11, 13])
#     idx2 = np.random.randint(low=0, high=3, size=[11, 13])
#     y_numpy = x_numpy[:, idx1, idx2]
#     x = makerv(x_numpy)
#     y = x[:, idx1, idx2]
#     y_samp = inference.sample_flat([y], [], [], niter=1)[0]
#     assert np.allclose(y_numpy, y_samp)
#
#
# def test_indexingC(inference):
#     if inference in [inference_jags, inference_stan]:
#         return  # jags and stan don't support step in slices
#
#     x_numpy = np.random.randn(3, 5, 7)
#     idx0 = np.random.randint(low=0, high=3, size=[11, 13])
#     idx2 = np.random.randint(low=0, high=3, size=[11, 13])
#     y_numpy = x_numpy[idx0, 1::2, idx2]
#     x = makerv(x_numpy)
#     y = x[idx0, 1::2, idx2]
#     y_samp = inference.sample_flat([y], [], [], niter=1)[0]
#     assert np.allclose(y_numpy, y_samp)


def assert_one_sample_close(inference, var, value):
    """
    assert that a single sample of var is close to value
    """
    # inf = inference_jags.JAGSInference(niter=1)
    out = inference(var)
    assert np.allclose(out, value)


def test_indexing1(inference):
    x = np.random.randn(5)
    y = makerv(x)[:]
    expected = x[:]
    assert_one_sample_close(inference, y, expected)


def test_indexing2(inference):
    x = np.random.randn(10)
    y = makerv(x)[2:8]
    expected = x[2:8]
    assert_one_sample_close(inference, y, expected)


def test_indexing3(inference):
    x = np.random.randn(5, 7)
    y = makerv(x)[:]
    expected = x[:]
    assert_one_sample_close(inference, y, expected)


def test_indexing4(inference):
    x = np.random.randn(5, 7)
    y = makerv(x)[2:4]
    expected = x[2:4]
    assert_one_sample_close(inference, y, expected)


def test_indexing5(inference):
    # if inference in [inference_jags, inference_stan]:
    #     return  # jags and stan don't support step in slices

    x = np.random.randn(5, 7)
    y = makerv(x)[2:4, 1:2]
    expected = x[2:4, 1:2]
    assert_one_sample_close(inference, y, expected)


def test_indexing6(inference):
    x = np.random.randn(5, 7)
    y = makerv(x)[[2, 3]]
    expected = x[[2, 3]]
    assert_one_sample_close(inference, y, expected)


def test_indexing7(inference):
    x = np.random.randn(3, 3)
    y = makerv(x)[2]
    expected = x[2]
    assert_one_sample_close(inference, y, expected)


def test_indexing8(inference):
    x = np.random.randn(3, 3)
    y = makerv(x)[2, :]
    expected = x[2, :]
    assert_one_sample_close(inference, y, expected)


def test_indexing9(inference):
    x = np.random.randn(3, 3)
    y = makerv(x)[2, 1]
    expected = x[2, 1]
    assert_one_sample_close(inference, y, expected)


def test_indexing10(inference):
    x = np.random.randn(3, 3, 5)
    y = makerv(x)[2, 1]
    expected = x[2, 1]
    assert_one_sample_close(inference, y, expected)


def test_indexing11(inference):
    x = np.random.randn(3, 4, 5)
    y = makerv(x)[2, :, 4]
    expected = x[2, :, 4]
    assert_one_sample_close(inference, y, expected)


def test_indexing12(inference):
    x = np.random.randn(5, 5)
    y = makerv(x)[:, [1, 4, 2]]
    expected = x[:, [1, 4, 2]]
    assert_one_sample_close(inference, y, expected)


def test_indexing13(inference):
    x = np.random.randn(5, 5, 5)
    y = makerv(x)[:, [1, 4, 2]]
    expected = x[:, [1, 4, 2]]
    assert_one_sample_close(inference, y, expected)


def test_indexing14(inference):
    x = np.random.randn(5, 5)
    y = makerv(x)[[2, 1, 2], [1, 4, 2]]
    expected = x[[2, 1, 2], [1, 4, 2]]
    assert_one_sample_close(inference, y, expected)


def test_indexing15(inference):
    x = np.random.randn(5, 5, 7)
    y = makerv(x)[[2, 1, 2], [1, 4, 2]]
    expected = x[[2, 1, 2], [1, 4, 2]]
    assert_one_sample_close(inference, y, expected)


def test_indexing16(inference):
    x = np.random.randn(5, 5, 7)
    y = makerv(x)[:, [2, 1, 2], [1, 4, 2]]
    expected = x[:, [2, 1, 2], [1, 4, 2]]
    assert_one_sample_close(inference, y, expected)


def test_indexing17(inference):
    x = np.random.randn(6, 5, 7)
    y = makerv(x)[[2, 1, 2], 1:4, [1, 4, 2]]
    expected = x[[2, 1, 2], 1:4, [1, 4, 2]]
    assert_one_sample_close(inference, y, expected)


def test_indexing18(inference):
    x = np.random.randn(5, 5, 7)
    y = makerv(x)[[2, 1, 2], 1:4]
    expected = x[[2, 1, 2], 1:4]
    assert_one_sample_close(inference, y, expected)


def test_indexing19(inference):
    x = np.random.randn(3, 5)
    #y = plate(makerv(x))(lambda x_i: x_i[1])
    y = vmap(lambda x_i: x_i[1])(makerv(x))

    expected = x[:, 1]
    assert_one_sample_close(inference, y, expected)


def test_indexing20(inference):
    x = np.random.randn(3, 5)
    #y = plate(makerv(x))(lambda x_i: x_i[[3, 1, 0, 2]])
    y = vmap(lambda x_i: x_i[[3, 1, 0, 2]])(makerv(x))

    expected = x[:, [3, 1, 0, 2]]
    assert_one_sample_close(inference, y, expected)


def test_indexing21(inference):
    x = np.random.randn(7, 5)
    #y = plate(makerv(x), in_axes=1)(lambda x_i: x_i[[3, 1, 0, 2]])
    y = vmap(lambda x_i: x_i[[3,1,0,2]], in_axes=1)(x)

    expected = x[[3, 1, 0, 2], :].T
    assert_one_sample_close(inference, y, expected)


def test_indexing22(inference):
    x = np.random.randn(7, 5)
    #y = plate(makerv(x), in_axes=1)(lambda x_i: x_i[:])
    y = vmap(lambda x_i: x_i[:], in_axes=1)(makerv(x))

    expected = x.T
    assert_one_sample_close(inference, y, expected)


def test_indexing23(inference):
    x = np.random.randn(7, 5)
    expected = jax.vmap(lambda x_i: x_i[:], in_axes=1)(x)
    y = vmap(lambda x_i: x_i[:], in_axes=1)(makerv(x))
    assert_one_sample_close(inference, y, expected)


def test_indexing24(inference):
    x = np.random.randn(7, 5, 11)
    expected = jax.vmap(lambda x_i: x_i[4, 2:4], in_axes=1)(x)
    y = vmap(lambda x_i: x_i[4, 2:4], in_axes=1)(makerv(x))
    assert_one_sample_close(inference, y, expected)


def test_indexing29(inference):
    x = np.random.randn(7, 5, 11, 13, 17)
    expected = jax.vmap(lambda x_i: x_i[:, [1, 2, 1], 2:4, :], in_axes=2)(x)
    y = vmap(lambda x_i: x_i[:, [1, 2, 1], 2:4, :], in_axes=2)(makerv(x))
    assert_one_sample_close(inference, y, expected)


def test_indexing30(inference):
    """
    simplest possible test to trigger advanced indexing
    """
    x = np.random.randn(4, 5, 6, 7)
    y = makerv(x)[:, [2, 1, 2], :, [1, 4, 2]]
    expected = x[:, [2, 1, 2], :, [1, 4, 2]]
    assert_one_sample_close(inference, y, expected)


def test_indexing31(inference):
    """
    comically complicated indexing and vmap
    this tries to test *everything*:
    * the starting array has 7 dimensions
    * we vmap over all pairs of 2 axes
    * then we do a super complex index using a mixture of slicing and advanced
    indexing and also dimensions that are just left implicit
    """

    x = np.random.randn(4, 5, 6, 4, 5, 3, 3)

    # try all possible vmap pairs!
    for in_axis1 in range(7):
        for in_axis2 in range(6):
            # only do 10% of cases
            if hash((in_axis1, in_axis2)) % 10 != 0:
                continue

            # since stan is expensive, only do 10% of cases
            # if inference == inference_stan:
            #     if hash((in_axis1, in_axis2)) % 10 != 0:
            #         continue

            y = vmap(
                lambda x_i: vmap(
                    lambda x_ij: x_ij[:, [1, 2, 1], 2:4, [0, 0, 0]], in_axes=in_axis2
                )(x_i),
                in_axes=in_axis1,
            )(makerv(x))
            expected = jax.vmap(
                lambda x_i: jax.vmap(
                    lambda x_ij: x_ij[:, [1, 2, 1], 2:4, [0, 0, 0]], in_axes=in_axis2
                )(x_i),
                in_axes=in_axis1,
            )(x)
            assert_one_sample_close(inference, y, expected)


def test_indexing32(inference):
    """
    test 2-d index arrays!
    """
    x = np.random.randn(5)
    idx = np.array([[0, 2], [3, 4]])
    y = makerv(x)[idx]
    expected = x[idx]
    assert_one_sample_close(inference, y, expected)


def test_indexing33(inference):
    x = np.random.randn(5, 6)
    idx = np.array([[0, 2], [3, 4]])
    y = makerv(x)[:, idx]
    expected = x[:, idx]
    assert_one_sample_close(inference, y, expected)


def test_indexing34(inference):
    x = np.random.randn(5, 6)
    idx = np.array([[0, 2], [3, 4]])
    y = makerv(x)[idx, :]
    expected = x[idx, :]
    assert_one_sample_close(inference, y, expected)


def test_indexing35(inference):
    """
    indices separated to trigger advanced indexing
    """
    x = np.random.randn(5, 6, 5)
    idx = np.array([[0, 2], [3, 4]])
    y = makerv(x)[idx, :, idx]
    expected = x[idx, :, idx]
    assert_one_sample_close(inference, y, expected)


def test_indexing36(inference):
    """
    indices together to trigger non-advanced indexing
    """
    x = np.random.randn(5, 6, 5)
    idx = [[0, 2], [3, 4]]
    y = makerv(x)[:, idx, idx]
    expected = x[:, idx, idx]
    assert_one_sample_close(inference, y, expected)


def test_indexing37(inference):
    """
    indices together to trigger non-advanced indexing
    """
    x = np.random.randn(5, 6, 5, 4)
    idx = [[0, 2], [3, 4]]
    y = makerv(x)[:, idx, idx, :]
    expected = x[:, idx, idx, :]
    assert_one_sample_close(inference, y, expected)


def test_indexing38(inference):
    """
    indices together inside vmap
    """
    x = np.random.randn(5, 6, 5, 7)
    idx = [[0, 2], [3, 4]]
    y = vmap(lambda x_i: x_i[:, idx, idx], in_axes=0)(makerv(x))
    expected = jax.vmap(lambda x_i: x_i[:, idx, idx], in_axes=0)(x)
    assert_one_sample_close(inference, y, expected)


def test_indexing39(inference):
    """
    indices separated inside vmap
    """
    x = np.random.randn(5, 6, 5, 7)
    idx = [[0, 2], [3, 4]]
    y = vmap(lambda x_i: x_i[idx, :, idx], in_axes=0)(makerv(x))
    expected = jax.vmap(lambda x_i: x_i[idx, :, idx], in_axes=0)(x)
    assert_one_sample_close(inference, y, expected)


def test_indexing40(inference):
    """
    even more comically complicated indexing and vmap
    this tries to test *everything*:
    * the starting array has 7 dimensions
    * we vmap over all pairs of 2 axes
    * then have a mixture of full slices, short slice, and 2-d indices
    * then we do a super complex index using a mixture of slicing and advanced
    indexing and also dimensions that are just left implicit
    """

    x = np.random.randn(4, 3, 4, 4, 5, 3, 3)

    # try all possible vmap pairs!
    for in_axis1 in range(7):
        for in_axis2 in range(6):
            # only do 10% of cases
            if hash((in_axis1, in_axis2)) % 10 != 0:
                continue

            y = vmap(
                lambda x_i: vmap(
                    lambda x_ij: x_ij[
                        :, [[1, 2, 1], [0, 0, 0]], 2:4, [[0, 0, 0], [1, 0, 1]]
                    ],
                    in_axes=in_axis2,
                )(x_i),
                in_axes=in_axis1,
            )(makerv(x))
            expected = jax.vmap(
                lambda x_i: jax.vmap(
                    lambda x_ij: x_ij[
                        :, [[1, 2, 1], [0, 0, 0]], 2:4, [[0, 0, 0], [1, 0, 1]]
                    ],
                    in_axes=in_axis2,
                )(x_i),
                in_axes=in_axis1,
            )(x)
            assert_one_sample_close(inference, y, expected)


def test_indexing41(inference):
    """
    like previous but indices put next to each other to avoid advanced indexing
    """

    x = np.random.randn(4, 5, 4, 4, 5, 4, 4)

    # try all possible vmap pairs!
    for in_axis1 in range(1):
        for in_axis2 in range(1):
            # since stan is expensive, only do 10% of cases
            # if inference == inference_stan:
            #     if hash((in_axis1, in_axis2)) % 10 != 0:
            #         continue

            y = vmap(
                lambda x_i: vmap(
                    lambda x_ij: x_ij[
                        :, [[1, 2, 1], [0, 0, 0]], [[0, 0, 0], [1, 0, 1]], 2:4
                    ],
                    in_axes=in_axis2,
                )(x_i),
                in_axes=in_axis1,
            )(makerv(x))
            expected = jax.vmap(
                lambda x_i: jax.vmap(
                    lambda x_ij: x_ij[
                        :, [[1, 2, 1], [0, 0, 0]], [[0, 0, 0], [1, 0, 1]], 2:4
                    ],
                    in_axes=in_axis2,
                )(x_i),
                in_axes=in_axis1,
            )(x)
            assert_one_sample_close(inference, y, expected)


