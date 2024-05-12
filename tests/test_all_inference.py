"""
A set of tests that can be run on any inference method that implements sample_flat.
Currently tests JAGS, Stan, and Numpyro
"""

import pytest
from pangolin import (
    inference_jags,
    inference_numpyro,
    inference_stan,
    inference_numpyro_modelbased,
)
import numpy as np
from pangolin.interface import makerv, vmap, plate
from pangolin.interface import (
    normal,
    beta,
    binomial,
    exp,
    abs,
    sum,
    dirichlet,
    multinomial,
)
import jax
from jax import numpy as jnp
from pangolin import ezstan

inference_engines = [
    inference_jags,
    inference_numpyro,
    inference_stan,
    inference_numpyro_modelbased,
]

# automatically run tests on all of these
pytestmark = pytest.mark.parametrize("inference", inference_engines)


def test_normal(inference):
    x = normal(0, 1)
    [xs] = inference.sample_flat([x], [], [], niter=10000)
    assert np.abs(np.mean(xs) - 0.0) < 0.1


def test_cond_normal(inference):
    z = normal(0.0, 1.0)
    x = normal(z, 1.0)
    [zs] = inference.sample_flat([z], [x], [np.array(1.0)], niter=10)
    assert zs.shape == (10,)


def test_branched_sampling1(inference):
    z = normal(0.0, 1)
    x = normal(z, 1)
    y = normal(z, 1.0)

    # print(f"{hash(z.parents[0])=}")
    # print(f"{hash(z.parents[1])=}")
    # print(f"{hash(z)=}")
    # print(f"{hash(x.parents[0])=}")
    # print(f"{hash(x.parents[1])=}")
    # print(f"{hash(x)=}")
    # print(f"{hash(y.parents[0])=}")
    # print(f"{hash(y.parents[1])=}")
    # print(f"{hash(y)=}")

    [ys] = inference.sample_flat([y], [x], [np.array(1.0)], niter=10)
    assert ys.shape == (10,)


def test_branched_sampling2(inference):
    z = normal(0.0, 1.0)
    x = normal(z, 0.00001)
    y = normal(z, 0.00001)  # ignored
    x_obs = np.array(0.5)
    [zs] = inference.sample_flat([z], [x], [x_obs], niter=100)
    assert zs.shape == (100,)
    assert max(np.abs(zs - x_obs)) < 0.01


def test_branched_sampling3(inference):
    z = normal(0.0, 1.0)
    x = normal(z, 0.00001)
    y = normal(z, 0.00001)
    x_obs = np.array(0.5)
    [zs, ys] = inference.sample_flat([z, y], [x], [x_obs], niter=100)
    assert zs.shape == (100,)
    assert max(np.abs(zs - x_obs)) < 0.01
    assert max(np.abs(ys - x_obs)) < 0.01


def test_dirichlet_multinomial(inference):
    if inference in [inference_numpyro]:
        return  # no support for constraints (yet)

    alpha = np.array([1.1, 1.3, 1.6])
    z = dirichlet(alpha)
    x = multinomial(5, z)
    x_val = np.array([2, 1, 2])
    niter = 10_000
    [zs] = inference.sample_flat([z], [x], [x_val], niter=niter)
    mean_zs = np.mean(zs, axis=0)
    alpha_post = alpha + x_val
    mean_post = alpha_post / np.sum(alpha_post)
    print(f"{mean_zs=}")
    print(f"{mean_post=}")
    assert np.max(np.abs(mean_zs - mean_post)) < 0.1


def test_deterministic_sample1(inference):
    x = normal(0.0, 1.0)
    y = x**2
    (xs, ys) = inference.sample_flat([x, y], [], [], niter=100)
    assert xs.shape == (100,)
    assert ys.shape == (100,)
    print(f"{xs**2=}")
    print(f"{ys=}")
    assert np.allclose(ys, xs**2, rtol=1e-3)


def test_normal_mean_std(inference):
    x = normal(0.5, 2.7)
    niter = 10_000
    # won't actually do MCMC because it can use ancestor sampling
    [xs] = inference.sample_flat([x], [], [], niter=niter)
    assert xs.shape == (niter,)
    assert np.abs(np.mean(xs) - 0.5) < 0.1
    assert np.abs(np.std(xs) - 2.7) < 0.1


def test_normal_plus_deterministic_mean_std(inference):
    x = normal(0.5, 2.7)
    y = 3 * x + x**2
    niter = 10_000
    [xs, ys] = inference.sample_flat([x, y], [], [], niter=niter)
    assert xs.shape == (niter,)
    assert ys.shape == (niter,)
    assert np.abs(np.mean(xs) - 0.5) < 0.1
    assert np.abs(np.std(xs) - 2.7) < 0.1
    assert np.allclose(ys, 3 * xs + xs**2, rtol=1e-3, atol=1e-3)


def test_cond_normal_mean_var(inference):
    x = normal(0, 1)
    y = normal(x, 1)
    niter = 10_000
    y_val = np.array(1.0)
    [xs] = inference.sample_flat([x], [y], [y_val], niter=niter)
    assert xs.shape == (niter,)
    assert np.abs(np.mean(xs) - 0.5) < 0.03
    assert np.abs(np.var(xs) - 0.5) < 0.03


def test_triple_normal_mean_var(inference):
    x = normal(0, 1)
    y = normal(x, 1)
    z = normal(y, 1)
    niter = 10_000
    x_val = np.array(-2.0)
    z_val = np.array(7.0)
    [ys] = inference.sample_flat([y], [x, z], [x_val, z_val], niter=niter)
    assert ys.shape == (niter,)
    assert np.abs(np.mean(ys) - (-2 + 7) / 2) < 0.03
    assert np.abs(np.var(ys) - 1 / 2) < 0.03


def test_indexingA(inference):
    x_numpy = np.random.randn(3, 7)
    y_numpy = x_numpy[:, [0, 1, 2]]
    x = makerv(x_numpy)
    y = x[:, [0, 1, 2]]
    y_samp = inference.sample_flat([y], [], [], niter=1)[0]
    assert np.allclose(y_numpy, y_samp)


def test_indexingB(inference):
    x_numpy = np.random.randn(3, 5, 7)
    idx1 = np.random.randint(low=0, high=3, size=[11, 13])
    idx2 = np.random.randint(low=0, high=3, size=[11, 13])
    y_numpy = x_numpy[:, idx1, idx2]
    x = makerv(x_numpy)
    y = x[:, idx1, idx2]
    y_samp = inference.sample_flat([y], [], [], niter=1)[0]
    assert np.allclose(y_numpy, y_samp)


def test_indexingC(inference):
    if inference in [inference_jags, inference_stan]:
        return  # jags and stan don't support step in slices

    x_numpy = np.random.randn(3, 5, 7)
    idx0 = np.random.randint(low=0, high=3, size=[11, 13])
    idx2 = np.random.randint(low=0, high=3, size=[11, 13])
    y_numpy = x_numpy[idx0, 1::2, idx2]
    x = makerv(x_numpy)
    y = x[idx0, 1::2, idx2]
    y_samp = inference.sample_flat([y], [], [], niter=1)[0]
    assert np.allclose(y_numpy, y_samp)


def assert_one_sample_close(inference, var, value):
    """
    assert that a single sample of var is close to value
    """
    # inf = inference_jags.JAGSInference(niter=1)
    [out] = inference.sample_flat([var], [], [], niter=1)
    assert np.allclose(out[0], value)


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
    if inference in [inference_jags, inference_stan]:
        return  # jags and stan don't support step in slices

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
    y = plate(makerv(x))(lambda x_i: x_i[1])

    expected = x[:, 1]
    assert_one_sample_close(inference, y, expected)


def test_indexing20(inference):
    x = np.random.randn(3, 5)
    y = plate(makerv(x))(lambda x_i: x_i[[3, 1, 0, 2]])

    expected = x[:, [3, 1, 0, 2]]
    assert_one_sample_close(inference, y, expected)


def test_indexing21(inference):
    x = np.random.randn(7, 5)
    y = plate(makerv(x), in_axes=1)(lambda x_i: x_i[[3, 1, 0, 2]])

    expected = x[[3, 1, 0, 2], :].T
    assert_one_sample_close(inference, y, expected)


def test_indexing22(inference):
    x = np.random.randn(7, 5)
    y = plate(makerv(x), in_axes=1)(lambda x_i: x_i[:])

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
            # since stan is expensive, only do 10% of cases
            if inference == inference_stan:
                if hash((in_axis1, in_axis2)) % 10 != 0:
                    continue

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

    x = np.random.randn(4, 5, 4, 4, 5, 4, 4)

    # try all possible vmap pairs!
    for in_axis1 in range(7):
        for in_axis2 in range(6):
            # since stan is expensive, only do 10% of cases
            if inference == inference_stan:
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
            if inference == inference_stan:
                if hash((in_axis1, in_axis2)) % 10 != 0:
                    continue

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


###############################################################################
# Test basic deterministic functions
###############################################################################


def test_exp(inference):
    x_numpy = np.random.randn(5, 3)
    x = makerv(x_numpy)
    y = vmap(vmap(exp))(x)
    expected = np.exp(x_numpy)
    assert_one_sample_close(inference, y, expected)


def test_abs(inference):
    x_numpy = np.random.randn(5, 3)
    x = makerv(x_numpy)
    y = vmap(vmap(abs))(x)
    expected = np.abs(x_numpy)
    assert_one_sample_close(inference, y, expected)


def test_pow(inference):
    x_numpy = np.random.rand(5, 3)
    y_numpy = np.random.rand(5, 3)
    x = makerv(x_numpy)
    y = makerv(y_numpy)
    z = vmap(vmap(lambda xi, yi: xi**yi))(x, y)
    expected = x_numpy**y_numpy
    assert_one_sample_close(inference, z, expected)


def test_sub(inference):
    x_numpy = np.random.randn(5, 3)
    y_numpy = np.random.randn(5, 3)
    x = makerv(x_numpy)
    y = makerv(y_numpy)
    z = vmap(vmap(lambda xi, yi: xi - yi))(x, y)
    expected = x_numpy - y_numpy
    assert_one_sample_close(inference, z, expected)


def test_div(inference):
    x_numpy = np.random.rand(5, 3)
    y_numpy = np.random.rand(5, 3)
    x = makerv(x_numpy)
    y = makerv(y_numpy)
    z = vmap(vmap(lambda xi, yi: xi / yi))(x, y)
    expected = x_numpy / y_numpy
    assert_one_sample_close(inference, z, expected)


def test_vec_vec_mul(inference):
    x_numpy = np.random.rand(5)
    y_numpy = np.random.rand(5)
    x = makerv(x_numpy)
    y = makerv(y_numpy)
    z = x @ y
    expected = x_numpy @ y_numpy
    assert_one_sample_close(inference, z, expected)


def test_mat_vec_mul(inference):
    x_numpy = np.random.rand(3, 5)
    y_numpy = np.random.rand(5)
    x = makerv(x_numpy)
    y = makerv(y_numpy)
    z = x @ y
    expected = x_numpy @ y_numpy
    assert_one_sample_close(inference, z, expected)


def test_vec_mat_mul(inference):
    x_numpy = np.random.rand(5)
    y_numpy = np.random.rand(5, 3)
    x = makerv(x_numpy)
    y = makerv(y_numpy)
    z = x @ y
    expected = x_numpy @ y_numpy
    assert_one_sample_close(inference, z, expected)


def test_mat_mat_mul(inference):
    x_numpy = np.random.rand(5, 4)
    y_numpy = np.random.rand(4, 3)
    x = makerv(x_numpy)
    y = makerv(y_numpy)
    z = x @ y
    expected = x_numpy @ y_numpy
    assert_one_sample_close(inference, z, expected)


def test_sum(inference):
    x_numpy = np.random.rand(5, 4)
    x = makerv(x_numpy)
    z = sum(x, axis=1)
    expected = np.sum(x_numpy, axis=1)
    assert_one_sample_close(inference, z, expected)


def test_vmap_sum1(inference):
    x_numpy = np.random.rand(5, 4, 6)
    x = makerv(x_numpy)
    z = vmap(lambda xi: sum(xi, axis=1))(x)
    expected = jax.vmap(lambda xi: jnp.sum(xi, axis=1))(x_numpy)
    assert_one_sample_close(inference, z, expected)


def test_vmap_sum2(inference):
    x_numpy = np.random.rand(5, 4, 6)
    x = makerv(x_numpy)
    z = vmap(lambda xi: sum(xi, axis=0), in_axes=2)(x)
    expected = jax.vmap(lambda xi: jnp.sum(xi, axis=0), in_axes=2)(x_numpy)
    assert_one_sample_close(inference, z, expected)


def test_vmap_sum3(inference):
    x_numpy = np.random.rand(4, 5, 6, 7)
    x = makerv(x_numpy)
    z = vmap(lambda xi: sum(sum(xi, axis=1), axis=0), in_axes=2)(x)
    expected = jax.vmap(lambda xi: jnp.sum(jnp.sum(xi, axis=1), axis=0), in_axes=2)(
        x_numpy
    )
    assert_one_sample_close(inference, z, expected)
