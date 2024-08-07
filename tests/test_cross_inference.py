import pytest
from pangolin import (
    inference_jags,
    inference_numpyro,
    inference_stan,
    inference_numpyro_modelbased,
)
import numpy as np

# from pangolin.interface import makerv, vmap, plate
# from pangolin.interface import normal, beta, binomial
import jax
from pangolin.interface import *

inference_engines = [
    inference_jags,
    inference_numpyro,
    inference_stan,
    inference_numpyro_modelbased,
]


def all_pairs_close(all_means, atol, rtol):
    for means1 in all_means:
        for means2 in all_means:
            for m1, m2 in zip(means1, means2):
                if not np.allclose(m1, m2, atol=atol, rtol=rtol):
                    return False
    return True


def assert_means_close_adaptive(
    vars, given, vals, excluded_engines=(), atol=1e-5, rtol=1e-5
):
    engines = list(set(inference_engines) - set(excluded_engines))
    assert len(engines) > 1
    vals = [np.array(val) for val in vals]
    niter = 100
    while True:
        if niter > 1e6:
            assert False, f"failed with {niter=}"

        all_means = []
        for inference in engines:
            print(f"{inference=}")
            samples = inference.sample_flat(vars, given, vals, niter=niter)
            means = [np.mean(s, axis=0) for s in samples]
            all_means.append(means)
        print(f"{niter=} {all_means=}")

        if all_pairs_close(all_means, atol=atol, rtol=rtol):
            return
        else:
            niter *= 2


def assert_means_close(
    vars, given_vars, given_vals, niter=10_000, excluded_engines=(), atol=1e-5, rtol=1e-5
):
    all_means = []
    engines = list(set(inference_engines) - set(excluded_engines))
    for inference in engines:
        samples = inference.sample_flat(vars, given_vars, given_vals, niter=niter)
        means = [np.mean(s, axis=0) for s in samples]
        all_means.append(means)
    print(f"{engines=} {all_means=}")
    for means1 in all_means:
        for means2 in all_means:
            for m1, m2 in zip(means1, means2):
                assert np.allclose(m1, m2, atol=atol, rtol=rtol)
                # assert np.max(np.abs(m1 - m2)) < 0.01


def test_abs():
    z = makerv(-1.1)
    x = abs(z)
    assert_means_close([x], [], [], niter=1)


def test_vec_abs():
    z = makerv(np.random.randn(10))
    x = vmap(abs)(z)
    assert_means_close([x], [], [], niter=1)


def test_arccos():
    z = makerv(-1 + 2 * np.random.rand(10))
    x = vmap(arccos)(z)
    assert_means_close([x], [], [], niter=1)


def test_arccosh():
    z = makerv(1 / np.random.rand(10))
    x = vmap(arccosh)(z)
    assert_means_close([x], [], [], niter=1)


def test_arcsin():
    z = makerv(-1 + 2 * np.random.rand(10))
    x = vmap(arcsin)(z)
    assert_means_close([x], [], [], niter=1)


def test_arcsinh():
    z = makerv(np.random.randn(10))
    x = vmap(arcsinh)(z)
    assert_means_close([x], [], [], niter=1)


def test_arctan():
    z = makerv(np.random.randn(10))
    x = vmap(arctan)(z)
    assert_means_close([x], [], [], niter=1)


def test_arctanh():
    z = makerv(-1 + 2 * np.random.rand(10))
    x = vmap(arctanh)(z)
    assert_means_close([x], [], [], niter=1)


def test_cos():
    z = makerv(np.random.randn(10))
    x = vmap(cos)(z)
    assert_means_close([x], [], [], niter=1)


def test_cosh():
    z = makerv(np.random.randn(10))
    x = vmap(cosh)(z)
    assert_means_close([x], [], [], niter=1)


def test_exp():
    z = makerv(np.random.randn(10))
    x = vmap(exp)(z)
    assert_means_close([x], [], [], niter=1)


def test_inv_logit():
    z = makerv(np.random.randn(10))
    x = vmap(inv_logit)(z)
    assert_means_close([x], [], [], niter=1)


def test_log():
    z = makerv(1 / np.random.rand(10))
    x = vmap(log)(z)
    assert_means_close([x], [], [], niter=1)


def test_loggamma():
    z = makerv(1 / np.random.rand(10))
    x = vmap(loggamma)(z)
    assert_means_close([x], [], [], niter=1)


def test_logit():
    z = makerv(np.random.rand(10))
    x = vmap(logit)(z)
    assert_means_close([x], [], [], niter=1)


def test_sin():
    z = makerv(np.random.randn(10))
    x = vmap(sin)(z)
    assert_means_close([x], [], [], niter=1)


def test_sinh():
    z = makerv(np.random.randn(10))
    x = vmap(sinh)(z)
    assert_means_close([x], [], [], niter=1)


def test_step():
    z = makerv(np.random.randn(10))
    x = vmap(step)(z)
    assert_means_close([x], [], [], niter=1)


def test_tan():
    z = makerv(np.random.randn(10))
    x = vmap(sin)(z)
    assert_means_close([x], [], [], niter=1)


def test_tanh():
    z = makerv(np.random.randn(10))
    x = vmap(sinh)(z)
    assert_means_close([x], [], [], niter=1)


def test_inverse_logit():
    z = makerv([1.1, 2.2, 3.3])
    x = vmap(inv_logit)(z)
    assert_means_close([x], [], [], niter=1)


def test_inv():
    val = np.random.randn(20, 7, 7)
    val = np.einsum("ijk,ilk->ijl", val, val)  # make it pos-def
    val = val + 1e-3 * np.stack([np.eye(7)] * 20)  # make it strictly pos-def
    for i in range(val.shape[0]):  # check all slices are pos-def
        assert np.all(np.linalg.eigvalsh(val[i]) > 0)
    z = makerv(val)
    x = vmap(inv)(z)
    assert_means_close([x], [], [], niter=1, atol=1e-3, rtol=1e-3)


def test_softmax():
    val = np.random.randn(7)
    z = makerv(val)
    x = softmax(z)
    assert_means_close([x], [], [], niter=1, excluded_engines=[inference_jags])


def test_normal_scale():
    z = normal_scale(0.5, 1.5)
    assert_means_close_adaptive([z], [], [], atol=0.05, rtol=0.05)


# def test_normal_prec():
#     z = normal_prec(0.5, 1.5)
#     assert_means_close_adaptive([z], [], [], atol=0.05, rtol=0.05)


def test_uniform():
    z = uniform(0.7, 2.9)
    # exclude stan because stan doesn't understand constraints very well
    assert_means_close_adaptive(
        [z], [], [], atol=0.05, rtol=0.05, excluded_engines=[inference_stan]
    )


def test_bernoulli():
    # z = uniform(makerv(0.0), makerv(1.0))
    z = normal_scale(0, 1)
    x = bernoulli(inv_logit(z))
    assert_means_close_adaptive(
        [z],
        [x],
        [np.array(0)],
        atol=0.1,
        rtol=0.1,
        excluded_engines=[
            inference_numpyro,
        ],
    )


def test_bernoulli_logit():
    z = normal_scale(0, 1)
    x = bernoulli_logit(z)
    assert_means_close_adaptive([z], [x], [0], atol=0.05, rtol=0.05)


def test_binomial():
    z = uniform(0, 1)
    x = binomial(20, z)
    assert_means_close_adaptive(
        [z], [x], [10], atol=0.05, rtol=0.05, excluded_engines=[inference_stan]
    )


def test_beta():
    z = beta(1.9, 2.9)
    assert_means_close_adaptive([z], [], [], atol=0.05, rtol=0.05)


def test_exponential():
    z = exponential(1.9)
    assert_means_close_adaptive([z], [], [], atol=0.05, rtol=0.05)


def test_beta_binomial():
    a = exp(normal(0, 1))
    b = exp(normal(0, 1))
    x = beta_binomial(17, a, b)
    assert_means_close_adaptive(
        [a, b],
        [x],
        [6],
        atol=0.05,
        rtol=0.05,
        excluded_engines=[inference_jags],
    )


# def test_log_prob():
#     loc = makerv(0)
#     scale = makerv(1)
#     x = normal(loc, scale)
#     l = log_prob(x, [loc, scale])
#     assert_means_close(
#         [l], [x], [np.array(0.0)], niter=1, excluded_engines=[inference_stan]
#     )


# def test_double_normal():
#     z = normal(1.7, 2.9)
#     x = normal(z, 0.1)
#     x_val = np.array(-1.2)
#     assert_means_close([z], [x], [x_val])
#
#
# def test_beta_binomial():
#     z = beta(1.3, 2.4)
#     x = binomial(10, z)
#     x_val = np.array(5)
#     assert_means_close([z], [x], [x_val])


# cos
# cosh
# sin
# sinh
# tan
# tanh
