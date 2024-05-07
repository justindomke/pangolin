from pangolin.interface import *

from pangolin import inference_numpyro, inference_numpyro_modelbased, calculate
from scipy import stats
from jax import numpy as jnp

calc = calculate.Calculate(inference_numpyro_modelbased)


def test_basic():
    x = normal(0, 1)
    print(f"{calc.E(x)=}")


def test_mixture_dist_construction():
    mixing_dist = bernoulli
    num_mixing_args = 1
    component_dist = VMapDist(bernoulli, [0])
    dist = Mixture(mixing_dist, num_mixing_args, component_dist)
    print(str(dist))
    print(repr(dist))


def test_mixture_rv_construction():
    mixing_dist = bernoulli
    num_mixing_args = 1
    component_dist = VMapDist(bernoulli, [0])
    dist = Mixture(mixing_dist, num_mixing_args, component_dist)
    mixing_w = makerv(0.2)
    component_w = makerv([0.5, 0.9])
    x = dist(mixing_w, component_w)
    assert x.shape == ()
    print_upstream(x)


def test_mixture_rv_E():
    dist = Mixture(bernoulli, 1, VMapDist(bernoulli, [0]))
    mixing_w = makerv(0.2)
    component_w = makerv([0.3, 0.85])
    x = dist(mixing_w, component_w)
    Ex = calc.E(x)

    expected = (1 - 0.2) * 0.3 + 0.2 * 0.85
    print(f"{Ex=}")
    print(f"{expected=}")
    assert np.abs(Ex - expected) < 0.01


def test_numpyro_mixture_var_log_prob():
    dist = Mixture(bernoulli, 1, VMapDist(bernoulli, [0]))
    mixing_w = makerv(0.2)
    component_w = makerv([0.3, 0.85])
    x = dist(mixing_w, component_w)
    numpyro_parents = [p.cond_dist.value for p in x.parents]
    numpyro_x = inference_numpyro_modelbased.numpyro_var(x.cond_dist, *numpyro_parents)
    l = numpyro_x.log_prob(1)
    expected = jnp.log((1 - 0.2) * 0.3 + 0.2 * 0.85)
    assert jnp.allclose(l, expected)


def test_mixture1():
    dist = Mixture(bernoulli, 1, VMapDist(normal_scale, (0, 0)))
    w = jnp.array(np.random.rand())
    locs = jnp.array(np.random.randn(2))
    scales = jnp.array(np.random.randn(2) ** 2)
    x = dist(w, locs, scales)
    E_x = calc.E(x)
    mu = (1 - w) * locs[0] + w * locs[1]
    assert jnp.abs(E_x - mu) < 0.05

    var_x = calc.var(x)
    sigma2 = (
        (1 - w) * (scales[0] ** 2 + locs[0] ** 2)
        + w * (scales[1] ** 2 + locs[1] ** 2)
        - mu**2
    )
    assert jnp.abs(var_x - sigma2) < 0.2


def test_mixture2():
    dist = Mixture(categorical, 1, VMapDist(normal_scale, (0, 0)))
    w = jnp.array(np.random.rand(3))
    w = w / jnp.sum(w)
    locs = jnp.array(np.random.randn(3))
    scales = jnp.array(np.random.randn(3) ** 2)
    x = dist(w, locs, scales)
    E_x = calc.E(x)
    mu = w @ locs
    assert jnp.abs(E_x - mu) < 0.05

    var_x = calc.var(x)
    sigma2 = w @ (scales**2 + locs**2) - mu**2
    print(f"{var_x=}")
    print(f"{sigma2=}")
    assert jnp.abs(var_x - sigma2) < 0.3


def test_functional_mixture1():
    x = mixture(bernoulli(0.2), lambda z: normal(z, 1))
    print(f"{x=}")
    print_upstream(x)
    print(f"{calc.E(x)=}")
    assert jnp.abs(calc.E(x) - 0.2) < 0.05
    print_upstream(x)


def test_functional_mixture2():
    w0 = jnp.array(np.random.rand(3))
    w0 = w0 / jnp.sum(w0)
    locs0 = jnp.array(np.random.randn(3))
    scales0 = jnp.array(np.random.randn(3) ** 2)
    w = makerv(w0)
    locs = makerv(locs0)
    scales = makerv(scales0)
    x = mixture(categorical(w), lambda i: normal(locs[i], scales[i]))
    print_upstream(x)

    E_x = calc.E(x)
    mu = w0 @ locs0
    assert jnp.abs(E_x - mu) < 0.05

    var_x = calc.var(x)
    sigma2 = w0 @ (scales0**2 + locs0**2) - mu**2
    assert jnp.abs(var_x - sigma2) < 0.3


def test_double_functional_mixture1():
    x = mixture(
        bernoulli(0.2), lambda z1: mixture(bernoulli(0.8), lambda z2: normal(z1, z2 + 1))
    )
    print_upstream(x)
    E_x = calc.E(x)
    mu = 0.2

    assert jnp.abs(E_x - mu) < 0.05


def test_double_functional_mixture2():
    dims1 = 3
    dims2 = 4
    w1 = np.random.rand(dims1)
    w1 = makerv(w1 / jnp.sum(w1))
    w2 = np.random.rand(dims2)
    w2 = makerv(w2 / jnp.sum(w2))
    locs = makerv(np.random.randn(dims1, dims2))
    scales = makerv(np.random.rand(dims1, dims2))
    x = mixture(
        categorical(w1),
        lambda i: mixture(categorical(w2), lambda j: normal(locs[i, j], scales[i, j])),
    )
    print_upstream(x)
    E_x = calc.E(x)
    p = jnp.outer(w1.cond_dist.value, w2.cond_dist.value)
    mu = jnp.sum(p * locs.cond_dist.value)
    print(f"{E_x=}")
    print(f"{mu=}")
    assert jnp.abs(E_x - mu) < 0.05

    var_x = calc.var(x)
    sigma2 = (
        jnp.sum(p * (locs.cond_dist.value**2 + scales.cond_dist.value**2)) - mu**2
    )
    print(f"{var_x=}")
    print(f"{sigma2=}")
    assert jnp.abs(var_x - sigma2) < 0.05


# def test_mixture2():
#     weights = makerv([0.5, 0.3, 0.2])
#     params = makerv([0.1, 0.5, 0.9])
#     x = Mixture(bernoulli, [0])(weights, params)
#     assert x.shape == ()
#     print(f"{x=}")
#
#
# def test_mixture3():
#     weights = makerv([0.5, 0.3, 0.2])
#     params = makerv(0.5)
#     x = Mixture(bernoulli, [None], 3)(weights, params)
#     assert x.shape == ()
#     print(f"{x=}")
#
#
# def test_mixture4():
#     weights = makerv([0.5, 0.3, 0.2])
#     locs = makerv([0.1, 0.5, 0.9])
#     scales = makerv([2.0, 3.0, 4.0])
#     x = Mixture(normal_scale, [0, 0])(weights, locs, scales)
#     assert x.shape == ()
#
#
# def test_mixture5():
#     weights = makerv([0.5, 0.3, 0.2])
#     locs = makerv([0.1, 0.5, 0.9])
#     scales = makerv(2.0)
#     x = Mixture(normal_scale, [0, None])(weights, locs, scales)
#     assert x.shape == ()
#
#
# def test_mixture6():
#     weights = makerv([0.5, 0.3, 0.2])
#     locs = makerv(0.1)
#     scales = makerv([2.0, 3.0, 4.0])
#     x = Mixture(normal_scale, [None, 0])(weights, locs, scales)
#     assert x.shape == ()
#
#
# def test_mixture7():
#     weights = makerv([0.5, 0.3, 0.2])
#     locs = makerv([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
#     cov = makerv(np.eye(2))
#     x = Mixture(multi_normal_cov, [0, None])(weights, locs, cov)
#     assert x.shape == (2,)
#
#
# def test_mixture8():
#     weights = makerv([0.5, 0.3, 0.2])
#     locs = makerv([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
#     cov = makerv(np.eye(2))
#     x = Mixture(multi_normal_cov, [0, None], 3)(weights, locs, cov)
#     assert x.shape == (2,)
