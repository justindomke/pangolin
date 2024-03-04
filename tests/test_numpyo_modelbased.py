import pangolin.interface as interface
import numpy as np
from numpyro import distributions as dist
from pangolin.interface import makerv
import pangolin.inference_numpyro_modelbased as inf
import jax.random
import numpyro
from jax import numpy as jnp


def test_normal():
    loc = makerv(0)
    scale = makerv(1)
    x = interface.normal(loc, scale)

    model, names = inf.get_model_flat([x], [], [])
    assert names[loc] == "v0"
    assert names[scale] == "v1"
    assert names[x] == "v2"
    nuts_kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=500, num_samples=500)
    key = jax.random.PRNGKey(0)
    mcmc.run(key)
    # mcmc.print_summary()


def test_normal_add():
    loc = makerv(0)
    scale = makerv(1)
    x = interface.normal(loc, scale)
    y = interface.normal(loc, scale)
    z = x + y
    model, names = inf.get_model_flat([z], [], [])
    nuts_kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=10000, num_samples=10000)
    key = jax.random.PRNGKey(0)
    mcmc.run(key)
    mcmc.print_summary(exclude_deterministic=False)

    # wierd trick to get samples for deterministic sites
    latent_samples = mcmc.get_samples()
    predictive = numpyro.infer.Predictive(model, latent_samples)
    predictive_samples = predictive(key)

    samples = {**latent_samples, **predictive_samples}  # merge

    xs = samples[names[x]]
    ys = samples[names[y]]
    zs = samples[names[z]]

    assert np.abs(np.mean(xs) - 0) < 0.05
    assert np.abs(np.var(xs) - 1) < 0.05

    assert np.abs(np.mean(ys) - 0) < 0.05
    assert np.abs(np.var(ys) - 1) < 0.05

    assert np.abs(np.mean(zs) - 0) < 0.05
    assert np.abs(np.var(zs) - 2) < 0.05


def test_diag_normal():
    def model():
        x = numpyro.sample(
            "x", inf.DiagNormal(jnp.array([0, 1, 2]), jnp.array([3, 4, 5]))
        )

    nuts_kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=100, num_samples=100)
    key = jax.random.PRNGKey(0)
    mcmc.run(key)
    mcmc.print_summary()


# def test_vmap():
#     loc = makerv(0)
#     scale = makerv(1)
#     x = interface.vmap(interface.normal_scale, None, axis_size=3)(loc, scale)
#     model, names = inf.get_model_flat([x], [], [])
#     nuts_kernel = numpyro.infer.NUTS(model)
#     mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=3, num_samples=3)
#     key = jax.random.PRNGKey(0)
#     mcmc.run(key)
#     mcmc.print_summary(exclude_deterministic=False)


# def test_binomial1():
#     n = 12
#     p = 0.3
#     val = 5
#
#     x = interface.binomial(n, p)
#
#     l = inf.log_prob(x.cond_dist, val, n, p)
#
#     expected = dist.Binomial(n, p).log_prob(val)
#
#     assert np.allclose(l, expected)
#
#
# def test_mul():
#     d = interface.mul
#
#     l = inf.evaluate(d, 2, 4)
#
#     assert np.allclose(l, 8)
#
#
# def test_constant():
#     d = interface.Constant(1.1)
#
#     l = inf.evaluate(d)
#
#     assert np.allclose(l, 1.1)
#
#
# def test_sum():
#     d = interface.Sum(axis=1)
#     arr = np.random.randn(3, 2)
#
#     l = inf.evaluate(d, arr)
#     expected = np.sum(arr, axis=1)
#
#     assert np.allclose(l, expected)
#
#
# def test_index1():
#     d = interface.Index(None)
#     arr = np.random.randn(3)
#
#     l = inf.evaluate(d, arr, 1)
#     expected = arr[1]
#
#     assert np.allclose(l, expected)
#
#
# def test_index2():
#     d = interface.Index(slice(None))
#     arr = np.random.randn(3)
#
#     l = inf.evaluate(d, arr)
#     expected = arr[:]
#
#     assert np.allclose(l, expected)
#
#
# def test_index3():
#     d = interface.Index(None, slice(None))
#     arr = np.random.randn(7, 5)
#
#     l = inf.evaluate(d, arr, [1, 3])
#     expected = arr[[1, 3], :]
#
#     assert np.allclose(l, expected)
#
#
# def test_deterministic_vmap1():
#     d = interface.VMapDist(interface.mul, (0, 0))
#     a = np.random.randn(5)
#     b = np.random.randn(5)
#
#     l = inf.evaluate(d, a, b)
#     expected = a * b
#
#     assert np.allclose(l, expected)
#
#
# def test_deterministic_vmap2():
#     d = interface.VMapDist(interface.mul, (None, 0))
#     a = np.random.randn()
#     b = np.random.randn(5)
#
#     l = inf.evaluate(d, a, b)
#     expected = a * b
#
#     assert np.allclose(l, expected)
#
#
# def test_sample_vmap1():
#     d = interface.VMapDist(interface.normal_scale, (0, None), 5)
#     a = np.random.randn(5)
#     b = np.random.randn() * 1e-12
#
#     key = jax.random.PRNGKey(0)
#     l = inf.sample(d, key, a, b)
#
#     assert np.allclose(l, a)
#
#
# def test_log_prob_vmap1():
#     d = interface.VMapDist(interface.normal_scale, (0, 0), 5)
#     a = np.random.randn(5)
#     b = np.random.rand(5)
#     val = np.random.randn(5)
#
#     l = inf.log_prob(d, val, a, b)
#
#     expected = sum(
#         [dist.Normal(a_i, b_i).log_prob(val_i) for a_i, b_i, val_i in zip(a, b, val)]
#     )
#     print(f"{l=}")
#     print(f"{expected=}")
#
#     assert np.allclose(l, expected)
#
#
# def test_ancestor_ops_flat1():
#     key = jax.random.PRNGKey(0)
#
#     x = interface.normal(0, 1)
#     y = interface.normal(x, 1)
#
#     x_val, y_val = inf.ancestor_sample_flat_key(key, [x, y], [], [])
#
#     assert x_val.shape == ()
#     assert y_val.shape == ()
#
#     l = inf.ancestor_log_prob_flat([x, y], [x_val, y_val], [], [])
#
#     expected_l = dist.Normal(0, 1).log_prob(x_val) + dist.Normal(x_val, 1).log_prob(
#         y_val
#     )
#
#     assert np.allclose(l, expected_l)
#
#
# def test_sampling1():
#     x = interface.normal(0, 1)
#     y = interface.normal(x, 1)
#
#     # npinf = inf.NumpyroInference(niter=1000)
#
#     xs, ys = inf.sample_flat([x, y], [], [], niter=1000)
#
#     assert xs.shape == (1000,)
#     assert ys.shape == (1000,)
#
#
# def test_sampling2():
#     x = interface.normal(0, 1)
#     y = interface.normal(x, 1)
#
#     # npinf = inf.NumpyroInference(niter=1000)
#
#     (xs,) = inf.sample_flat([x], [y], [np.array(2)], niter=1000)
#
#     assert xs.shape == (1000,)
#
#     assert abs(np.mean(xs) - 1.0) < 0.1
