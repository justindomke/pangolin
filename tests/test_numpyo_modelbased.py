import pangolin.interface as interface
import numpy as np
from numpyro import distributions as dist
from pangolin.interface import makerv
import pangolin.inference_numpyro_modelbased as inf
import jax.random
import numpyro
from jax import numpy as jnp
from jax.scipy import stats


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


def test_normal_normal():
    loc = makerv(0)
    scale = makerv(1)
    z = interface.normal(loc, scale)
    two = makerv(2)
    z_squared = z**two
    x = interface.normal(z, z_squared)
    model, names = inf.get_model_flat([z, x], [], [])
    assert names[loc] == "v0"
    assert names[scale] == "v1"
    assert names[z] == "v2"
    assert names[two] == "v3"
    assert names[z_squared] == "v4"
    assert names[x] == "v5"

    print(f"{model=}")
    print(f"{names=}")

    nuts_kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=100, num_samples=100)
    key = jax.random.PRNGKey(0)
    mcmc.run(key)
    mcmc.print_summary(exclude_deterministic=False)


def test_vmapdist():
    cond_dist = interface.VMapDist(interface.normal_scale, (None, 0))
    # dist_class = inf.get_numpyro_vmapdist(cond_dist)
    # dist = dist_class(np.array(0.5), np.array([1, 2, 3]))
    dist = inf.get_numpyro_vmapdist(cond_dist, np.array(0.5), np.array([1, 2, 3]))

    key = jax.random.PRNGKey(0)
    x = dist.sample(key)
    assert x.shape == (3,)

    keys = jax.random.split(key, 3)
    expected = jax.vmap(jax.random.normal)(keys) * jnp.array([1, 2, 3]) + 0.5
    assert np.allclose(x, expected)

    l = dist.log_prob(x)
    ls = stats.norm.logpdf(x, 0.5, jnp.array([1, 2, 3]))
    expected = jnp.sum(ls)
    assert np.allclose(l, expected)


def test_double_vmapdist():
    cond_dist0 = interface.VMapDist(interface.normal_scale, (None, 0))
    cond_dist = interface.VMapDist(cond_dist0, (None, 0))
    # dist_class = inf.get_numpyro_vmapdist(cond_dist)
    # dist = dist_class(np.array(0.5), np.array([[1, 2, 3], [4, 5, 6]]))
    dist = inf.get_numpyro_vmapdist(
        cond_dist, np.array(0.5), np.array([[1, 2, 3], [4, 5, 6]])
    )

    key = jax.random.PRNGKey(0)
    x = dist.sample(key)
    assert x.shape == (2, 3)

    keys = jax.vmap(jax.random.split, [0, None])(jax.random.split(key, 2), 3)
    expected = (
        jax.vmap(jax.vmap(jax.random.normal))(keys) * jnp.array([[1, 2, 3], [4, 5, 6]])
        + 0.5
    )
    assert np.allclose(x, expected)

    l = dist.log_prob(x)
    ls = stats.norm.logpdf(x, 0.5, jnp.array(jnp.array([[1, 2, 3], [4, 5, 6]])))
    expected = jnp.sum(ls)
    assert np.allclose(l, expected)


def test_mcmc_vmapdist():
    cond_dist = interface.VMapDist(interface.normal_scale, (None, None), axis_size=3)
    dist_class = inf.get_numpyro_vmapdist(cond_dist)

    def model():
        loc = numpyro.sample("loc", dist.Normal(0, 1))
        scale = numpyro.sample("scale", dist.Exponential(1))
        x = numpyro.sample("x", dist_class(loc, scale))

    nuts_kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=10000, num_samples=10000)
    key = jax.random.PRNGKey(0)
    mcmc.run(key)
    mcmc.print_summary(exclude_deterministic=False)


def test_mcmc_vmap_exponential():
    cond_dist = interface.VMapDist(interface.exponential, (None,), axis_size=3)
    dist_class = inf.get_numpyro_vmapdist(cond_dist)

    def model():
        x = numpyro.sample("x", dist_class(np.array(1)))

    nuts_kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=10000, num_samples=10000)
    key = jax.random.PRNGKey(0)
    mcmc.run(key)
    mcmc.print_summary(exclude_deterministic=False)


def test_mcmc_vmap_dirichlet1():
    cond_dist = interface.VMapDist(interface.dirichlet, (None,), axis_size=3)
    dist_class = inf.get_numpyro_vmapdist(cond_dist)

    def model():
        x = numpyro.sample("x", dist_class(np.array([1.2, 5.0, 2.0])))

    nuts_kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=10000, num_samples=10000)
    key = jax.random.PRNGKey(0)
    mcmc.run(key)
    mcmc.print_summary(exclude_deterministic=False)


def test_mcmc_vmap_dirichlet2():
    cond_dist = interface.VMapDist(interface.dirichlet, (0,), axis_size=3)
    dist_class = inf.get_numpyro_vmapdist(cond_dist)

    def model():
        x = numpyro.sample(
            "x", dist_class(np.array([[1.2, 5.0, 2.0], [3, 0.1, 10], [3, 5, 7]]))
        )

    nuts_kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=10000, num_samples=10000)
    key = jax.random.PRNGKey(0)
    mcmc.run(key)
    mcmc.print_summary(exclude_deterministic=False)


def test_sample_flat():
    x = interface.normal(0.5, 2.0)
    [xs] = inf.sample_flat([x], [], [])
    assert abs(np.mean(xs) - 0.5) < 0.1
    assert abs(np.std(xs) - 2.0) < 0.1


def test_vmap_sample_flat():
    loc = np.array([0.5, 1.1, 1.5])
    scale = np.array([2.2, 3.3, 4.4])
    x = interface.vmap(interface.normal, (0, 0), 3)(loc, scale)
    [xs] = inf.sample_flat([x], [], [])
    assert xs.shape == (10000, 3)
    assert max(abs(np.mean(xs, axis=0) - loc)) < 0.1
    assert max(abs(np.std(xs, axis=0) - scale)) < 0.1


def test_deterministic_vmap():
    loc = np.array([0.5, 1.1, 1.5])
    scale = np.array([2.2, 3.3, 4.4])
    x = interface.vmap(
        lambda loc_i, scale_i: interface.exp(interface.normal(loc_i, scale_i)), 0
    )(loc, scale)
    [xs] = inf.sample_flat([x], [], [])
    assert xs.shape == (10000, 3)


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
