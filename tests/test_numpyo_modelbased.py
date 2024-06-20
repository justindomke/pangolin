from pangolin.interface import *
import pangolin.interface as interface
import numpy as np
from numpyro import distributions as dist
from pangolin.interface import makerv
import pangolin.inference_numpyro_modelbased as inf
import jax.random
import numpyro
from jax import numpy as jnp
from jax.scipy import stats

def test_autoregressive_exponential_sampling():
    op = interface.Autoregressive(interface.exponential, length=5)
    z = op(makerv(2.0))
    [z_samp] = inf.ancestor_sample_flat([z],niter=1000)
    print(f"{z_samp=}")

def test_autoregressive_exponential_log_prob():
    op = interface.Autoregressive(interface.exponential, length=5)
    z = op(makerv(1.0))
    z_samp = np.arange(2.0,7.0)
    l = inf.numpyro_var(z.cond_dist, 1.0).log_prob(z_samp)
    expected = 0.0
    last = 1.0
    for zi in z_samp:
        expected += stats.expon.logpdf(zi,0,1/last) # loc/scale parameterization
        last = zi
    assert np.allclose(l,expected)

def test_autoregressive_normal_sampling():
    op = interface.Autoregressive(interface.normal_scale, 5)
    z = op(makerv(0.0),makerv(np.ones(5)))
    [z_samp] = inf.ancestor_sample_flat([z],niter=5000)
    z_std = np.std(z_samp,axis=0)
    expected_z_std = np.sqrt(np.arange(1,6)) # random walk variance is additive
    assert np.all(np.abs(z_std-expected_z_std)<.1)

def test_autoregressive_normal_log_prob():
    op = interface.Autoregressive(interface.normal_scale, 5)
    scales = np.arange(5)/2 + 1.5
    z = op(makerv(0.25),makerv(scales))
    [z_samp] = inf.ancestor_sample_flat([z])
    l = inf.numpyro_var(z.cond_dist, 0.25, scales).log_prob(z_samp)
    expected = 0.0
    last = 0.25
    for zi, scale in zip(z_samp, scales):
        expected += stats.norm.logpdf(zi,last,scale)
        last = zi
    assert np.allclose(l, expected)


def test_autoregressive_nonrandom1():
    op = interface.Autoregressive(interface.sin, 5)
    z = op(makerv(0.5))
    z_samp = inf.ancestor_sample_flat([z])
    expected = []
    y = 0.5
    for i in range(5):
        y = np.sin(y)
        expected.append(y)
    expected = np.array(expected)
    assert np.allclose(z_samp,expected)

def test_autoregressive_cumsum1():
    op = interface.Autoregressive(interface.add, 5)
    z = op(makerv(0),makerv([1,2,3,4,5]))
    z_samp = inf.ancestor_sample_flat([z])
    expected = np.cumsum([1,2,3,4,5])
    assert np.allclose(z_samp,expected)

def test_autoregressive_cumdiv1():
    op = interface.Autoregressive(interface.div, 5)
    z = op(makerv(1.0),makerv([1,2,3,4,5]))
    z_samp = inf.ancestor_sample_flat([z])
    expected = []
    y = 1.0
    for i in range(1,6):
        y = y/i
        expected.append(y)
    expected = np.array(expected)
    print(f"{z_samp=}")
    print(f"{expected=}")
    assert np.allclose(z_samp,expected)

def test_composite1():
    op = interface.Composite(2, [interface.add], [[0,1]])
    l = inf.numpyro_var(op, 1.1, 2.2)
    expected = 3.3
    assert np.allclose(l, expected)

def test_composite2():
    op = interface.Composite(1, [interface.add, interface.mul],[[0,0],[0,1]])
    l = inf.numpyro_var(op, 2.2)
    expected = 4.4*2.2
    assert np.allclose(l, expected)

def test_composite3():
    op = interface.Composite(1,[interface.mul,interface.normal_scale],[[0,0],[0,1]])
    x = makerv(2.2)
    z = op(x)
    [zs] = inf.sample_flat([z], [], [], niter=20000)
    assert jnp.abs(jnp.mean(zs) - 2.2) < 0.1
    assert jnp.abs(jnp.std(zs) - 2.2**2) < 0.1

def test_autoregressive_composite():
    op0 = interface.Composite(1, [interface.mul, interface.normal_scale], [[0, 0], [0, 1]])
    op = interface.Autoregressive(op0, 0, 5)
    x = interface.normal_scale(0,1)
    z = op(x)
    [z_samp] = inf.ancestor_sample_flat([z])
    print(f"{z_samp=}")
    print(f"{z_samp.shape=}")

def test_autoregressive_exponential1():
    x = exponential(1)
    z = autoregressive(lambda last: exponential(last), length=100)(x)
    [z_samp] = inf.ancestor_sample_flat([z])
    assert z_samp.shape == (100,)


def test_autoregressive_normal1():
    x = normal_scale(0,1)
    scales = makerv(np.arange(100))
    z = autoregressive(lambda last: normal_scale(last,1), 100)(x)
    [z_samp] = inf.ancestor_sample_flat([z])
    assert z_samp.shape == (100,)

def test_autoregressive_normal2():
    x = normal_scale(0,1)
    scales = makerv(np.arange(100))
    z = autoregressive(lambda last, scale: normal_scale(last, scale))(x, scales)
    [z_samp] = inf.ancestor_sample_flat([z])
    assert z_samp.shape == (100,)

def test_autoregressive_normal3():
    x = normal_scale(0,1)
    locs = makerv(np.arange(100))
    z = autoregressive(lambda last, loc: normal_scale(loc,.9*abs(last) + .1))(x, locs)
    [z_samp] = inf.ancestor_sample_flat([z])
    print(z_samp)
    assert z_samp.shape == (100,)

def test_autoregressive_closure_exponential():
    a = interface.exponential(1)
    x = interface.exponential(1)
    z = autoregressive(lambda last: exponential(a), 10)(x)
    assert z.shape == (10,)

def test_autoregressive_closure_normal1():
    scale = interface.exponential(1)
    x = normal_scale(0,1)
    z = autoregressive(lambda last: normal_scale(last, scale), 10)(x)

def test_autoregressive_closure_normal2():
    x = normal_scale(0,1)
    z = autoregressive(lambda last: normal_scale(last, 1), 10)(x)

def test_autoregressive_closure_normal():
    loc = interface.normal_scale(0,1)
    scale = interface.exponential(1)
    x = normal_scale(0,1)
    z = autoregressive(lambda last: normal_scale(loc+last, scale), 10)(x)


# def test_autoregressive_decorator():
#     x = interface.normal_scale(0, 1)
#     locs = makerv(np.arange(100))
#     # this is pretty horrible and wouldn't work with a length argument
#     @autoregressive
#     def f(last, loc):
#         return normal_scale(loc, 0.9*abs(last) + 0.1)
#     z = f(x,locs)
#     [z_samp] = inf.ancestor_sample_flat([z])
#     assert z_samp.shape == (100,)


# def test_timeseries():
#     # true params
#     a_true = 0.7
#     b_true = 2.5
#     # generate dataset
#     x_obs = []
#     tmp = 0.0
#     for i in range(100):
#         loc = a_true * tmp
#         scale = b_true
#         tmp = loc + scale*np.random.randn()
#         x_obs.append(tmp)
#     x_obs = np.array(x_obs)
#
#     a = normal(0,1)
#     b = normal(0,1)
#     x = autoregressive(lambda last: normal_scale(a*last, b), 100)

# def test_numpyro_var1():
#     var = inf.numpyro_var(interface.normal_scale, 0, 1)
#     assert isinstance(var, dist.Normal)
#     assert var.loc == 0
#     assert var.scale == 1
#
#
# def test_binomial1():
#     n = 12
#     p = 0.3
#     val = 5
#     x = interface.binomial(n, p)
#     l = inf.numpyro_var(x.cond_dist, n, p).log_prob(val)
#     expected = dist.Binomial(n, p).log_prob(val)
#     assert np.allclose(l, expected)
#
#
# def test_mul():
#     d = interface.mul
#     l = inf.numpyro_var(d, 2, 4)
#     assert np.allclose(l, 8)
#
#
# def test_constant():
#     d = interface.Constant(1.1)
#     l = inf.numpyro_var(d)
#     assert np.allclose(l, 1.1)
#
#
# def test_sum():
#     d = interface.Sum(axis=1)
#     arr = np.random.randn(3, 2)
#     l = inf.numpyro_var(d, arr)
#     expected = np.sum(arr, axis=1)
#     assert np.allclose(l, expected)
#
#
# def test_index1():
#     d = interface.Index(None)
#     arr = np.random.randn(3)
#     l = inf.numpyro_var(d, arr, 1)
#     expected = arr[1]
#     assert np.allclose(l, expected)
#
#
# def test_index2():
#     d = interface.Index(slice(None))
#     arr = np.random.randn(3)
#     l = inf.numpyro_var(d, arr)
#     expected = arr[:]
#     assert np.allclose(l, expected)
#
#
# def test_index3():
#     d = interface.Index(None, slice(None))
#     arr = np.random.randn(7, 5)
#     l = inf.numpyro_var(d, arr, [1, 3])
#     expected = arr[[1, 3], :]
#     assert np.allclose(l, expected)
#
#
# def test_deterministic_vmap1():
#     d = interface.VMapDist(interface.mul, (0, 0))
#     a = np.random.randn(5)
#     b = np.random.randn(5)
#
#     l = inf.numpyro_var(d, a, b)
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
#     l = inf.numpyro_var(d, a, b)
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
#     l = inf.numpyro_var(d, a, b).sample(key)
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
#     l = inf.numpyro_var(d, a, b).log_prob(val)
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
#     x = interface.normal(0, 1)
#     y = interface.normal(x, 1)
#
#     x_val, y_val = inf.ancestor_sample_flat([x, y])
#
#     assert x_val.shape == ()
#     assert y_val.shape == ()
#
#     # l = inf.ancestor_log_prob_flat([x, y], [x_val, y_val], [], [])
#     #
#     # expected_l = dist.Normal(0, 1).log_prob(x_val) + dist.Normal(x_val, 1).log_prob(y_val)
#     #
#     # assert np.allclose(l, expected_l)
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
#
#
# def test_normal():
#     loc = makerv(0)
#     scale = makerv(1)
#     x = interface.normal(loc, scale)
#
#     model, names = inf.get_model_flat([x], [], [])
#     assert names[loc] == "v0"
#     assert names[scale] == "v1"
#     assert names[x] == "v2"
#     nuts_kernel = numpyro.infer.NUTS(model)
#     mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=500, num_samples=500)
#     key = jax.random.PRNGKey(0)
#     mcmc.run(key)
#     # mcmc.print_summary()
#
#
# def test_normal_add():
#     loc = makerv(0)
#     scale = makerv(1)
#     x = interface.normal(loc, scale)
#     y = interface.normal(loc, scale)
#     z = x + y
#     model, names = inf.get_model_flat([z], [], [])
#     nuts_kernel = numpyro.infer.NUTS(model)
#     mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=10000, num_samples=10000)
#     key = jax.random.PRNGKey(0)
#     mcmc.run(key)
#     mcmc.print_summary(exclude_deterministic=False)
#
#     # wierd trick to get samples for deterministic sites
#     latent_samples = mcmc.get_samples()
#     predictive = numpyro.infer.Predictive(model, latent_samples)
#     predictive_samples = predictive(key)
#
#     samples = {**latent_samples, **predictive_samples}  # merge
#
#     xs = samples[names[x]]
#     ys = samples[names[y]]
#     zs = samples[names[z]]
#
#     assert np.abs(np.mean(xs) - 0) < 0.05
#     assert np.abs(np.var(xs) - 1) < 0.05
#
#     assert np.abs(np.mean(ys) - 0) < 0.05
#     assert np.abs(np.var(ys) - 1) < 0.05
#
#     assert np.abs(np.mean(zs) - 0) < 0.05
#     assert np.abs(np.var(zs) - 2) < 0.05
#
#
# def test_diag_normal():
#     def model():
#         x = numpyro.sample(
#             "x", inf.DiagNormal(jnp.array([0, 1, 2]), jnp.array([3, 4, 5]))
#         )
#
#     nuts_kernel = numpyro.infer.NUTS(model)
#     mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=100, num_samples=100)
#     key = jax.random.PRNGKey(0)
#     mcmc.run(key)
#     mcmc.print_summary()
#
#
# def test_normal_normal():
#     loc = makerv(0)
#     scale = makerv(1)
#     z = interface.normal(loc, scale)
#     two = makerv(2)
#     z_squared = z**two
#     x = interface.normal(z, z_squared)
#     model, names = inf.get_model_flat([z, x], [], [])
#     assert names[loc] == "v0"
#     assert names[scale] == "v1"
#     assert names[z] == "v2"
#     assert names[two] == "v3"
#     assert names[z_squared] == "v4"
#     assert names[x] == "v5"
#
#     print(f"{model=}")
#     print(f"{names=}")
#
#     nuts_kernel = numpyro.infer.NUTS(model)
#     mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=100, num_samples=100)
#     key = jax.random.PRNGKey(0)
#     mcmc.run(key)
#     mcmc.print_summary(exclude_deterministic=False)
#
#
# def test_vmap_var():
#     cond_dist = interface.VMapDist(interface.normal_scale, (None, 0))
#     # dist_class = inf.get_numpyro_vmapdist(cond_dist)
#     # dist = dist_class(np.array(0.5), np.array([1, 2, 3]))
#     dist = inf.numpyro_vmap_var(cond_dist, np.array(0.5), np.array([1, 2, 3]))
#
#     key = jax.random.PRNGKey(0)
#     x = dist.sample(key)
#     assert x.shape == (3,)
#
#     keys = jax.random.split(key, 3)
#     expected = jax.vmap(jax.random.normal)(keys) * jnp.array([1, 2, 3]) + 0.5
#     assert np.allclose(x, expected)
#
#     l = dist.log_prob(x)
#     ls = stats.norm.logpdf(x, 0.5, jnp.array([1, 2, 3]))
#     expected = jnp.sum(ls)
#     assert np.allclose(l, expected)
#
#
# def test_double_vmap_var():
#     cond_dist0 = interface.VMapDist(interface.normal_scale, (None, 0))
#     cond_dist = interface.VMapDist(cond_dist0, (None, 0))
#     dist = inf.numpyro_vmap_var(
#         cond_dist, np.array(0.5), np.array([[1, 2, 3], [4, 5, 6]])
#     )
#
#     key = jax.random.PRNGKey(0)
#     x = dist.sample(key)
#     assert x.shape == (2, 3)
#
#     keys = jax.vmap(jax.random.split, [0, None])(jax.random.split(key, 2), 3)
#     expected = (
#         jax.vmap(jax.vmap(jax.random.normal))(keys) * jnp.array([[1, 2, 3], [4, 5, 6]])
#         + 0.5
#     )
#     assert np.allclose(x, expected)
#
#     l = dist.log_prob(x)
#     ls = stats.norm.logpdf(x, 0.5, jnp.array(jnp.array([[1, 2, 3], [4, 5, 6]])))
#     expected = jnp.sum(ls)
#     assert np.allclose(l, expected)
#
#
# def test_mcmc_vmapdist():
#     cond_dist = interface.VMapDist(interface.normal_scale, (None, None), axis_size=3)
#
#     def model():
#         loc = numpyro.sample("loc", dist.Normal(0, 1))
#         scale = numpyro.sample("scale", dist.Exponential(1))
#         # x = numpyro.sample("x", dist_class(loc, scale))
#         x = numpyro.sample("x", inf.numpyro_vmap_var(cond_dist, loc, scale))
#
#     nuts_kernel = numpyro.infer.NUTS(model)
#     mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=10000, num_samples=10000)
#     key = jax.random.PRNGKey(0)
#     mcmc.run(key)
#     mcmc.print_summary(exclude_deterministic=False)
#
#
# def test_mcmc_vmap_exponential():
#     cond_dist = interface.VMapDist(interface.exponential, (None,), axis_size=3)
#
#     def model():
#         x = numpyro.sample("x", inf.numpyro_var(cond_dist, np.array(1)))
#
#     nuts_kernel = numpyro.infer.NUTS(model)
#     mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=10000, num_samples=10000)
#     key = jax.random.PRNGKey(0)
#     mcmc.run(key)
#     mcmc.print_summary(exclude_deterministic=False)
#
#
# def test_mcmc_vmap_dirichlet1():
#     cond_dist = interface.VMapDist(interface.dirichlet, (None,), axis_size=3)
#
#     def model():
#         x = numpyro.sample("x", inf.numpyro_var(cond_dist, np.array([1.2, 5.0, 2.0])))
#
#     nuts_kernel = numpyro.infer.NUTS(model)
#     mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=10000, num_samples=10000)
#     key = jax.random.PRNGKey(0)
#     mcmc.run(key)
#     mcmc.print_summary(exclude_deterministic=False)
#
#
# def test_mcmc_vmap_dirichlet2():
#     cond_dist = interface.VMapDist(interface.dirichlet, (0,), axis_size=3)
#
#     def model():
#         x = numpyro.sample(
#             "x",
#             inf.numpyro_var(
#                 cond_dist, np.array([[1.2, 5.0, 2.0], [3, 0.1, 10], [3, 5, 7]])
#             ),
#         )
#
#     nuts_kernel = numpyro.infer.NUTS(model)
#     mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=10000, num_samples=10000)
#     key = jax.random.PRNGKey(0)
#     mcmc.run(key)
#     mcmc.print_summary(exclude_deterministic=False)
#
#
# def test_sample_flat():
#     x = interface.normal(0.5, 2.0)
#     [xs] = inf.sample_flat([x], [], [])
#     assert abs(np.mean(xs) - 0.5) < 0.1
#     assert abs(np.std(xs) - 2.0) < 0.1
#
#
# def test_vmap_sample_flat():
#     loc = np.array([0.5, 1.1, 1.5])
#     scale = np.array([2.2, 3.3, 4.4])
#     x = interface.vmap(interface.normal, (0, 0), 3)(loc, scale)
#     [xs] = inf.sample_flat([x], [], [])
#     assert xs.shape == (10000, 3)
#     assert max(abs(np.mean(xs, axis=0) - loc)) < 0.1
#     assert max(abs(np.std(xs, axis=0) - scale)) < 0.1
#
#
# def test_deterministic_vmap():
#     loc = np.array([0.5, 1.1, 1.5])
#     scale = np.array([2.2, 3.3, 4.4])
#     x = interface.vmap(
#         lambda loc_i, scale_i: interface.exp(interface.normal(loc_i, scale_i)), 0
#     )(loc, scale)
#     [xs] = inf.sample_flat([x], [], [])
#     assert xs.shape == (10000, 3)
#
#
# def test_vmap():
#     loc = makerv(0)
#     scale = makerv(1)
#     x = interface.vmap(interface.normal_scale, None, axis_size=3)(loc, scale)
#     model, names = inf.get_model_flat([x], [], [])
#     nuts_kernel = numpyro.infer.NUTS(model)
#     mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=100, num_samples=100)
#     key = jax.random.PRNGKey(0)
#     mcmc.run(key)
#     mcmc.print_summary(exclude_deterministic=False)
#
#
# def test_evidence():
#     z = interface.normal(0, 1)
#     x = interface.normal(z, 1)
#     [zs] = inf.sample_flat([z], [x], [np.array(5.0)])
#     assert abs(jnp.mean(zs) - 2.5) < 0.1
#     assert abs(jnp.var(zs) - 0.5) < 0.1
#
#
# def test_deterministic():
#     z = interface.makerv(1.1)
#     x = interface.exp(z)
#     [zs] = inf.sample_flat([z], [], [])
#     print(f"{zs=}")
