from numpyro.contrib.funsor import config_enumerate

from pangolin.simple_interface import (
    normal,
    lognormal,
    ir,
    vmap,
    bernoulli,
    categorical,
    binomial,
    beta_binomial,
    multinomial,
    bernoulli_logit,
    uniform,
    exponential,
    constant,
    exp,
)


from pangolin.inference.numpyro import E, sample
from pangolin.inference.numpyro.test_util import (
    inf_until_match,
    sample_until_match,
    sample_flat_until_match,
)
import numpy as np
from jax import numpy as jnp
import numpyro
import jax
from numpyro import distributions as numpyro_dist
from pangolin.inference.numpyro.vmap import (
    handle_vmap_random,
    handle_vmap_nonrandom,
    vmap_numpyro_pars,
)


def test_handle_vmap_nonrandom_1d():
    x_jax = jnp.array([0.0, 1.0, 2.0])
    y = vmap(exp)(x_jax)
    out = handle_vmap_nonrandom(y.op, x_jax, is_observed=False)
    assert jnp.allclose(out, jnp.exp(x_jax))


def test_handle_vmap_nonrandom_2d():
    x_jax = jnp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    y = vmap(vmap(exp))(x_jax)
    out = handle_vmap_nonrandom(y.op, x_jax, is_observed=False)
    assert jnp.allclose(out, jnp.exp(x_jax))


def test_handle_vmap_random_1d():
    x_jax = jnp.array([0.0, 1.0, 2.0])
    y = vmap(exponential)(x_jax)
    out = handle_vmap_random(y.op, x_jax, is_observed=False)

    with numpyro.handlers.seed(rng_seed=0):  # type: ignore
        out_sampled = numpyro.sample("out", out)

    assert out_sampled.shape == (3,)  # type: ignore


def test_handle_vmap_random_1d_normal():
    loc = jnp.array([0.0, 1.0, 2.0])
    scale = jnp.array(0.000001)
    y = vmap(normal, [0, None])(loc, scale)
    out = handle_vmap_random(y.op, loc, scale, is_observed=False)

    with numpyro.handlers.seed(rng_seed=0):  # type: ignore
        out_sampled = numpyro.sample("out", out)

    assert out_sampled.shape == (3,)  # type: ignore
    assert jnp.allclose(out_sampled, loc, atol=1e-3, rtol=1e-3)


def test_vmap_normal1():
    y = vmap(normal, in_axes=None, axis_size=3)(0.5, 1.5)

    def testfun(y_samps):
        E_y = jnp.mean(y_samps, axis=0)
        std_y = jnp.std(y_samps, axis=0)
        return np.max(np.abs(E_y - 0.5)) < 0.1 and np.max(np.abs(std_y - 1.5)) < 0.1

    inf_until_match(sample, y, [], [], testfun)


def test_vmap_normal2():
    locs = jnp.array([3, 4, 5])
    std = 1.5
    y = vmap(normal, in_axes=[0, None], axis_size=3)(locs, std)

    def testfun(y_samps):
        E_y = jnp.mean(y_samps, axis=0)
        std_y = jnp.std(y_samps, axis=0)
        return np.max(np.abs(E_y - locs)) < 0.1 and np.max(np.abs(std_y - std) < 0.1)

    inf_until_match(sample, y, [], [], testfun)


def test_vmap_normal3():
    loc = 3
    stds = jnp.array([6, 7, 8])
    y = vmap(normal, in_axes=[None, 0], axis_size=3)(loc, stds)

    def testfun(y_samps):
        E_y = jnp.mean(y_samps, axis=0)
        std_y = jnp.std(y_samps, axis=0)
        return np.max(np.abs(E_y - loc)) < 0.1 and np.max(np.abs(std_y - stds)) < 0.1

    inf_until_match(sample, y, [], [], testfun)


def test_vmap_normal4():
    locs = jnp.array([3, 4, 5])
    stds = jnp.array([6, 7, 8])
    y = vmap(normal, in_axes=[0, 0], axis_size=3)(locs, stds)

    def testfun(y_samps):
        E_y = jnp.mean(y_samps, axis=0)
        std_y = jnp.std(y_samps, axis=0)
        return np.max(np.abs(E_y - locs)) < 0.1 and np.max(np.abs(std_y - stds)) < 0.1

    inf_until_match(sample, y, [], [], testfun)


# def test_vmap_bernoulli_support():
#     y = vmap(bernoulli,None,5)(0.5)
#     op = y.op
#     numpyro_var = handle_vmap_random(op, jnp.array(0.5), is_observed=False)
#     assert numpyro_var.support == numpyro.distributions.Bernoulli(0.5).support
#     assert numpyro_var.support.is_discrete


# def test_bernoulli_logit_inference1():
#     w = normal(0, 1)
#     y = bernoulli_logit(w)
#     yp = bernoulli_logit(w)
#     yp_samples = sample(yp, y, 0, niter=1000)
#     assert all(yi in [0, 1] for yi in yp_samples)


# def test_bernoulli_logit_inference2():
#     w = normal(0, 1)
#     y = vmap(bernoulli_logit, None, 5)(w)
#     yp = bernoulli_logit(w)
#     yp_samples = sample(yp, y, [0, 0, 1, 1, 0], niter=1000)
#     assert all(yi in [0, 1] for yi in yp_samples)


# def test_bernoulli_logit_inference3():
#     length = 5
#     w = normal(0, 1)
#     yp = vmap(bernoulli_logit, None, length)(w)
#     yp_samples = sample(yp, w, 0.0, niter=1000)
#     assert all(yi in [0, 1] for yi in yp_samples.ravel())


# def test_bernoulli_logit_inference4():
#     length = 5
#     w = normal(0, 1)
#     y = bernoulli_logit(w)
#     yp = vmap(bernoulli_logit, None, length)(w)
#     y_obs = jnp.array(0)
#     yp_samples = sample(yp, y, y_obs, niter=1000)
#     assert all(yi in [0, 1] for yi in yp_samples.ravel())


# def test_bernoulli_logit_inference5():
#     length = 5
#     w = normal(0, 1)
#     y = vmap(bernoulli_logit, None, length)(w)
#     yp = vmap(bernoulli_logit, None, length)(w)
#     y_obs = jnp.array(np.random.randint(0, 2, size=length))
#     yp_samples = sample(yp, y, y_obs, niter=1000)
#     assert all(yi in [0, 1] for yi in yp_samples.ravel())


# def test_bernoulli_logit_inference6():
#     w = normal(0, 1)
#     y = vmap(bernoulli_logit, None, 5)(w)
#     yp = vmap(bernoulli_logit, None, 5)(w)
#     yp_samples = sample(yp, niter=1000)
#     # print(yp_samples)
#     assert all(yi in [0, 1] for yi in yp_samples.ravel())


# def test_bernoulli_logit_inference_2d():
#     length1 = 3
#     length2 = 4
#     w = normal(0, 1)
#     y = vmap(vmap(bernoulli_logit, None, length1), None, length2)(w)
#     yp = vmap(vmap(bernoulli_logit, None, length1), None, length2)(w)
#     yp_samples = sample(
#         yp, y, np.random.randint(2, size=(length2, length1)), niter=1000
#     )
#     assert all(yi in [0, 1] for yi in yp_samples.ravel())


# def test_bernoulli_logit_inference3_2d_mapped():
#     length1 = 3
#     length2 = 4
#     # w = normal(0,1)
#     w = vmap(vmap(normal, None, length1), None, length2)(0, 1)
#     y = vmap(vmap(bernoulli_logit, 0, length1), 0, length2)(w)
#     yp = vmap(vmap(bernoulli_logit, 0, length1), 0, length2)(w)
#     yp_samples = sample(
#         yp, y, np.random.randint(2, size=(length2, length1)), niter=1000
#     )
#     assert all(yi in [0, 1] for yi in yp_samples.ravel())


def test_exponential_inference1():
    w = lognormal(0, 1)
    y = exponential(w)
    yp = exponential(w)
    yp_samples = sample(yp, y, 0, niter=1000)
    assert all(yi > 0 for yi in yp_samples)


def test_exponential_inference2():
    w = lognormal(0, 1)
    y = vmap(exponential, None, 5)(w)
    yp = exponential(w)
    yp_samples = sample(yp, y, [0.1, 0.1, 1.5, 1.5, 0.1], niter=1000)
    assert all(yi > 0 for yi in yp_samples)


def test_exponential_inference3():
    length = 5
    w = lognormal(0, 1)
    yp = vmap(exponential, None, length)(w)
    yp_samples = sample(yp, w, 0.5, niter=1000)
    assert all(yi > 0 for yi in yp_samples.ravel())


def test_exponential_inference4():
    length = 5
    w = lognormal(0, 1)
    y = exponential(w)
    yp = vmap(exponential, None, length)(w)
    y_obs = jnp.array(0)
    yp_samples = sample(yp, y, y_obs, niter=1000)
    assert all(yi > 0 for yi in yp_samples.ravel())


def test_exponential_inference5():
    length = 5
    w = lognormal(0, 1)
    y = vmap(exponential, None, length)(w)
    yp = vmap(exponential, None, length)(w)
    y_obs = jnp.array(1 / np.random.uniform(0, 1, size=length))
    yp_samples = sample(yp, y, y_obs, niter=1000)
    assert all(yi > 0 for yi in yp_samples.ravel())


# def test_bernoulli_logit_inference6():
#     w = normal(0, 1)
#     y = vmap(bernoulli_logit, None, 5)(w)
#     yp = vmap(bernoulli_logit, None, 5)(w)
#     yp_samples = sample(yp, niter=1000)
#     # print(yp_samples)
#     assert all(yi in [0, 1] for yi in yp_samples.ravel())


# def test_bernoulli_logit_inference_2d():
#     length1 = 3
#     length2 = 4
#     w = normal(0, 1)
#     y = vmap(vmap(bernoulli_logit, None, length1), None, length2)(w)
#     yp = vmap(vmap(bernoulli_logit, None, length1), None, length2)(w)
#     yp_samples = sample(
#         yp, y, np.random.randint(2, size=(length2, length1)), niter=1000
#     )
#     assert all(yi in [0, 1] for yi in yp_samples.ravel())


# def test_bernoulli_logit_inference3_2d_mapped():
#     length1 = 3
#     length2 = 4
#     # w = normal(0,1)
#     w = vmap(vmap(normal, None, length1), None, length2)(0, 1)
#     y = vmap(vmap(bernoulli_logit, 0, length1), 0, length2)(w)
#     yp = vmap(vmap(bernoulli_logit, 0, length1), 0, length2)(w)
#     yp_samples = sample(
#         yp, y, np.random.randint(2, size=(length2, length1)), niter=1000
#     )
#     assert all(yi in [0, 1] for yi in yp_samples.ravel())


def assert_numpyro_pars_correct(in_axes_list, p1, p2, *, axis_size_list=None):
    if axis_size_list is None:
        axis_size_list = [None] * len(in_axes_list)

    fun = jnp.add
    for in_axes, axis_size in zip(
        reversed(in_axes_list), reversed(axis_size_list), strict=True
    ):
        fun = jax.vmap(fun, in_axes, axis_size=axis_size)
    vmap_sum = fun(p1, p2)
    # print(f"{vmap_sum.shape=}")

    op = ir.Normal()
    for in_axes, axis_size in zip(
        reversed(in_axes_list), reversed(axis_size_list), strict=True
    ):
        op = ir.VMap(op, in_axes, axis_size=axis_size)
    new_p1, new_p2 = vmap_numpyro_pars(op, p1, p2)
    # print(f"{new_p1.shape=}")
    # print(f"{new_p2.shape=}")

    broadcast_sum = new_p1 + new_p2

    np.testing.assert_allclose(broadcast_sum, vmap_sum)


def test_vmap_numpyro_pars_single1():
    in_axes_list = [[None, None]]
    axis_size_list = [3]
    p1 = jnp.array(1)
    p2 = jnp.array(2)
    assert_numpyro_pars_correct(in_axes_list, p1, p2, axis_size_list=axis_size_list)


def test_vmap_numpyro_pars_single2():
    in_axes_list = [[0, None]]
    p1 = jnp.array([1, 2, 3])
    p2 = jnp.array(4)
    assert_numpyro_pars_correct(in_axes_list, p1, p2)


def test_vmap_numpyro_pars_single3():
    in_axes_list = [[None, 0]]
    p1 = jnp.array(1.1)
    p2 = jnp.array([2.2, 3.3, 4.4])
    assert_numpyro_pars_correct(in_axes_list, p1, p2)


def test_vmap_numpyro_pars_single4():
    in_axes_list = [[0, 0]]
    p1 = jnp.array([1, 2, 3])
    p2 = jnp.array([4, 4, 6])
    assert_numpyro_pars_correct(in_axes_list, p1, p2)


def test_vmap_numpyro_pars_double1():
    in_axes_list = [[0, 0], [0, 0]]
    p1 = jnp.array(np.random.randn(4, 3))
    p2 = jnp.array(np.random.randn(4, 3))
    assert_numpyro_pars_correct(in_axes_list, p1, p2)


def test_vmap_numpyro_pars_double2():
    in_axes_list = [[None, 0], [0, 0]]
    p1 = jnp.array(np.random.randn(4))
    p2 = jnp.array(np.random.randn(3, 4))
    assert_numpyro_pars_correct(in_axes_list, p1, p2)


def test_vmap_numpyro_pars_double3():
    in_axes_list = [[0, None], [0, 0]]
    p1 = jnp.array(np.random.randn(3, 4))
    p2 = jnp.array(np.random.randn(4))
    assert_numpyro_pars_correct(in_axes_list, p1, p2)


def test_vmap_numpyro_pars_double4():
    in_axes_list = [[0, 0], [None, 0]]
    p1 = jnp.array(np.random.randn(3))
    p2 = jnp.array(np.random.randn(3, 4))
    assert_numpyro_pars_correct(in_axes_list, p1, p2)


def test_vmap_numpyro_pars_double5():
    in_axes_list = [[0, None], [None, 0]]
    p1 = jnp.array(np.random.randn(3))
    p2 = jnp.array(np.random.randn(4))
    assert_numpyro_pars_correct(in_axes_list, p1, p2)


def test_vmap_numpyro_pars_double6():
    in_axes_list = [[None, 0], [0, None]]
    p1 = jnp.array(np.random.randn(3))
    p2 = jnp.array(np.random.randn(4))
    assert_numpyro_pars_correct(in_axes_list, p1, p2)


def test_uniform_bernoulli():
    z = vmap(vmap(uniform, None, 3), None, 5)(0, 1)
    x = vmap(vmap(bernoulli))(z)
    x_obs = np.round(np.random.rand(5, 3))

    expected_Ez = (1 + x_obs) / 3  # 1/3 for x=0 and 2/3 for x=1

    def testfun(Ez):
        return np.all(np.abs(Ez - expected_Ez) < 0.01)

    inf_until_match(E, z, x, x_obs, testfun)


# def test_bernoulli_bernoulli():
#     p = np.ones((5, 3)) * 0.5
#     z = vmap(vmap(bernoulli))(p)
#     assert isinstance(z.parents[0].op, ir.Constant)

#     # x = vmap(vmap(bernoulli))(0.1 + 0.8 * z)
#     # x = vmap(vmap(lambda zij: bernoulli(0.1 + 0.8 * zij)))(z)
#     x = vmap(vmap(lambda zij: bernoulli(zij * 0.8 + 0.1)))(z)
#     x_obs = np.round(np.random.rand(5, 3))

#     assert z.shape == (5, 3)
#     assert x.shape == (5, 3)

#     expected_Ez = 0.1 + 0.8 * x_obs

#     def testfun(Ez):
#         return np.all(np.abs(Ez - expected_Ez) < 0.01)

#     inf_until_match(E, z, x, x_obs, testfun)


def test_uniform_uniform_numpyro1():
    from numpyro import distributions as dists
    from numpyro import enable_validation

    enable_validation()

    a = 1.4
    b = 10.0
    x_obs = 1.5

    def model():
        z = numpyro.sample("z", dists.Uniform(a, b))
        x = numpyro.sample("x", dists.Uniform(z - 1, z + 1), obs=x_obs)

    kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=1000, num_samples=1000)
    mcmc.run(jax.random.PRNGKey(0))
    samples = mcmc.get_samples()

    lo = max(a, x_obs - 1)
    hi = min(b, x_obs + 1)
    expected = (lo + hi) / 2

    zs = samples["z"]
    assert np.abs(np.mean(zs) - expected) < 0.1


def test_uniform_uniform():
    # z ~ uniform(a, b)
    # x ~ uniform(z-0.5, z+0.5)
    #
    # each z between x-0.5 and x+0.5 equally likely
    # but bounded by a and b

    a = -0.5
    b = 1.3

    p1 = np.ones((5, 3)) * a
    p2 = np.ones((5, 3)) * b
    z = vmap(vmap(uniform))(p1, p2)
    assert isinstance(z.parents[0].op, ir.Constant)

    x = vmap(vmap(lambda zij: uniform(zij - 1, zij + 1)))(z)
    x_obs = 0.2 + 0.6 * np.random.rand(5, 3)

    assert z.shape == (5, 3)
    assert x.shape == (5, 3)

    lb = np.maximum(a, x_obs - 1)
    ub = np.minimum(b, x_obs + 1)

    expected = (lb + ub) / 2

    def testfun(Ez):
        return np.all(np.abs(Ez - expected) < 0.01)

    inf_until_match(E, z, x, x_obs, testfun)


# def test_bernoulli_uniform():
#     p = np.ones((5,3))*0.5
#     z = vmap(vmap(bernoulli))(p)
#     x = vmap(vmap(uniform))(z, z+1)
#     x_obs = np.random.rand(5,3)
#
#     #print(f"{E(z,x,x_obs)=}")
#
#     # p(z,x) \propto 0.5 * I[0.1*z <= x <= 0.9+0.1z]
#     #
#


def test_bernoulli_uniform_manual():
    x_obs = 0.9

    def eval(z):
        return float(z * 0.5 <= x_obs <= 1 + 0.5 * z)

    def mcmc(niter):
        zs = []
        z = 0
        p = eval(z)
        while p == 0:
            z = np.random.randint(2)
            p = eval(z)
        for i in range(niter):
            new_z = np.random.randint(2)
            new_p = eval(new_z)
            if np.random.rand() < new_p / p:
                z = new_z
                p = new_p
            zs.append(z)
        return zs

    zs = mcmc(10000)

    # print(f"{np.mean(zs)}")


def test_bernoulli_uniform_raw_numpyro():
    p = jnp.array(np.random.rand() * 0 + 0.1)
    # x_obs = jnp.array(np.random.rand()*2)
    x_obs = jnp.array(0.4)
    # print(f"{x_obs=}")

    def model():
        # a = numpyro.sample("a", numpyro_dist.Normal(0,1))
        z = numpyro.sample("z", numpyro_dist.Bernoulli(p))
        x = numpyro.sample("x", numpyro_dist.Uniform(0.5 * z, 1 + 0.5 * z), obs=x_obs)

    # kernel = numpyro.infer.DiscreteHMCGibbs(
    #     numpyro.infer.NUTS(model), modified=True
    # )
    # kernel = numpyro.infer.MixedHMC(numpyro.infer.HMC(model), num_discrete_updates=200)
    kernel = numpyro.infer.NUTS(model)

    mcmc = numpyro.infer.MCMC(
        kernel,
        num_warmup=1000,
        num_samples=1000,
        progress_bar=False,
    )
    key = jax.random.PRNGKey(0)

    # numpyro gives some annoying future warnings
    import warnings

    with warnings.catch_warnings(action="ignore", category=FutureWarning):  # type: ignore
        mcmc.run(key)

    # mcmc.print_summary()
    print(mcmc.get_samples())

    posterior_samples = mcmc.get_samples()

    # predictive = numpyro.infer.Predictive(model, posterior_samples)
    predictive = numpyro.infer.Predictive(model, num_samples=1000)
    key = jax.random.PRNGKey(0)
    conditional_samples = predictive(rng_key=key)

    # print(f"{conditional_samples['z'].shape=}")
    # print(f"{conditional_samples['x'].shape=}")
