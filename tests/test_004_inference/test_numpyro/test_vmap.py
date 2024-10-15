from numpyro.contrib.funsor import config_enumerate

from pangolin import (
    makerv,
    normal,
    print_upstream,
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
)
from pangolin.inference.numpyro import E, sample
from util import inf_until_match, sample_until_match, sample_flat_until_match
import numpy as np
from jax import numpy as jnp
import numpyro
import jax
from numpyro import distributions as numpyro_dist
from pangolin.inference.numpyro.vmap import handle_vmap_random, handle_vmap_nonrandom

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
    y = vmap(normal, in_axes=[0,None], axis_size=3)(locs, std)

    def testfun(y_samps):
        E_y = jnp.mean(y_samps, axis=0)
        std_y = jnp.std(y_samps, axis=0)
        return np.max(np.abs(E_y - locs)) < 0.1 and np.max(np.abs(std_y - std) < 0.1)

    inf_until_match(sample, y, [], [], testfun)


def test_vmap_normal3():
    loc = 3
    stds = jnp.array([6, 7, 8])
    y = vmap(normal, in_axes=[None,0], axis_size=3)(loc, stds)

    def testfun(y_samps):
        E_y = jnp.mean(y_samps, axis=0)
        std_y = jnp.std(y_samps, axis=0)
        return np.max(np.abs(E_y - loc)) < 0.1 and np.max(np.abs(std_y - stds)) < 0.1

    inf_until_match(sample, y, [], [], testfun)


def test_vmap_normal4():
    locs = jnp.array([3, 4, 5])
    stds = jnp.array([6, 7, 8])
    y = vmap(normal, in_axes=[0,0], axis_size=3)(locs, stds)

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

def test_bernoulli_logit_inference1():
    w = normal(0,1)
    y = bernoulli_logit(w)
    yp = bernoulli_logit(w)
    yp_samples = sample(yp,y,0, niter=1000)
    assert all(yi in [0,1] for yi in yp_samples)

def test_bernoulli_logit_inference2():
    w = normal(0,1)
    y = vmap(bernoulli_logit,None,5)(w)
    yp = bernoulli_logit(w)
    yp_samples = sample(yp,y,[0,0,1,1,0], niter=1000)
    print(yp_samples)
    assert all(yi in [0,1] for yi in yp_samples)

def test_bernoulli_logit_inference3():
    w = normal(0,1)
    y = vmap(bernoulli_logit,None,5)(w)
    yp = vmap(bernoulli_logit,None,5)(w)
    print_upstream((y,yp))
    yp_samples = sample(yp,y,[0,0,1,1,0], niter=1000)
    print(yp_samples)
    assert all(yi in [0,1] for yi in yp_samples.ravel())

def test_bernoulli_logit_inference4():
    w = normal(0,1)
    y = vmap(bernoulli_logit,None,5)(w)
    yp = vmap(bernoulli_logit,None,5)(w)
    yp_samples = sample(yp, niter=1000)
    print(yp_samples)
    assert all(yi in [0,1] for yi in yp_samples.ravel())

def assert_numpyro_pars_correct(in_axes_list,p1,p2,*,axis_size_list=None):
    if axis_size_list is None:
        axis_size_list = [None]*len(in_axes_list)

    fun = jnp.add
    for in_axes, axis_size in zip(reversed(in_axes_list), reversed(axis_size_list), strict=True):
        fun = jax.vmap(fun, in_axes, axis_size=axis_size)
    vmap_sum = fun(p1, p2)
    print(f"{vmap_sum.shape=}")

    op = ir.Normal()
    for in_axes, axis_size in zip(reversed(in_axes_list), reversed(axis_size_list), strict=True):
        op = ir.VMap(op, in_axes, axis_size=axis_size)
    new_p1, new_p2 = vmap_numpyro_pars(op, p1, p2)
    print(f"{new_p1.shape=}")
    print(f"{new_p2.shape=}")

    broadcast_sum = new_p1 + new_p2

    np.testing.assert_allclose(broadcast_sum, vmap_sum)

def test_vmap_numpyro_pars_single1():
    in_axes_list = [[None, None]]
    axis_size_list = [3]
    p1 = jnp.array(1)
    p2 = jnp.array(2)
    assert_numpyro_pars_correct(in_axes_list, p1, p2, axis_size_list=axis_size_list)

def test_vmap_numpyro_pars_single2():
    in_axes_list = [[0,None]]
    p1 = jnp.array([1,2,3])
    p2 = jnp.array(4)
    assert_numpyro_pars_correct(in_axes_list,p1,p2)

def test_vmap_numpyro_pars_single3():
    in_axes_list = [[None,0]]
    p1 = jnp.array(1.1)
    p2 = jnp.array([2.2,3.3,4.4])
    assert_numpyro_pars_correct(in_axes_list,p1,p2)

def test_vmap_numpyro_pars_single4():
    in_axes_list = [[0, 0]]
    p1 = jnp.array([1, 2, 3])
    p2 = jnp.array([4, 4, 6])
    assert_numpyro_pars_correct(in_axes_list, p1, p2)

def test_vmap_numpyro_pars_double1():
    in_axes_list = [[0, 0],[0, 0]]
    p1 = jnp.array(np.random.randn(4,3))
    p2 = jnp.array(np.random.randn(4,3))
    assert_numpyro_pars_correct(in_axes_list, p1, p2)

def test_vmap_numpyro_pars_double2():
    in_axes_list = [[None, 0],[0, 0]]
    p1 = jnp.array(np.random.randn(4))
    p2 = jnp.array(np.random.randn(3,4))
    assert_numpyro_pars_correct(in_axes_list, p1, p2)

def test_vmap_numpyro_pars_double3():
    in_axes_list = [[0, None], [0, 0]]
    p1 = jnp.array(np.random.randn(3, 4))
    p2 = jnp.array(np.random.randn(4))
    assert_numpyro_pars_correct(in_axes_list, p1, p2)

def test_vmap_numpyro_pars_double4():
    in_axes_list = [[0, 0], [None, 0]]
    p1 = jnp.array(np.random.randn(3))
    p2 = jnp.array(np.random.randn(3,4))
    assert_numpyro_pars_correct(in_axes_list, p1, p2)

def test_vmap_numpyro_pars_double5():
    in_axes_list = [[0, None], [None, 0]]
    p1 = jnp.array(np.random.randn(3))
    p2 = jnp.array(np.random.randn(4))
    assert_numpyro_pars_correct(in_axes_list, p1, p2)

def test_vmap_numpyro_pars_double6():
    in_axes_list = [[None,0], [0,None]]
    p1 = jnp.array(np.random.randn(3))
    p2 = jnp.array(np.random.randn(4))
    assert_numpyro_pars_correct(in_axes_list, p1, p2)

def test_uniform_bernoulli():
    z = vmap(vmap(uniform,None,3),None,5)(0,1)
    x = vmap(vmap(bernoulli))(z)
    x_obs = np.round(np.random.rand(5,3))

    expected_Ez = (1 + x_obs)/3 # 1/3 for x=0 and 2/3 for x=1

    def testfun(Ez):
        return np.all(np.abs(Ez-expected_Ez)<.01)

    inf_until_match(E, z, x, x_obs, testfun)

def test_bernoulli_bernoulli():
    p = np.ones((5,3))*0.5
    z = vmap(vmap(bernoulli))(p)
    x = vmap(vmap(bernoulli))(.1+0.8*z)
    x_obs = np.round(np.random.rand(5,3))

    expected_Ez = (.1 + 0.8*x_obs)
    def testfun(Ez):
        return np.all(np.abs(Ez-expected_Ez)<.01)

    inf_until_match(E, z, x, x_obs, testfun)


def test_bernoulli_uniform():
    p = np.ones((5,3))*0.5
    z = vmap(vmap(bernoulli))(p)
    x = vmap(vmap(uniform))(z, z+1)
    x_obs = np.random.rand(5,3)

    print(f"{E(z,x,x_obs)=}")

    # p(z,x) \propto 0.5 * I[0.1*z <= x <= 0.9+0.1z]
    #

def test_bernoulli_uniform_manual():
    x_obs = 0.9

    def eval(z):
        return float(z*0.5 <= x_obs <= 1 + 0.5*z)

    def mcmc(niter):
        zs = []
        z = 0
        p = eval(z)
        while p==0:
            z = np.random.randint(2)
            p = eval(z)
        for i in range(niter):
            new_z = np.random.randint(2)
            new_p = eval(new_z)
            if np.random.rand() < new_p/p:
                z = new_z
                p = new_p
            zs.append(z)
        return zs

    zs = mcmc(10000)

    print(f"{np.mean(zs)}")


def test_bernoulli_uniform_raw_numpyro():
    p = jnp.array(np.random.rand()*0+0.1)
    #x_obs = jnp.array(np.random.rand()*2)
    x_obs = jnp.array(0.4)
    print(f"{x_obs=}")

    def model():
        #a = numpyro.sample("a", numpyro_dist.Normal(0,1))
        z = numpyro.sample("z", numpyro_dist.Bernoulli(p))
        x = numpyro.sample("x", numpyro_dist.Uniform(0.5*z,1+0.5*z), obs=x_obs)

    # kernel = numpyro.infer.DiscreteHMCGibbs(
    #     numpyro.infer.NUTS(model), modified=True
    # )
    #kernel = numpyro.infer.MixedHMC(numpyro.infer.HMC(model), num_discrete_updates=200)
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

    #mcmc.print_summary()
    print(mcmc.get_samples())

    posterior_samples = mcmc.get_samples()

    #predictive = numpyro.infer.Predictive(model, posterior_samples)
    predictive = numpyro.infer.Predictive(model, num_samples=1000)
    key = jax.random.PRNGKey(0)
    conditional_samples = predictive(rng_key=key)

    print(f"{conditional_samples['z'].shape=}")
    print(f"{conditional_samples['x'].shape=}")

