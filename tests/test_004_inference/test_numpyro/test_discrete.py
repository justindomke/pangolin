from numpyro.contrib.funsor import config_enumerate

from pangolin import makerv, normal, print_upstream, ir, sample, vmap, bernoulli, categorical, bernoulli_logit, uniform, exponential
#from pangolin.inference.numpyro import numpyro_vmap_var_random, vmap_numpyro_pars, E
from pangolin.inference.numpyro import E
from pangolin.inference.numpyro.vmap import handle_vmap_random, handle_vmap_nonrandom
from util import inf_until_match, sample_until_match, sample_flat_until_match
import numpy as np
from jax import numpy as jnp
import numpyro
import jax
from numpyro import distributions as numpyro_dist
import scipy.stats

# In numpyro, we should be able to have ANY discrete RV if observed
# but we only support SIMPLE discrete LATENT RVs

def bernoulli_normal_expected(p, x_obs, scale):
    "if z~bern(p) and x~norm(z,scale), what is E[z|x=x_obs]?"
    # like0 = scipy.stats.norm.pdf(x_obs, 0, scale)
    # like1 = scipy.stats.norm.pdf(x_obs, 1, scale)
    # return p*like1 / ((1-p)*like0 + p*like1)

    logp0 = np.log(1-p) + scipy.stats.norm.logpdf(x_obs, 0, scale)
    logp1 = np.log(p) + scipy.stats.norm.logpdf(x_obs, 1, scale)
    return jnp.exp(logp1 - jnp.logaddexp(logp0, logp1))

def test_bernoulli_normal_numpyro():
    p = np.random.rand()
    x_obs = jnp.array(0.5 + np.random.randn())
    scale = np.abs(np.random.randn())

    def model():
        z = numpyro.sample("z", numpyro_dist.Bernoulli(p))
        x = numpyro.sample("x", numpyro_dist.Normal(z, scale), obs=x_obs)

    kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_warmup=1000,
        num_samples=1000,
        progress_bar=False,
    )
    key = jax.random.PRNGKey(np.random.randint(1000000))
    mcmc.run(key)

    #posterior_samples = mcmc.get_samples() # not needed
    predictive = numpyro.infer.Predictive(model, num_samples=10000, infer_discrete=True)
    conditional_samples = predictive(rng_key=key)
    E_z = np.mean(conditional_samples['z'])
    expected = bernoulli_normal_expected(p, x_obs, scale)
    print(f"{E_z=} {expected=}")
    assert np.abs(E_z-expected) < .05

def test_bernoulli_normal_numpyro_batched():
    batchsize = 10

    p = jnp.array(np.random.rand(batchsize))
    x_obs = jnp.array(0.5 + np.random.randn(batchsize))
    scale = jnp.abs(np.random.randn(batchsize))

    def model():
        with numpyro.plate("batchsize", batchsize) as i:
            z = numpyro.sample("z", numpyro_dist.Bernoulli(p))
            x = numpyro.sample("x", numpyro_dist.Normal(z, scale), obs=x_obs)

    kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_warmup=1000,
        num_samples=1000,
        progress_bar=False,
    )
    key = jax.random.PRNGKey(np.random.randint(1000000))
    mcmc.run(key)

    #posterior_samples = mcmc.get_samples() # not needed
    predictive = numpyro.infer.Predictive(model, num_samples=10000, infer_discrete=True)
    conditional_samples = predictive(rng_key=key)
    E_z = np.mean(conditional_samples['z'], axis=0)
    expected = bernoulli_normal_expected(p, x_obs, scale)
    print(f"{E_z=} {expected=}")
    assert np.all(np.abs(E_z-expected) < .05)

def test_bernoulli_normal():
    "Simplest test with a single latent variable that can be completely integrated out"

    p = np.random.rand()
    x_obs = 0.5 + np.random.randn()
    scale = np.abs(np.random.randn())

    z = bernoulli(p)
    x = normal(z,scale)

    expected = bernoulli_normal_expected(p, x_obs, scale)

    def testfun(Ez):
        print(f"{Ez=}")
        print(f"{expected=}")
        return np.abs(Ez-expected)<.01

    inf_until_match(E, z, x, x_obs, testfun)

def test_bernoulli_normal_with_distraction():
    "Same as above but add other an unrelated continuous variables"

    p = np.random.rand()
    x_obs = 0.5 + np.random.randn()
    scale = np.abs(np.random.randn())

    z = bernoulli(p)
    u = normal(0, scale)
    x = normal(z, scale)

    expected_Ez = bernoulli_normal_expected(p, x_obs, scale)
    expected_Eu = 0

    def testfun(expectations):
        Ez, Eu = expectations
        print(f"{Ez=}")
        print(f"{expected_Ez=}")
        print(f"{Eu=}")
        print(f"{expected_Eu=}")
        return np.abs(Ez - expected_Ez) < .01 and np.abs(Eu - expected_Eu) < .01

    inf_until_match(E, (z,u), x, x_obs, testfun)

def test_bernoulli_normal_with_downstream():
    "Same as above but add other a continuous variables downstream of z"

    p = np.random.rand()
    x_obs = 0.5 + np.random.randn()
    scale = np.abs(np.random.randn())

    z = bernoulli(p)
    u = normal(z, scale)
    x = normal(z, scale)

    expected_Ez = bernoulli_normal_expected(p, x_obs, scale)
    expected_Eu = expected_Ez

    def testfun(expectations):
        Ez, Eu = expectations
        print(f"{Ez=} {expected_Ez=} {Eu=} {expected_Eu=}")
        return np.abs(Ez - expected_Ez) < .01 and np.abs(Eu - expected_Eu) < .01

    inf_until_match(E, (z,u), x, x_obs, testfun)

def bernoulli_bernoulli_expected(p, x_obs):
    "if z~bern(p) and x~bern(z), what is E[z|x=x_obs]?"
    like0 = scipy.stats.bernoulli.pmf(x_obs, 0.25)
    like1 = scipy.stats.bernoulli.pmf(x_obs, 0.25+0.5)
    return p*like1 / ((1-p)*like0 + p*like1)

def test_bernoulli_bernoulli():
    p = np.random.rand()
    x_obs = np.random.randint(2)

    z = bernoulli(p)
    x = bernoulli(0.25+0.5*z)

    expected = bernoulli_bernoulli_expected(p, x_obs)

    def testfun(Ez):
        print(f"{Ez=} {expected=} {x_obs=} {p=}")
        return np.abs(Ez - expected) < .01

    inf_until_match(E, z, x, x_obs, testfun)

def test_bernoulli_bernoulli_with_downstream():
    p = np.random.rand()
    x_obs = np.random.randint(2)

    z = bernoulli(p)
    x = bernoulli(0.25+0.5*z)
    u = normal(z, 1.0)

    expected_Ez = bernoulli_bernoulli_expected(p, x_obs)
    expected_Eu = expected_Ez

    def testfun(expectations):
        Ez, Eu = expectations
        print(f"{Ez=} {expected_Ez=} {Eu=} {expected_Eu=} {x_obs=} {p=}")
        return np.abs(Ez - expected_Ez) < .01 and np.abs(Eu - expected_Eu) < .01

    inf_until_match(E, (z,u), x, x_obs, testfun)

def categorical_categorical_expected(p,M,x_obs):
    nstates = len(p)
    probs = np.array([p[z] * M[z,x_obs] for z in range(nstates)])
    probs = probs / np.sum(probs)
    return np.arange(nstates) @ probs

def test_categorical_categorical():
    nstates = 2 + np.random.randint(5)
    # initial probs
    p = np.random.rand(nstates)
    p = p / np.sum(p)
    # stochastic matrix
    M = np.random.rand(nstates, nstates)
    M = M / np.sum(M, axis=1, keepdims=True)

    x_obs = np.random.randint(nstates)

    z = categorical(p)
    x = categorical(makerv(M)[z,:])

    expected_Ez = categorical_categorical_expected(p,M,x_obs)

    def testfun(Ez):
        print(f"{Ez=} {expected_Ez=}")
        return np.abs(Ez - expected_Ez) < .01

    inf_until_match(E, z, x, x_obs, testfun)

def triple_bernoulli_bernoulli_expected(p, x_obs):
    "if z~bern(p) and x~bern(z), what is E[z|x=x_obs]?"
    probs = np.zeros((2,2))
    for z in range(2):
        for y in range(2):
            p_z = p*z + (1-p)*(1-z)
            p_y_given_z = scipy.stats.bernoulli.pmf(y, 0.25 + 0.5*z)
            p_x_given_y = scipy.stats.bernoulli.pmf(x_obs, 0.25 + 0.5 * y)
            probs[z,y] = p_z * p_y_given_z * p_x_given_y

    probs = probs / np.sum(probs)
    Ez = np.sum(probs[1,:])
    Ey = np.sum(probs[:, 1])
    return Ez, Ey

def test_triple_bernoulli():
    p = np.random.rand()

    z = bernoulli(p)
    y = bernoulli(0.25 + 0.5 * z)
    x = bernoulli(0.25 + 0.5 * y)

    x_obs = np.random.randint(2)

    Ez = E(z,x,x_obs)
    print(f"{Ez=}")

    expected_Ez, expected_Ey = triple_bernoulli_bernoulli_expected(p,x_obs)

    def testfun(expectations):
        Ez, Ey = expectations
        print(f"{Ez=} {expected_Ez=} {Ey=} {expected_Ey=}")
        return np.abs(Ez - expected_Ez) < .01 and np.abs(Ey - expected_Ey) < .01

    inf_until_match(E, (z,y), x, x_obs, testfun)

def long_bernoulli_chain_expected(p,length,x_obs):
    m = [np.array([1-p,p])]
    for i in range(length):
        #p0 = .95 * m[-1][0] + .05 * m[-1][1]
        #p1 = .05 * m[-1][0] + .95 * m[-1][1]
        #m.append(np.array([p0, p1]))
        q = .05 * m[-1][0] + .95 * m[-1][1]
        m.append(np.array([1-q, q]))
    m = np.array(m)

    if x_obs == 0:
        q = .05
    else:
        q = .95
    n = [np.array([1-q, q])]
    for i in range(length):
        q = .05 * n[-1][0] + .95 * n[-1][1]
        n.append(np.array([1 - q, q]))
    n = np.array(list(reversed(n)))

    print(f"{m=}")
    print(f"{n=}")

    probs = m*n
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs[:,1]

def test_long_bernoulli_chain():
    p = np.random.rand()
    length = 5
    x_obs = np.random.randint(2)

    expected_Ez = long_bernoulli_chain_expected(p, length,x_obs)

    z = [bernoulli(p)]
    for i in range(length):
        z.append(bernoulli(0.05 + 0.9 * z[-1]))

    x = bernoulli(0.05 + 0.9 * z[-1])

    def testfun(Ez):
        Ez = np.array(Ez)
        print(f"         {Ez=}")
        print(f"{expected_Ez=}")
        return np.all(np.abs(Ez - expected_Ez) < .05)

    inf_until_match(E, z, x, x_obs, testfun)


    #Ez = E(z,x,x_obs, niter=100000)
    #Ez = np.array(Ez)
    #print(f"{x_obs=}")
    #print(f"         {Ez=}")
    #print(f"{expected_Ez=}")

def disabled_test_long_bernoulli_chain_raw_numpyro():
    """
    Try to use numpyro's scan unitlity.
    It seems to be very unreliable. It seems to return lists of values rather than, like,
    random variables. And can't be used inside a vmap. Just not very reliable at all.
    """

    import numpyro.contrib.control_flow

    scales = jnp.ones(50)
    x_obs = jnp.array(np.random.randint(2))

    def model():

        def transition(carry, scale):
            next = numpyro.sample('z',numpyro_dist.Bernoulli(0.1 + 0.8*carry))
            return next, next

        z_last, tmp = numpyro.contrib.control_flow.scan(transition, 0, scales)
        x = numpyro.sample('x',numpyro_dist.Bernoulli(0.1 + 0.8*z_last), obs=x_obs)

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

    z_samples = conditional_samples['z']
    print(jnp.mean(z_samples,axis=0))

    #print(f"{conditional_samples['z'].shape=}")
    #print(f"{conditional_samples['x'].shape=}")
