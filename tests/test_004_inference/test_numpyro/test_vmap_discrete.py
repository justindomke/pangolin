from numpyro.contrib.funsor import config_enumerate

from inference.old_numpyro import vmap_numpyro_pars
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
    poisson,
)
from pangolin.inference.numpyro import E, sample
from util import inf_until_match, sample_until_match, sample_flat_until_match
import numpy as np
from jax import numpy as jnp
import numpyro
import jax
from numpyro import distributions as numpyro_dist
from pangolin.inference.numpyro.vmap import handle_vmap_random, handle_vmap_nonrandom
from pangolin.inference.numpyro.vmap import vmap_rv_plate

thresh = 0.05

def test_vmap_bernoulli_iid():
    x = vmap(bernoulli, None, 3)(0.5)

    def testfun(Ex):
        return np.all(np.abs(Ex - 0.5) < thresh)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_bernoulli_0():
    p = jnp.array([0.2, 0.7, 0.9])
    x = vmap(bernoulli)(p)

    def testfun(Ex):
        return np.all(np.abs(Ex - p) < thresh)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_poisson_iid():
    l = jnp.array(0.7)
    x = vmap(poisson,None,axis_size=3)(l)
    expected = numpyro_dist.Poisson(l).mean

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < thresh)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_poisson_0():
    l = jnp.array([.5, 1.5, 2.5])
    x = vmap(poisson)(l)
    expected = numpyro_dist.Poisson(l).mean

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < thresh)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_categorical_iid():
    p = jnp.array([0.1, 0.2, 0.7])
    x = vmap(categorical, None, 5)(p)

    expected = p @ np.arange(3)

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < thresh)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_categorical_0():
    p = jnp.array(np.random.rand(5, 3))
    p = p / jnp.sum(p, axis=1, keepdims=True)
    x = vmap(categorical)(p)

    expected = p @ np.arange(3)

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < thresh)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_categorical_1():
    p = jnp.array(np.random.rand(3, 5))
    p = p / jnp.sum(p, axis=0, keepdims=True)
    x = vmap(categorical, 1)(p)

    expected = np.arange(3) @ p

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < thresh)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_binomial_iid():
    n = 10
    p = 0.5
    x = vmap(binomial, None, axis_size=3)(n, p)
    expected = numpyro_dist.Binomial(n, p).mean

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < 0.05)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_binomial_0_None():
    n = jnp.array([10, 20, 30])
    p = 0.5
    x = vmap(binomial, [0, None], axis_size=3)(n, p)
    expected = numpyro_dist.Binomial(n, p).mean

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < 0.05)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_binomial_None_0():
    n = 10
    p = jnp.array([0.2, 0.3, 0.5])
    x = vmap(binomial, [None, 0], axis_size=3)(n, p)
    expected = numpyro_dist.Binomial(n, p).mean

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < 0.05)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_beta_binomial_iid():
    n = 10
    alpha = 0.9
    beta = 1.2
    x = vmap(beta_binomial, None, axis_size=3)(n, alpha, beta)
    # watch out! numpyro uses other parameterization!
    expected = numpyro_dist.BetaBinomial(alpha, beta, n).mean

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < 0.05)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_beta_binomial_0_None_0():
    n = jnp.array([10, 5, 15])
    alpha = 0.9
    beta = jnp.array([1.2, 1.5, 1.9])
    x = vmap(beta_binomial, [0, None, 0], axis_size=3)(n, alpha, beta)
    # watch out! numpyro uses other parameterization!
    expected = numpyro_dist.BetaBinomial(alpha, beta, n).mean

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < 0.05)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_multinomial_iid():
    n = 7
    p = jnp.array([0.1, 0.2, 0.7])
    x = vmap(multinomial, None, axis_size=5)(n, p)
    expected = numpyro_dist.Multinomial(n, p).mean

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < 0.05)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_multinomial_None_0():
    n = 7
    p = jnp.array(np.random.rand(5, 3))
    p = p / jnp.sum(p, axis=1, keepdims=True)
    x = vmap(multinomial, [None,0], axis_size=5)(n, p)
    expected = numpyro_dist.Multinomial(n, p).mean

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < 0.05)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_multinomial_None_1():
    n = 7
    p = jnp.array(np.random.rand(3, 5))
    p = p / jnp.sum(p, axis=0, keepdims=True)
    x = vmap(multinomial, [None,1], axis_size=5)(n, p)
    expected = numpyro_dist.Multinomial(n, p.T).mean

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < 0.05)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_multinomial_0_0():
    n = jnp.array([5,3,3,2,9])
    p = jnp.array(np.random.rand(5, 3))
    p = p / jnp.sum(p, axis=1, keepdims=True)
    x = vmap(multinomial, [0,0], axis_size=5)(n, p)
    expected = numpyro_dist.Multinomial(n, p).mean

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < 0.05)

    inf_until_match(E, x, [], [], testfun)

def test_vmap_multinomial_0_1():
    n = jnp.array([5,3,3,2,9])
    p = jnp.array(np.random.rand(3, 5))
    p = p / jnp.sum(p, axis=0, keepdims=True)
    x = vmap(multinomial, [0,1], axis_size=5)(n, p)
    expected = numpyro_dist.Multinomial(n, p.T).mean

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < 0.05)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_multinomial_0_None():
    n = jnp.array([5,3,3,2,9])
    p = jnp.array(np.random.rand(3))
    p = p / jnp.sum(p)
    x = vmap(multinomial, [0,None], axis_size=5)(n, p)
    expected = numpyro_dist.Multinomial(n, p).mean

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < 0.05)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_numpyro_pars1():
    op = ir.VMap(ir.Add(), (0, None))
    a = jnp.array([1,2,3])
    b = jnp.array(4)
    new_a, new_b = vmap_numpyro_pars(op, a, b)
    assert jnp.all(new_a == jnp.array([1, 2, 3]))
    assert jnp.all(new_b == jnp.array([4, 4, 4]))

def test_vmap_numpyro_pars2():
    op = ir.VMap(ir.Add(), (None, None), 3)
    a = jnp.array(1)
    b = jnp.array(4)
    new_a, new_b = vmap_numpyro_pars(op, a, b)
    assert jnp.all(new_a == jnp.array([1, 1, 1]))
    assert jnp.all(new_b == jnp.array([4, 4, 4]))

def test_vmap_numpyro_pars3():
    op = ir.VMap(ir.VMap(ir.Add(), (0, None)),(None,0))
    a = jnp.array([1,2,3])
    b = jnp.array([4,5,6])
    new_a, new_b = vmap_numpyro_pars(op, a, b)
    assert jnp.all(new_a == jnp.array([[1, 2, 3]]*3))
    assert jnp.all(new_b == jnp.array([[4]*3, [5]*3, [6]*3]))


def test_vmap_bernoulli_2d_iid_iid():
    x = vmap(vmap(bernoulli, None, 3), None, axis_size=4)(0.5)

    def testfun(Ex):
        assert Ex.shape == (4,3)
        return np.all(np.abs(Ex - 0.5) < thresh)

    inf_until_match(E, x, [], [], testfun)


def test_vmap_bernoulli_2d_0_None():
    p = jnp.array([.1, .2, .7])
    x = vmap(vmap(bernoulli, 0, 3), None, axis_size=4)(p)

    expected = numpyro_dist.Bernoulli(p).mean

    def testfun(Ex):
        assert Ex.shape == (4,3)
        return np.all(np.abs(Ex - expected) < thresh)

    inf_until_match(E, x, [], [], testfun)

def test_vmap_bernoulli_2d_None_0():
    p = jnp.array([.1, .2, .3, .6])
    x = vmap(vmap(bernoulli, None, 3), 0, axis_size=4)(p)

    expected = jax.vmap(jax.vmap(lambda p : numpyro_dist.Bernoulli(p).mean, None, axis_size=3), 0, axis_size=4)(p)

    def testfun(Ex):
        assert Ex.shape == (4,3)
        return np.all(np.abs(Ex - expected) < thresh)

    inf_until_match(E, x, [], [], testfun)

def test_vmap_bernoulli_2d_0_0():
    p = jnp.array(np.random.rand(4,3))
    x = vmap(vmap(bernoulli, 0, 3), 0, axis_size=4)(p)

    expected = jax.vmap(jax.vmap(lambda p : numpyro_dist.Bernoulli(p).mean, 0, axis_size=3), 0, axis_size=4)(p)

    def testfun(Ex):
        assert Ex.shape == (4,3)
        return np.all(np.abs(Ex - expected) < thresh)

    inf_until_match(E, x, [], [], testfun)

def test_vmap_multinomial_2d_None_None():
    n = jnp.array(10)
    p = jnp.array(np.random.rand(3))
    p = p / jnp.sum(p)
    x = vmap(vmap(multinomial, None, axis_size=4), None, axis_size=5)(n, p)
    expected = numpyro_dist.Multinomial(n, p).mean

    def testfun(Ex):
        return np.all(np.abs(Ex - expected) < thresh)

    inf_until_match(E, x, [], [], testfun)

def test_vmap_multinomial_2d_None_0():
    n = jnp.array(10)
    p = jnp.array(np.random.rand(5,3))
    p = p / jnp.sum(p, axis=1, keepdims=True)
    x = vmap(vmap(multinomial, None, axis_size=4), [None,0], axis_size=5)(n, p)
    fun = lambda n, p: numpyro_dist.Multinomial(n, p).mean
    expected = jax.vmap(jax.vmap(fun, None, axis_size=4), [None,0], axis_size=5)(n, p)

    def testfun(Ex):
        assert Ex.shape == (5, 4, 3)
        return np.all(np.abs(Ex - expected) < thresh)

    inf_until_match(E, x, [], [], testfun)

