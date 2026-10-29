from pangolin.jax_backend import (
    ancestor_sample_flat,
    lower,
    require_array,
    require_distribution,
)
import numpy as np
from pangolin import ir

import jax
import numpyro.distributions
from jax import numpy as jnp

# handle_vmap_nonrandom(op: ir.VMap, *numpyro_parents, is_observed):


def test_handle_nonrandom_exp():
    op = ir.VMap(ir.Exp(), (None,), 5)
    x = jnp.array(1.0)
    out = require_array(lower(op, [x]))
    expected = jnp.exp(1.0) * jnp.ones(5)
    assert jnp.allclose(out, expected)

    y = ir.RV(op, ir.RV(ir.Constant(x)))
    [out2] = ancestor_sample_flat([y], None)
    assert jnp.allclose(out2, expected)


def test_handle_nonrandom_exp_2d():
    op = ir.VMap(ir.VMap(ir.Exp(), (None,), 5), (0,))
    x = jnp.array([1.0, 2.0])
    out = require_array(lower(op, [x]))
    expected = jnp.exp(jnp.array([1.0, 2.0]))[:, None] * jnp.ones((2, 5))
    assert jnp.allclose(out, expected)


def test_handle_nonrandom_add():
    op = ir.VMap(ir.Add(), (None, 0))
    x = jnp.array(1.0)
    y = jnp.array([2.0, 3.0, 4.0])
    z = require_array(lower(op, [x, y]))
    expected = x + y
    assert jnp.allclose(z, expected)


def test_handle_nonrandom_add_2d():
    op = ir.VMap(ir.VMap(ir.Add(), (None, 0)), (None, 0))
    x = jnp.array(1.0)
    y = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    z = require_array(lower(op, [x, y]))
    expected = x + y
    assert jnp.allclose(z, expected)


def test_normal_iid():
    op = ir.VMap(ir.Normal(), (None, None), 5)
    mu = jnp.array(1.0)
    sigma = jnp.array(1.0)
    key = jax.random.PRNGKey(0)

    def my_sample(key):
        return numpyro.distributions.Normal(mu, sigma).sample(key)

    y = require_distribution(lower(op, [mu, sigma])).sample(key)
    subkey = jax.random.split(key, 5)
    expected = jax.vmap(my_sample)(subkey)
    assert jnp.allclose(y, expected)

    def my_log_prob(v):
        return numpyro.distributions.Normal(mu, sigma).log_prob(v)

    l = require_distribution(lower(op, [mu, sigma])).log_prob(y)
    expected_l = jnp.sum(jax.vmap(my_log_prob)(y))
    assert jnp.allclose(l, expected_l)


def test_normal_non_iid():
    op = ir.VMap(ir.Normal(), (0, 0))
    mu = jnp.array([1.0, 2.0, 3.0])
    sigma = jnp.array([3.0, 2.0, 1.0])
    key = jax.random.PRNGKey(0)

    def my_sample(key, mu, sigma):
        return numpyro.distributions.Normal(mu, sigma).sample(key)

    y = require_distribution(lower(op, [mu, sigma])).sample(key)
    subkey = jax.random.split(key, 3)
    expected = jax.vmap(my_sample)(subkey, mu, sigma)
    assert jnp.allclose(y, expected)

    def my_log_prob(v, mu, sigma):
        return numpyro.distributions.Normal(mu, sigma).log_prob(v)

    l = require_distribution(lower(op, [mu, sigma])).log_prob(y)
    expected_l = jnp.sum(jax.vmap(my_log_prob)(y, mu, sigma))
    assert jnp.allclose(l, expected_l)


def test_normal_iid_2d():
    op = ir.VMap(ir.VMap(ir.Normal(), (None, None), 5), (None, None), 3)
    mu = jnp.array(1.0)
    sigma = jnp.array(1.0)
    key = jax.random.PRNGKey(0)
    y = require_distribution(lower(op, [mu, sigma])).sample(key)

    def my_sample0(key):
        return numpyro.distributions.Normal(mu, sigma).sample(key)

    def my_sample1(key):
        return jax.vmap(my_sample0)(jax.random.split(key, 5))

    subkey = jax.random.split(key, 3)
    expected = jax.vmap(my_sample1)(subkey)

    assert jnp.allclose(y, expected)
