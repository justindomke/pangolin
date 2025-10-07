from pangolin.backend.handlers import (
    # get_handler,
    # NumpyroHandler,
    # DeterministicHandler,
    log_prob_op,
    sample_op,
    ancestor_sample_flat,
)
from pangolin import ir
import numpyro.distributions
import numpy as np
import jax


# def test_normal_handler():
#     op = ir.Normal()
#     handler = get_handler(op)
#     assert isinstance(handler, NumpyroHandler)

#     parent_values = [0.5, 1.1]
#     value = -0.3
#     log_prob = handler.log_prob(value, parent_values)
#     expected = numpyro.distributions.Normal(*parent_values).log_prob(value)
#     assert np.allclose(log_prob, expected)

#     key = jax.random.PRNGKey(5)
#     out = handler.sample(key, parent_values)
#     expected = numpyro.distributions.Normal(*parent_values).sample(key)
#     assert np.allclose(out, expected)


# def test_add_handler():
#     op = ir.Add()
#     handler = get_handler(op)
#     assert isinstance(handler, DeterministicHandler)

#     parent_values = [0.5, 1.1]
#     key = jax.random.PRNGKey(5)
#     out = handler.sample(key, parent_values)
#     expected = 1.6
#     assert np.allclose(out, expected)


def test_normal():
    op = ir.Normal()
    parent_values = [0.5, 1.1]
    value = -0.3

    log_prob = log_prob_op(op, value, parent_values)
    expected = numpyro.distributions.Normal(*parent_values).log_prob(value)
    assert np.allclose(log_prob, expected)

    key = jax.random.PRNGKey(3)
    out = sample_op(op, key, parent_values)
    expected = numpyro.distributions.Normal(*parent_values).sample(key)
    assert np.allclose(out, expected)


def test_add():
    op = ir.Add()
    parent_values = [0.5, 1.1]

    key = None
    out = sample_op(op, key, parent_values)
    expected = 1.6
    assert np.allclose(out, expected)


def test_ancestor_sample_flat():
    a = ir.RV(ir.Constant(3.0))
    b = ir.RV(ir.Constant(0.0))
    c = ir.RV(ir.Normal(), a, b)

    key = jax.random.PRNGKey(0)
    out = ancestor_sample_flat([a, b, c], key)
    assert np.allclose(out, 3.0)
