from pangolin.jax_backend import (
    # get_handler,
    # NumpyroHandler,
    # DeterministicHandler,
    log_prob_op,
    sample_op,
    eval_op,
    ancestor_sample_flat,
    ancestor_sample,
)
from pangolin import ir
import numpyro.distributions
import numpy as np
import jax
from pangolin.util import tree_allclose


def test_constant():
    op = ir.Constant(2.0)
    out = eval_op(op, [])
    assert np.allclose(out, 2.0)


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

    out = eval_op(op, parent_values)
    expected = 1.6
    assert np.allclose(out, expected)


def test_ancestor_sample_flat():
    a = ir.RV(ir.Constant(3.0))
    b = ir.RV(ir.Constant(0.0))
    c = ir.RV(ir.Normal(), a, b)

    key = jax.random.PRNGKey(0)
    out = ancestor_sample_flat([a, b, c], key)
    assert np.allclose(out, [3.0, 0.0, 3.0])


def test_ancestor_sample():
    a = ir.RV(ir.Constant(3.0))
    b = ir.RV(ir.Constant(0.0))
    c = ir.RV(ir.Normal(), a, b)

    key = jax.random.PRNGKey(0)
    out1 = ancestor_sample(c, key)
    assert np.allclose(out1, 3.0)

    out2 = ancestor_sample({"dog": c}, key)
    assert tree_allclose(out2, {"dog": 3.0})

    out3 = ancestor_sample([a, {"dog": (b, c)}], key)
    assert tree_allclose(out3, [3.0, {"dog": (0.0, 3.0)}])
