import pytest
import jax
import jax.numpy as jnp
import numpy as np
from pangolin import ir

# Enable float64 for highly precise exact Jacobian testing
jax.config.update("jax_enable_x64", True)

from pangolin.jax_backend import (
    log_diagonal_bijector,
    log_prob_op,
    sample_op,
    eval_op,
    constrain_op,
    log_bijector,
    scaled_logit_bijector,
    cholesky_bijector,
    unconstrain_spd_bijector,
    logit_bijector,
    compose_jax_bijectors,
    unconstrained_log_prob_op,
    unconstrained_sample_op,
    default_bijector_dict,
)

bijector_dict = default_bijector_dict


def _test_op(op, parent_values):
    key = jax.random.PRNGKey(1)
    out = sample_op(op, key, parent_values)
    out_bijected, out2 = unconstrained_sample_op(op, key, parent_values, bijector_dict)
    out_constrained = constrain_op(op, out_bijected, parent_values, bijector_dict)
    assert jnp.allclose(out, out_constrained)
    assert jnp.allclose(out, out2)
    log_prob = log_prob_op(op, out, parent_values)
    log_prob_bijected = unconstrained_log_prob_op(op, out_bijected, parent_values, bijector_dict)


def test_normal_op():
    _test_op(ir.Normal(), [0.0, 1.0])


def test_lognormal_op():
    _test_op(ir.Lognormal(), [0.0, 1.0])


def test_uniform_op():
    _test_op(ir.Uniform(), [-3.1, 1.5])


def test_wishart_op():
    _test_op(ir.Wishart(), [2.5, jnp.eye(3)])


def test_beta_op():
    _test_op(ir.Beta(), [2.5, 3.1])


def test_vmap_lognormal_0_0_op():
    op = ir.VMap(ir.Lognormal(), [0, 0])
    parent_values = [jnp.array([-1.0, 1.5]), jnp.array([0.5, 1.5])]
    _test_op(op, parent_values)


def test_vmap_lognormal_0_None_op():
    op = ir.VMap(ir.Lognormal(), [0, None])
    parent_values = [jnp.array([-1.0, 1.5]), 1.5]
    _test_op(op, parent_values)


def test_vmap_lognormal_None_None_op():
    op = ir.VMap(ir.Lognormal(), [None, None], axis_size=5)
    parent_values = [-jnp.array(1.5), jnp.array(3.0)]
    _test_op(op, parent_values)


def test_double_vmap_lognormal_op():
    op = ir.VMap(ir.VMap(ir.Lognormal(), [0, None]), [None, 0])
    parent_values = [-jnp.array([1.5, 2.7]), jnp.array([3.0, 5.0, 7.0])]
    _test_op(op, parent_values)


def test_exponential():
    _test_op(ir.Exponential(), [1.1])


def test_scan_exponential():
    _test_op(ir.Scan(ir.Exponential(), 5, []), [1.1])
