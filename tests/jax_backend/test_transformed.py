import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from pangolin import functional, ir
from pangolin.jax_backend import (
    ancestor_log_prob,
    ancestor_sample_graph,
    lower,
    require_array,
    require_distribution,
)


def exp_bijector() -> ir.Bijector:
    return ir.Bijector(ir.Exp(), ir.Log(), ir.Composite(2, (ir.Identity(),), ((0,),)))


def vector_exp_bijector() -> ir.Bijector:
    return ir.Bijector(
        ir.VMap(ir.Exp(), (0,)),
        ir.VMap(ir.Log(), (0,)),
        ir.Composite(2, (ir.Sum(0),), ((0,),)),
    )


def affine_bijector() -> ir.Bijector:
    return ir.Bijector(
        ir.Composite(3, (ir.Mul(), ir.Add()), ((0, 1), (3, 2))),
        ir.Composite(3, (ir.Sub(), ir.Div()), ((0, 2), (3, 1))),
        ir.Composite(4, (ir.Abs(), ir.Log()), ((2,), (4,))),
        n_biject_params=2,
    )


def test_exp_normal_sample_preserves_the_base_key():
    key = jax.random.PRNGKey(12)
    op = ir.Bijected(ir.Normal(), exp_bijector())
    actual = require_distribution(lower(op, (0.3, 0.8))).sample(key)
    expected = jnp.exp(dist.Normal(0.3, 0.8).sample(key))
    assert jnp.array_equal(actual, expected)


@pytest.mark.parametrize("value", [0.1, 0.7, 1.0, 4.0])
def test_exp_normal_log_density(value):
    op = ir.Bijected(ir.Normal(), exp_bijector())
    distribution = require_distribution(lower(op, (0.3, 0.8)))
    assert jnp.allclose(distribution.log_prob(value), dist.LogNormal(0.3, 0.8).log_prob(value))


@pytest.mark.parametrize("scale", [2.5, -2.5])
def test_parameterized_affine_transform_splits_parents_and_uses_absolute_determinant(scale):
    op = ir.Bijected(ir.Normal(), affine_bijector())
    offset, loc, std = 4.0, -0.7, 0.3
    distribution = require_distribution(lower(op, (scale, offset, loc, std)))
    key = jax.random.PRNGKey(13)
    expected_sample = scale * dist.Normal(loc, std).sample(key) + offset
    expected_density = dist.Normal(scale * loc + offset, abs(scale) * std).log_prob(3.0)
    assert jnp.allclose(distribution.sample(key), expected_sample)
    assert jnp.allclose(distribution.log_prob(3.0), expected_density)


def test_nested_transformed_ops_cancel_exp_and_log():
    log_bijector = ir.Bijector(
        ir.Log(),
        ir.Exp(),
        ir.Composite(2, (ir.Log(), ir.Constant(-1), ir.Mul()), ((0,), (), (2, 3))),
    )
    op = ir.Bijected(ir.Bijected(ir.Normal(), exp_bijector()), log_bijector)
    distribution = require_distribution(lower(op, (0.3, 0.8)))
    key = jax.random.PRNGKey(14)
    assert jnp.allclose(distribution.sample(key), dist.Normal(0.3, 0.8).sample(key))
    assert jnp.allclose(distribution.log_prob(-0.4), dist.Normal(0.3, 0.8).log_prob(-0.4))


@pytest.mark.parametrize("transformed_outside", [False, True])
def test_transformed_and_composite_nest_in_both_orders(transformed_outside):
    if transformed_outside:
        base = ir.Composite(2, (ir.Add(), ir.Normal()), ((0, 1), (2, 1)))
        op = ir.Bijected(base, exp_bijector())
    else:
        op = ir.Composite(2, (ir.Add(), ir.Bijected(ir.Normal(), exp_bijector())), ((0, 1), (2, 1)))
    distribution = require_distribution(lower(op, (0.3, 0.8)))
    key = jax.random.PRNGKey(15)
    assert jnp.allclose(distribution.sample(key), jnp.exp(dist.Normal(1.1, 0.8).sample(key)))
    assert jnp.allclose(distribution.log_prob(2.0), dist.LogNormal(1.1, 0.8).log_prob(2.0))


@pytest.mark.parametrize("transformed_outside", [False, True])
def test_transformed_and_vmap_nest_in_both_orders(transformed_outside):
    if transformed_outside:
        op = ir.Bijected(ir.VMap(ir.Normal(), (0, None)), vector_exp_bijector())
    else:
        op = ir.VMap(ir.Bijected(ir.Normal(), exp_bijector()), (0, None))
    loc = jnp.array([-0.2, 0.1, 0.4])
    distribution = require_distribution(lower(op, (loc, 0.8)))
    key = jax.random.PRNGKey(16)
    expected_sample = jax.vmap(lambda k, m: jnp.exp(dist.Normal(m, 0.8).sample(k)))(
        jax.random.split(key, 3), loc
    )
    value = jnp.array([0.5, 1.0, 1.5])
    expected_density = jnp.sum(dist.LogNormal(loc, 0.8).log_prob(value))
    assert jnp.allclose(distribution.sample(key), expected_sample)
    assert distribution.log_prob(value).shape == ()
    assert jnp.allclose(distribution.log_prob(value), expected_density)


@pytest.mark.parametrize("transformed_outside", [False, True])
def test_transformed_and_scan_use_the_correct_carry_space(transformed_outside):
    start, std = -0.8, 0.1
    if transformed_outside:
        op = ir.Bijected(ir.Scan(ir.Normal(), 3, (None,)), vector_exp_bijector())
    else:
        op = ir.Scan(ir.Bijected(ir.Normal(), exp_bijector()), 3, (None,))
    distribution = require_distribution(lower(op, (start, std)))
    key = jax.random.PRNGKey(17)
    previous = start
    expected_samples = []
    for subkey in jax.random.split(key, 3):
        x = dist.Normal(previous, std).sample(subkey)
        y = jnp.exp(x)
        expected_samples.append(y)
        previous = x if transformed_outside else y
    assert jnp.allclose(distribution.sample(key), jnp.array(expected_samples))

    values = jnp.array([0.4, 0.5, 0.6])
    previous = start
    expected_density = 0.0
    for y in values:
        x = jnp.log(y)
        expected_density += dist.Normal(previous, std).log_prob(x) - x
        previous = x if transformed_outside else y
    assert jnp.allclose(distribution.log_prob(values), expected_density)


def test_cholesky_transformed_wishart_density_and_matrix_sample():
    df, scale = ir.RV(ir.Constant(5.0)), ir.RV(ir.Constant(jnp.eye(2)))
    transformed = functional.tforms.cholesky(lambda df, scale: ir.RV(ir.Wishart(), df, scale))(
        df, scale
    )
    distribution = require_distribution(lower(transformed.op, (5.0, jnp.eye(2))))
    base = dist.Wishart(5.0, jnp.eye(2))
    key = jax.random.PRNGKey(18)
    assert jnp.allclose(distribution.sample(key), jnp.linalg.cholesky(base.sample(key)))

    y = jnp.array([[1.2, 0.0], [0.3, 0.8]])
    inverse_ldj = 2 * jnp.log(2.0) + 2 * jnp.log(y[0, 0]) + jnp.log(y[1, 1])
    expected = base.log_prob(y @ y.T) + inverse_ldj
    assert jnp.allclose(distribution.log_prob(y), expected)


@pytest.mark.parametrize(
    "transform",
    [
        functional.tforms.stick_breaking,
        functional.tforms.softmax_centered,
        functional.tforms.softmax_centered_ilr,
    ],
)
def test_shape_changing_dirichlet_transform(transform):
    alpha = jnp.array([2.0, 3.0, 4.0])
    transformed = transform(lambda a: ir.RV(ir.Dirichlet(), a))(ir.RV(ir.Constant(alpha)))
    assert isinstance(transformed.op, ir.Bijected)
    bijector = transformed.op.bijector
    distribution = require_distribution(lower(transformed.op, (alpha,)))
    key = jax.random.PRNGKey(19)
    base_sample = dist.Dirichlet(alpha).sample(key)
    expected_sample = require_array(lower(bijector.forward, (base_sample,)))
    assert distribution.sample(key).shape == (2,)
    assert jnp.allclose(distribution.sample(key), expected_sample)

    def forward_from_chart(head):
        x = jnp.concatenate([head, jnp.array([1 - jnp.sum(head)])])
        return require_array(lower(bijector.forward, (x,)))

    x = jnp.array([0.2, 0.3, 0.5])
    y = forward_from_chart(x[:-1])
    ldj = jnp.linalg.slogdet(jax.jacfwd(forward_from_chart)(x[:-1])).logabsdet
    assert jnp.allclose(distribution.log_prob(y), dist.Dirichlet(alpha).log_prob(x) - ldj)


def test_transformed_sample_and_density_are_jittable():
    op = ir.Bijected(ir.Normal(), exp_bijector())
    distribution = require_distribution(lower(op, (0.3, 0.8)))
    key = jax.random.PRNGKey(20)
    assert jnp.allclose(jax.jit(distribution.sample)(key), distribution.sample(key))
    assert jnp.allclose(jax.jit(distribution.log_prob)(1.2), distribution.log_prob(1.2))


@pytest.mark.parametrize("scale", [2.0, -2.0])
def test_gradients_include_values_base_parameters_and_bijector_parameters(scale):
    op = ir.Bijected(ir.Normal(), affine_bijector())

    def actual(y, scale, offset, loc, std):
        return require_distribution(lower(op, (scale, offset, loc, std))).log_prob(y)

    def expected(y, scale, offset, loc, std):
        return dist.Normal(scale * loc + offset, jnp.abs(scale) * std).log_prob(y)

    args = (0.7, scale, -0.2, 0.3, 0.8)
    actual_grad = jax.jit(jax.grad(actual, argnums=(0, 1, 2, 3, 4)))(*args)
    expected_grad = jax.grad(expected, argnums=(0, 1, 2, 3, 4))(*args)
    assert jnp.allclose(jnp.array(actual_grad), jnp.array(expected_grad))


def test_graph_density_counts_random_bijector_parameters_once():
    zero, one = ir.RV(ir.Constant(0.0)), ir.RV(ir.Constant(1.0))
    scale = ir.RV(ir.Constant(2.0))
    offset = ir.RV(ir.Normal(), zero, one)
    y = ir.RV(ir.Bijected(ir.Normal(), affine_bijector()), scale, offset, zero, one)
    log_prob = ancestor_log_prob(offset, y)
    expected = dist.Normal(0.0, 1.0).log_prob(0.3) + dist.Normal(0.3, 2.0).log_prob(1.5)
    assert jnp.allclose(jax.jit(log_prob)(0.3, 1.5), expected)
    offset_value, y_value = ancestor_sample_graph((offset, y), jax.random.PRNGKey(21))
    assert jnp.isfinite(log_prob(offset_value, y_value))


def test_transformed_log_density_rejects_the_wrong_value_shape():
    op = ir.Bijected(ir.VMap(ir.Normal(), (0, None)), vector_exp_bijector())
    distribution = require_distribution(lower(op, (jnp.zeros(3), 1.0)))
    with pytest.raises(AssertionError, match="shape was not as expected"):
        distribution.log_prob(jnp.ones(2))


def test_transformed_rejects_an_unsupported_bijector_op():
    class UnsupportedOp(ir.Op):
        random: bool = False
        """False"""

        @staticmethod
        def get_shape(shape):
            return shape

    bijector = ir.Bijector(
        UnsupportedOp(), ir.Identity(), ir.Composite(2, (ir.Constant(0),), ((),))
    )
    distribution = require_distribution(lower(ir.Bijected(ir.Normal(), bijector), (0.0, 1.0)))
    with pytest.raises(NotImplementedError, match="UnsupportedOp"):
        distribution.sample(jax.random.PRNGKey(22))


@pytest.mark.parametrize("with_sign", [False, True])
def test_interior_density_and_gradients_do_not_require_sign(with_sign):
    original = exp_bijector()
    bijector = ir.Bijector(
        original.forward,
        original.inverse,
        original.log_det_jac,
        sign=ir.Constant(1) if with_sign else None,
    )
    op = ir.Bijected(ir.Normal(), bijector)
    distribution = require_distribution(lower(op, (0.3, 0.8)))
    values = jnp.array([0.5, 1.0, 2.0, jnp.nan])
    densities = jax.jit(jax.vmap(distribution.log_prob))(values)
    assert jnp.isnan(densities[-1])
    assert jnp.allclose(densities[:3], dist.LogNormal(0.3, 0.8).log_prob(values[:3]))

    def density(y, loc, scale):
        return require_distribution(lower(op, (loc, scale))).log_prob(y)

    gradients = jax.jit(jax.vmap(jax.grad(density, (0, 1, 2)), (0, None, None)))(
        jnp.array([0.5, 1.0, 2.0]), 0.3, 0.8
    )
    for gradient in gradients:
        assert jnp.all(jnp.isfinite(gradient))
    key = jax.random.key(0)
    assert jnp.array_equal(distribution.sample(key), jnp.exp(dist.Normal(0.3, 0.8).sample(key)))


def test_bijected_density_respects_finite_open_output_interval():
    bijector = ir.Bijector(
        ir.InvLogit(),
        ir.Logit(),
        ir.Composite(
            2, (ir.Constant(1.0), ir.Sub(), ir.Mul(), ir.Log()), ((), (2, 1), (1, 3), (4,))
        ),
    )
    distribution = require_distribution(lower(ir.Bijected(ir.Normal(), bijector), (0.0, 1.0)))
    values = jnp.array([0.5])
    densities = jax.jit(jax.vmap(distribution.log_prob))(values)
    expected = dist.Normal(0.0, 1.0).log_prob(0.0) - jnp.log(0.25)
    assert jnp.allclose(densities, jnp.array([expected]))
    assert jnp.all(jax.jit(jax.vmap(jax.grad(distribution.log_prob)))(values) == 0.0)


def test_vector_bijection_density():
    original = vector_exp_bijector()
    bijector = ir.Bijector(
        original.forward,
        original.inverse,
        original.log_det_jac,
    )
    op = ir.Bijected(ir.VMap(ir.Normal(), (0, None)), bijector)
    loc = jnp.array([-0.3, 0.5])
    distribution = require_distribution(lower(op, (loc, 0.8)))
    values = jnp.array([0.5, 1.5])
    actual = jax.jit(distribution.log_prob)(values)
    assert actual.shape == ()
    assert jnp.allclose(actual, dist.LogNormal(loc, 0.8).log_prob(values).sum())
