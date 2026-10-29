import jax
import jax.numpy as jnp
import pytest

from pangolin import ir
from pangolin.jax_backend import ancestor_sample_graph, lower, require_array


@pytest.mark.parametrize(("argnum", "expected"), [(0, 12.0), (1, 8.0)])
def test_scalar_multiplication(argnum: int, expected: float):
    actual = require_array(lower(ir.VJP(ir.Mul(), argnum), (4.0, 2.0, 3.0)))
    assert actual.shape == ()
    assert jnp.allclose(actual, expected)


def test_cotangent_is_a_fixed_weight_not_another_differentiated_expression():
    x = ir.RV(ir.Constant(3.0))
    result = ir.RV(ir.VJP(ir.Mul(), 0), x, x, x)
    actual = ancestor_sample_graph(result, None)

    assert isinstance(actual, jax.Array)
    assert jnp.allclose(actual, 9.0)


@pytest.mark.parametrize("argnum", [0, 1])
def test_matrix_product(argnum: int):
    left = jnp.arange(6.0).reshape(2, 3)
    right = jnp.arange(12.0).reshape(3, 4)
    cotangent = jnp.arange(8.0).reshape(2, 4) - 2
    actual = require_array(lower(ir.VJP(ir.Matmul(), argnum), (cotangent, left, right)))
    expected = cotangent @ right.T if argnum == 0 else left.T @ cotangent

    assert actual.shape == (left if argnum == 0 else right).shape
    assert jnp.allclose(actual, expected)


def test_array_output_sums_weighted_components():
    value = jnp.arange(6.0).reshape(2, 3)
    actual = require_array(lower(ir.VJP(ir.Sum(1), 0), (jnp.array([2.0, -3.0]), value)))
    assert jnp.allclose(actual, jnp.array([[2.0, 2.0, 2.0], [-3.0, -3.0, -3.0]]))


def test_composite():
    base = ir.Composite(2, (ir.Mul(), ir.Exp()), ((0, 1), (2,)))
    actual = require_array(lower(ir.VJP(base, 0), (2.0, 0.3, 0.8)))
    assert jnp.allclose(actual, 2.0 * 0.8 * jnp.exp(0.3 * 0.8))


@pytest.mark.parametrize("argnum", [0, 1])
def test_vmap_including_shared_input(argnum: int):
    values = jnp.array([1.0, 2.0, 4.0])
    cotangent = jnp.array([2.0, -1.0, 3.0])
    base = ir.VMap(ir.Mul(), (0, None))
    actual = require_array(lower(ir.VJP(base, argnum), (cotangent, values, 0.5)))
    expected = 0.5 * cotangent if argnum == 0 else jnp.sum(values * cotangent)

    assert jnp.allclose(actual, expected)


@pytest.mark.parametrize("argnum", [0, 1])
def test_scan_including_carry_dependence(argnum: int):
    increments = jnp.array([0.2, -0.1, 0.4])
    cotangent = jnp.array([2.0, -1.0, 3.0])
    base = ir.Scan(ir.Add(), 3, (0,))
    actual = require_array(lower(ir.VJP(base, argnum), (cotangent, 0.5, increments)))
    expected = cotangent.sum() if argnum == 0 else jnp.cumsum(cotangent[::-1])[::-1]

    assert jnp.allclose(actual, expected)


def test_nested_scan_vmap_composite():
    step = ir.Composite(2, (ir.Mul(), ir.Add()), ((0, 1), (2, 1)))
    base = ir.Scan(ir.VMap(step, (0, 0)), 3, (0,))
    start = jnp.array([0.3, 0.5])
    inputs = jnp.array([[0.5, 0.2], [0.3, 0.4], [0.2, 0.6]])
    cotangent = jnp.array([[1.0, -2.0], [3.0, 0.5], [-1.0, 2.0]])
    actual = require_array(lower(ir.VJP(base, 0), (cotangent, start, inputs)))
    expected = jnp.sum(jnp.cumprod(inputs, axis=0) * cotangent, axis=0)

    assert jnp.allclose(actual, expected)


@pytest.mark.parametrize("argnum", [0, 1, 2])
def test_normal_log_prob(argnum: int):
    value, location, scale, weight = 0.4, 0.3, 0.8, 2.0
    actual = require_array(
        lower(ir.VJP(ir.LogProb(ir.Normal()), argnum), (weight, value, location, scale))
    )
    expected = [
        -(value - location) / scale**2,
        (value - location) / scale**2,
        -1 / scale + (value - location) ** 2 / scale**3,
    ][argnum] * weight

    assert jnp.allclose(actual, expected)


def test_log_prob_of_random_scan():
    values = jnp.array([0.2, -0.1, 0.4])
    base = ir.LogProb(ir.Scan(ir.Normal(), 3, (None,)))
    actual = require_array(lower(ir.VJP(base, 1), (2.0, values, 0.3, 0.8)))
    assert jnp.allclose(actual, 2.0 * (values[0] - 0.3) / 0.8**2)


def test_discrete_value_is_allowed_when_differentiating_a_log_mass_parameter():
    actual = require_array(lower(ir.VJP(ir.LogProb(ir.Bernoulli()), 1), (2.0, 1, 0.4)))
    assert jnp.allclose(actual, 2.0 / 0.4)


def test_integer_indices_are_held_fixed():
    actual = require_array(lower(ir.VJP(ir.Index(), 0), (2, jnp.array([2.0, 3.0, 4.0]), 1)))
    assert jnp.allclose(actual, jnp.array([0.0, 2.0, 0.0]))


def test_integer_cotangent_is_converted_to_output_dtype():
    actual = require_array(lower(ir.VJP(ir.Exp(), 0), (2, 0.3)))
    assert jnp.allclose(actual, 2 * jnp.exp(0.3))


def test_nested_vjp_gives_second_derivative():
    op = ir.VJP(ir.VJP(ir.Pow(), 0), 1)
    actual = require_array(lower(op, (5.0, 4.0, 2.0, 3.0)))
    assert jnp.allclose(actual, 6.0 * 2.0 * 4.0 * 5.0)


def test_nested_vjp_can_differentiate_the_inner_cotangent():
    op = ir.VJP(ir.VJP(ir.Pow(), 0), 0)
    actual = require_array(lower(op, (5.0, 4.0, 2.0, 3.0)))
    assert jnp.allclose(actual, 5.0 * 3.0 * 2.0**2)


def test_vjp_can_be_jitted_vmapped_and_differentiated():
    def derivative(x, scale, cotangent):
        return require_array(lower(ir.VJP(ir.Mul(), 0), (cotangent, x, scale)))

    assert jnp.allclose(jax.jit(derivative)(2.0, 3.0, 4.0), 12.0)
    gradients = jax.grad(derivative, argnums=(0, 1, 2))(2.0, 3.0, 4.0)
    assert jnp.allclose(jnp.array(gradients), jnp.array([0.0, 4.0, 3.0]))
    actual = jax.vmap(derivative, (0, None, 0))(jnp.ones(3), 2.0, jnp.arange(3.0))
    assert jnp.allclose(actual, jnp.array([0.0, 2.0, 4.0]))


def test_vjp_inside_an_ir_composite():
    op = ir.Composite(3, (ir.VJP(ir.Mul(), 0), ir.Exp()), ((2, 0, 1), (3,)))
    actual = require_array(lower(op, (0.2, 0.3, 2.0)))
    assert jnp.allclose(actual, jnp.exp(0.6))


def test_vjp_inside_an_ir_scan():
    op = ir.Scan(ir.VJP(ir.Mul(), 1), 3, (None, None))
    actual = require_array(lower(op, (0.5, 2.0, 0.3)))
    assert jnp.allclose(actual, jnp.array([1.0, 2.0, 4.0]))


@pytest.mark.parametrize("selected", [2, True, 1.0 + 2.0j])
def test_nondifferentiable_input_dtype_is_rejected(selected: int | complex):
    with pytest.raises(TypeError, match="real floating-point differentiation input"):
        lower(ir.VJP(ir.Identity(), 0), (1.0, selected))


def test_integer_output_is_rejected():
    base = ir.Composite(1, (ir.Constant(2),), ((),))
    with pytest.raises(TypeError, match="real floating-point base output"):
        lower(ir.VJP(base, 0), (1, 1.0))


def test_complex_cotangent_is_rejected():
    with pytest.raises(TypeError, match="real cotangent"):
        lower(ir.VJP(ir.Exp(), 0), (1.0 + 2.0j, 1.0))


def test_incorrect_cotangent_shape_is_rejected_before_differentiation():
    with pytest.raises(ValueError, match="cotangent shape .* does not match"):
        lower(ir.VJP(ir.Exp(), 0), (jnp.ones(2), 1.0))


def test_unsupported_base_lowering_propagates_error():
    class Unsupported(ir.Op):
        random: bool = False
        """False"""

        def get_shape(self, shape: ir.Shape) -> ir.Shape:
            return shape

    with pytest.raises(NotImplementedError):
        lower(ir.VJP(Unsupported(), 0), (1.0, 1.0))
