import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from pangolin import ir
from pangolin.functional import Transform, compose_transforms, tforms
from pangolin.jax_backend import lower_deterministic


def constant(value: float) -> ir.RV:
    return ir.RV(ir.Constant(value))


def normal() -> ir.RV:
    return ir.RV(ir.Normal(), constant(0.0), constant(1.0))


@pytest.mark.parametrize(
    "transform,sign",
    [
        (tforms.exp, 1),
        (tforms.log, 1),
        (tforms.logit, 1),
        (tforms.inv_logit, 1),
        (tforms.scale_by(constant(-2)), -1),
        (tforms.shift_by(constant(3)), 1),
        (tforms.scaled_logit(constant(-2), constant(3)), 1),
    ],
)
def test_library_sign_and_reversal(transform: Transform, sign: int):
    for candidate in [transform, transform.reverse, transform.reverse.reverse]:
        result = candidate(normal)()
        assert isinstance(result.op, ir.Bijected)
        bijector = result.op.bijector
        params = []
        for parent in result.parents[: bijector.n_biject_params]:
            assert isinstance(parent.op, ir.Constant)
            params.append(parent.op.value)
        assert bijector.sign is not None
        assert lower_deterministic(bijector.sign, params) == sign


def test_parameter_dependent_sign_and_transform_remain_dynamic():
    factor = constant(2)
    result = tforms.scale_by(factor)(normal)()
    assert isinstance(result.op, ir.Bijected)
    op = ir.CDF(result.op)
    evaluate = jax.jit(jax.vmap(lambda scale: lower_deterministic(op, (0.7, scale, 0.3, 0.8))))
    scales = jnp.array([-3.0, -0.5, 0.7, 2.0])
    expected = stats.norm.cdf(0.7, np.asarray(scales) * 0.3, np.abs(scales) * 0.8)
    np.testing.assert_allclose(evaluate(scales), expected, rtol=2e-6)

    logit = tforms.scaled_logit(constant(-2), constant(3)).reverse
    output = logit(normal)()
    assert isinstance(output.op, ir.Bijected)
    # The inverse implementation captures upper before lower; sign must follow that order.
    parameter_values = []
    for parent in output.parents[:2]:
        assert isinstance(parent.op, ir.Constant)
        parameter_values.append(6.0 if parent.op.value == 3 else 2.0)
    op = ir.CDF(output.op)
    evaluate_logit = jax.jit(lambda y: lower_deterministic(op, (y, *parameter_values, 0.0, 1.0)))
    assert evaluate_logit(4.0) == 0.5
    np.testing.assert_allclose(evaluate_logit(5.0), stats.norm.cdf(np.log(3.0)), rtol=2e-6)


@pytest.mark.parametrize("first,second", [(-2.0, -3.0), (-2.0, 3.0), (2.0, -3.0), (2.0, 3.0)])
def test_composition_multiplies_signs_and_preserves_cdf(first: float, second: float):
    transform = compose_transforms(
        [tforms.scale_by(constant(first)), tforms.scale_by(constant(second))]
    )
    output = transform(normal)()
    assert isinstance(output.op, ir.Bijected)
    sign = output.op.bijector.sign
    assert sign is not None
    assert lower_deterministic(sign, (first, second)) == np.sign(first * second)
    actual = lower_deterministic(ir.CDF(output.op), (0.3, first, second, 0.0, 1.0))
    np.testing.assert_allclose(actual, stats.norm.cdf(0.3, scale=abs(first * second)), rtol=2e-6)


def test_successive_transform_applications_keep_cdf_support():
    output = tforms.inv_logit(tforms.exp(normal))()
    assert isinstance(output.op, ir.Bijected)
    actual = lower_deterministic(ir.CDF(output.op), (0.8, 0.0, 1.0))
    expected = stats.norm.cdf(np.log(np.log(0.8 / 0.2)))
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
