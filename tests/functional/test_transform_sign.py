from typing import Any
from collections.abc import Callable

import pytest
import jax.numpy as jnp
import numpy as np

from pangolin import ir
from pangolin.functional import Transform, compose_transforms, tforms
from pangolin.jax_backend import lower_deterministic


def constant(value: Any) -> ir.RV:
    return ir.RV(ir.Constant(value))


def normal() -> ir.RV:
    return ir.RV(ir.Normal(), constant(0.0), constant(1.0))


def identity(*, sign: Callable[[], ir.RV] | None = None) -> Transform:
    return Transform(lambda x: x, lambda y: y, lambda x, y: constant(0.0), sign=sign)


def compiled(transform: Transform) -> ir.Bijector:
    result = transform(normal)()
    assert isinstance(result.op, ir.Bijected)
    return result.op.bijector


def test_sign_defaults_to_none():
    transform = identity()
    for candidate in [transform, transform.reverse, compiled(transform)]:
        assert candidate.sign is None


def test_callbacks_are_deferred_and_called_once_per_application():
    calls = []

    def sign() -> ir.RV:
        calls.append("sign")
        return constant(1)

    transform = identity(sign=sign)
    fun = transform.apply_and_invert(normal)
    assert calls == []
    y, x = fun()
    assert calls == ["sign"]
    assert x.parents[0] is y


@pytest.mark.parametrize("value", [None, 1, [], [constant(1)]])
def test_sign_requires_one_rv(value: Any):
    with pytest.raises(TypeError):
        compiled(identity(sign=lambda: value))


def test_sign_cannot_capture_extra_rvs():
    extra = constant(1)
    with pytest.raises(ValueError, match="sign cannot capture"):
        compiled(identity(sign=lambda: extra))


def test_sign_cannot_depend_on_transformed_input():
    captured = []

    def forward(x: ir.RV) -> ir.RV:
        captured.append(x)
        return x

    transform = Transform(forward, lambda y: y, lambda x, y: constant(0), sign=lambda: captured[0])
    with pytest.raises(ValueError, match="sign cannot capture"):
        compiled(transform)


@pytest.mark.parametrize("value", [[1], [1, -1], [[1]]])
def test_sign_output_shapes_are_checked(value: Any):
    with pytest.raises(ValueError, match="sign must return a scalar"):
        compiled(identity(sign=lambda: constant(value)))


def test_sign_must_be_deterministic():
    with pytest.raises(ValueError):
        compiled(identity(sign=normal))


def test_sign_captures_are_reordered_to_match_forward():
    lower, upper = constant(-2.0), constant(3.0)
    base = tforms.scaled_logit(lower, upper)

    def sign() -> ir.RV:
        return ir.RV(ir.Div(), upper, lower)

    transform = Transform(
        base.forward,
        base.inverse,
        base.log_det_jac,
        sign=sign,
    )
    result = transform(normal)()
    assert result.parents[:2] == (lower, upper)
    assert isinstance(result.op, ir.Bijected)
    assert result.op.bijector.sign == ir.Composite(2, (ir.Div(),), ((1, 0),))


def test_existing_parameter_can_be_sign_output():
    parameter = constant(1)
    base = tforms.scale_by(parameter)
    transform = Transform(base.forward, base.inverse, base.log_det_jac, sign=lambda: parameter)
    assert compiled(transform).sign == ir.Identity()


def test_constant_sign_still_accepts_all_parameters():
    bijector = compiled(tforms.shift_by(constant(2)))
    assert bijector.sign == ir.Composite(1, (ir.Constant(1),), ((),))


def test_reversal_preserves_sign_callback_and_its_interface():
    interface = ir.RVInterface()

    def sign() -> ir.RV:
        return ir.RV(ir.Constant(-1), interface=interface)

    transform = identity(sign=sign)
    reverse = transform.reverse
    assert reverse.sign is sign
    assert reverse.reverse.sign is sign
    assert reverse.sign().interface is interface
    assert compiled(reverse).sign == ir.Constant(-1)


def test_missing_composition_sign_does_not_call_other_signs():
    def unused() -> ir.RV:
        raise AssertionError("should not be called")

    transform = compose_transforms([identity(sign=unused), identity()])
    assert transform.sign is None
    assert compiled(transform).sign is None


def test_sign_does_not_change_output_interfaces():
    interface = ir.RVInterface()
    transform = Transform(
        lambda x: ir.RV(ir.Identity(), x, interface=interface),
        lambda y: ir.RV(ir.Identity(), y, interface=interface),
        lambda x, y: constant(0),
        sign=lambda: constant(1),
    )
    y, x = transform.apply_and_invert(normal)()
    assert y.interface is interface
    assert x.interface is interface


def test_multivariate_library_sign_remains_absent():
    for transform in [
        tforms.stick_breaking,
        tforms.softmax_centered,
        tforms.cholesky,
        tforms.fill_tril,
        tforms.shift_by(constant([1, 2])),
    ]:
        assert transform.sign is None


def test_repr_includes_only_supplied_sign():
    assert "sign=" not in repr(identity())
    assert "sign=" in repr(tforms.exp)


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_composition_multiplies_signs(count: int):
    transforms = [tforms.scale_by(constant(-2.0)) for _ in range(count)]
    result = compose_transforms(transforms)(normal)()
    bijector = result.op.bijector
    assert bijector.sign is not None
    params = (jnp.array(-2.0),) * count
    assert lower_deterministic(bijector.sign, params) == (-1) ** count


@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_composed_parameter_dependent_sign_forward_inverse_and_ldj(direction: str):
    factor, offset = constant(-2.0), constant(3.0)
    transform = compose_transforms(
        [tforms.exp, tforms.scale_by(factor), tforms.shift_by(offset)],
        log_det_direction=direction,
    )
    result = transform(normal)()
    assert result.parents[:2] == (factor, offset)
    bijector = result.op.bijector
    assert bijector.sign is not None
    for scale in (-2.0, 4.0):
        x = jnp.array(0.7)
        params = (jnp.array(scale), jnp.array(3.0))
        y = lower_deterministic(bijector.forward, (x, *params))
        np.testing.assert_allclose(y, scale * np.exp(0.7) + 3, rtol=1e-6)
        np.testing.assert_allclose(
            lower_deterministic(bijector.inverse, (y, *params)), x, rtol=1e-6
        )
        np.testing.assert_allclose(
            lower_deterministic(bijector.log_det_jac, (x, y, *params)),
            np.log(abs(scale)) + 0.7,
            rtol=1e-6,
        )
        assert lower_deterministic(bijector.sign, params) == np.sign(scale)


def test_reversed_decreasing_transform_has_same_sign_and_opposite_ldj():
    bijector = compiled(tforms.scale_by(constant(-2.0)).reverse)
    assert bijector.sign is not None
    params = (jnp.array(-2.0),)
    x = jnp.array(3.0)
    y = lower_deterministic(bijector.forward, (x, *params))
    np.testing.assert_allclose(y, -1.5)
    np.testing.assert_allclose(lower_deterministic(bijector.inverse, (y, *params)), x)
    np.testing.assert_allclose(
        lower_deterministic(bijector.log_det_jac, (x, y, *params)), -np.log(2), rtol=1e-6
    )
    assert lower_deterministic(bijector.sign, params) == -1


def test_increasing_library_signs():
    for transform in (
        tforms.exp,
        tforms.log,
        tforms.logit,
        tforms.inv_logit,
        tforms.shift_by(constant(2.0)),
        tforms.scaled_logit(constant(-2.0), constant(3.0)),
    ):
        result = transform(normal)()
        bijector = result.op.bijector
        assert bijector.sign is not None
        params = tuple(
            jnp.asarray(parent.op.value) for parent in result.parents[: bijector.n_biject_params]
        )
        assert lower_deterministic(bijector.sign, params) == 1


def test_sign_cannot_hide_new_randomness_behind_deterministic_output():
    with pytest.raises(ValueError):
        compiled(identity(sign=lambda: ir.RV(ir.Exp(), normal())))
