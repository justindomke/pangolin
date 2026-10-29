"""Static typing tests for the Transform interface.

Run these tests with::

    pyright tests/interface/test_transform_factories_typing.py

Positive cases use `assert_type`. Negative cases are suppressed individually under
`TYPE_CHECKING`; the file-level setting makes an unexpectedly accepted case fail because its
suppression becomes unnecessary.
"""

from pangolin import interface as pi, ir
from pangolin.ir import RV

# pyright: reportUnnecessaryTypeIgnoreComment=error


from typing import TYPE_CHECKING, assert_type

from pangolin.interface import tforms

exp = pi.tforms.exp
scale_by = pi.tforms.scale_by
shift_by = pi.tforms.shift_by
from pangolin.functional.transforming import Transform
from pangolin.functional.transforming import compose_transforms


def scalar_normal(mean: RV, scale: RV) -> RV:
    return RV(ir.Normal(), mean, scale, interface=pi.default)


def test_transform_preserves_the_wrapped_function_signature():
    transformed = exp(scalar_normal)
    mean = RV(ir.Constant(0), interface=pi.default)
    scale = RV(ir.Constant(1), interface=pi.default)
    result = transformed(mean=mean, scale=scale)

    assert_type(
        result,
        RV,
    )


def test_apply_and_invert_preserves_the_signature_and_result_types():
    transformed = exp.apply_and_invert(scalar_normal)
    mean = RV(ir.Constant(0), interface=pi.default)
    scale = RV(ir.Constant(1), interface=pi.default)
    result = transformed(mean=mean, scale=scale)

    assert_type(
        result,
        tuple[
            RV,
            RV,
        ],
    )


def test_transform_constructor_accepts_exact_callback_signatures():
    def forward(x: RV) -> RV:
        return x + 1

    def inverse(y: RV) -> RV:
        return y - 1

    def log_det_jac(x: RV, y: RV) -> RV:
        return x * 0

    transform = Transform(forward, inverse, log_det_jac)

    assert_type(transform, Transform)
    assert_type(transform.forward(RV(ir.Constant(0), interface=pi.default)), RV)
    assert_type(transform.inverse(RV(ir.Constant(0), interface=pi.default)), RV)
    assert_type(
        transform.log_det_jac(
            RV(ir.Constant(0), interface=pi.default), RV(ir.Constant(0), interface=pi.default)
        ),
        RV,
    )


def test_factories_and_composition_return_transforms():
    scaled = scale_by(2)
    shifted = shift_by(3)
    composed = compose_transforms([scaled, shifted])

    assert_type(scaled, Transform)
    assert_type(shifted, Transform)
    assert_type(composed, Transform)
    assert_type(composed.reverse, Transform)


def test_standard_transform_library_has_precise_transform_types():
    assert_type(tforms.exp, Transform)
    assert_type(tforms.log, Transform)
    assert_type(tforms.logit, Transform)
    assert_type(tforms.inv_logit, Transform)
    assert_type(tforms.cholesky, Transform)
    assert_type(tforms.fill_tril, Transform)
    assert_type(tforms.extract_tril, Transform)
    assert_type(tforms.exp_diagonal, Transform)
    assert_type(tforms.log_diagonal, Transform)
    assert_type(tforms.unconstrain_spd, Transform)
    assert_type(tforms.scaled_logit(0, 1), Transform)
    assert_type(tforms.scale_by(2), Transform)
    assert_type(tforms.shift_by(3), Transform)


if TYPE_CHECKING:

    def binary_forward(x: RV, parameter: RV) -> RV:
        return x + parameter

    Transform(
        binary_forward,  # type: ignore
        lambda y: y,
        lambda x, y: x * 0,
    )

    def integer_inverse(y: RV) -> int:
        return 1

    Transform(
        lambda x: x,
        integer_inverse,  # type: ignore
        lambda x, y: x * 0,
    )

    def unary_log_det(x: RV) -> RV:
        return x * 0

    Transform(
        lambda x: x,
        lambda y: y,
        unary_log_det,  # type: ignore
    )

    def integer_distribution(mean: RV, scale: RV) -> int:
        return 1

    exp(integer_distribution)  # type: ignore

    compose_transforms([exp, "log"])  # type: ignore
