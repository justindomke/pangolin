# pyright: reportUnnecessaryTypeIgnoreComment=error

"""Static typing tests for `functional.Transform`."""

from typing import TYPE_CHECKING, assert_type

from pangolin import ir
from pangolin.functional import Transform, compose_transforms


def test_transform_preserves_the_distribution_signature():
    transform = Transform(
        lambda x: ir.RV(ir.Exp(), x),
        lambda y: ir.RV(ir.Log(), y),
        lambda x, y: ir.RV(ir.Log(), y),
    )

    def distribution(mean: ir.RV, *, scale: ir.RV, configuration: str) -> ir.RV:
        return ir.RV(ir.Normal(), mean, scale)

    transformed = transform(distribution)
    transformed_and_base = transform.apply_and_invert(distribution)
    mean = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))

    assert_type(transform, Transform)
    assert_type(transformed(mean, scale=scale, configuration="default"), ir.RV)
    assert_type(
        transformed_and_base(mean, scale=scale, configuration="default"),
        tuple[ir.RV, ir.RV],
    )
    assert_type(transform.reverse, Transform)


def test_transform_callbacks_use_plain_ir_rv_types():
    def forward(x: ir.RV) -> ir.RV:
        raise NotImplementedError

    def inverse(y: ir.RV) -> ir.RV:
        raise NotImplementedError

    def log_det_jac(x: ir.RV, y: ir.RV) -> ir.RV:
        raise NotImplementedError

    transform = Transform(forward, inverse, log_det_jac)

    assert_type(transform, Transform)
    assert_type(transform.reverse, Transform)


def test_composition_preserves_a_common_rv_family():
    transform = Transform(
        lambda x: ir.RV(ir.Exp(), x),
        lambda y: ir.RV(ir.Log(), y),
        lambda x, y: ir.RV(ir.Log(), y),
    )

    composed = compose_transforms([transform, transform.reverse])

    assert_type(composed, Transform)


if TYPE_CHECKING:

    def invalid_distribution() -> int:
        return 1

    transform = Transform(
        lambda x: ir.RV(ir.Exp(), x),
        lambda y: ir.RV(ir.Log(), y),
        lambda x, y: ir.RV(ir.Log(), y),
    )
    transform(invalid_distribution)  # type: ignore

    def invalid_forward(x: ir.RV, parameter: ir.RV) -> ir.RV:
        return ir.RV(ir.Add(), x, parameter)

    Transform(
        invalid_forward,  # type: ignore
        lambda y: ir.RV(ir.Log(), y),
        lambda x, y: ir.RV(ir.Log(), y),
    )
