# pyright: reportUnnecessaryTypeIgnoreComment=error

"""Static typing tests for `functional.tforms`."""

from typing import TYPE_CHECKING, assert_type

from pangolin import ir
from pangolin.functional import Transform, tforms


def test_transform_objects_and_factories_have_plain_ir_types():
    parameter = ir.RV(ir.Constant(1))

    assert_type(tforms.exp, Transform)
    assert_type(tforms.log, Transform)
    assert_type(tforms.logit, Transform)
    assert_type(tforms.inv_logit, Transform)
    assert_type(tforms.stick_breaking, Transform)
    assert_type(tforms.softmax_centered, Transform)
    assert_type(tforms.softmax_centered_ilr, Transform)
    assert_type(tforms.cholesky, Transform)
    assert_type(tforms.fill_tril, Transform)
    assert_type(tforms.extract_tril, Transform)
    assert_type(tforms.exp_diagonal, Transform)
    assert_type(tforms.log_diagonal, Transform)
    assert_type(tforms.unconstrain_spd, Transform)
    assert_type(tforms.scale_by(parameter), Transform)
    assert_type(tforms.shift_by(parameter), Transform)
    assert_type(tforms.scaled_logit(parameter, parameter), Transform)


if TYPE_CHECKING:
    tforms.scale_by(2)  # type: ignore
    tforms.shift_by(2)  # type: ignore
    tforms.scaled_logit(0, 1)  # type: ignore
