# pyright: reportUnnecessaryTypeIgnoreComment=error

"""Check signatures remain precise through re-exports and configured conveniences."""

from typing import TYPE_CHECKING, assert_type

from pangolin import ir
from pangolin import interface as pi


def test_convenience_return_types():
    for interface in (pi.default, pi.Interface()):
        value: pi.RVLike = [1, 4]
        assert_type(interface.sqrt(value), ir.RV)
        assert_type(interface.expit(value), ir.RV)
        assert_type(interface.sigmoid(value), ir.RV)
        assert_type(interface.fill_tril([1, 2, 3]), ir.RV)
        assert_type(interface.extract_tril([[1, 0], [2, 3]]), ir.RV)
        assert_type(interface.composite(interface.normal)(0, 1), ir.RV)
        assert_type(interface.tforms.exp(interface.normal)(0, 1), ir.RV)
        assert_type(interface.tforms.scale_by(factor=2), pi.Transform)
        assert_type(interface.tforms.shift_by(offset=[1, 2]), pi.Transform)
        assert_type(interface.tforms.scaled_logit(lower=1, upper=5), pi.Transform)
        assert_type(interface.vmap(interface.exp)(interface.constant([1, 2])), ir.RV)
        assert_type(interface.scan(interface.exp, 2)(interface.constant(0)), ir.RV)
    assert_type(pi.sqrt(4), ir.RV)
    assert_type(pi.composite(pi.normal)(0, 1), ir.RV)
    assert_type(pi.vmap(pi.exp)(pi.constant([1, 2])), ir.RV)
    assert_type(pi.scan(pi.exp, 2)(pi.constant(0)), ir.RV)


if TYPE_CHECKING:
    pi.tforms.scale_by()  # type: ignore
    pi.tforms.scale_by(unknown=2)  # type: ignore
    pi.tforms.shift_by(object())  # type: ignore
    pi.tforms.scaled_logit(1)  # type: ignore
    pi.sqrt()  # type: ignore
    pi.sigmoid(1, 2)  # type: ignore
    pi.expit(missing=1)  # type: ignore
    pi.fill_tril(object())  # type: ignore
    pi.extract_tril()  # type: ignore
    pi.composite(pi.normal)(0)  # type: ignore
    pi.default.composite(pi.normal)(0)  # type: ignore
    pi.vmap(pi.exp)()  # type: ignore
    pi.default.vmap(pi.exp)()  # type: ignore
    pi.scan(pi.exp, 2)(0)  # type: ignore
    pi.default.scan(pi.exp, 2)(0)  # type: ignore
