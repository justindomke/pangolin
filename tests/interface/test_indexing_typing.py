# pyright: reportUnnecessaryTypeIgnoreComment=error

"""Check the types of explicit and bracket indexing calls."""

from typing import Any, TYPE_CHECKING, assert_type

from pangolin import ir
from pangolin import interface as pi


def test_indexing_returns_plain_rvs():
    for interface in (pi.default, pi.Interface()):
        x = interface.constant([[1, 2], [3, 4]])
        index = ir.RV(ir.Constant([0, 1]))
        assert_type(interface.oindex(x, index, ...), ir.RV)
        assert_type(interface.getitem(x, index, ...), ir.RV)
        assert_type(interface.vindex(x, index, index), ir.RV)
        assert_type(x[index, ...], ir.RV)
        assert_type(x.vindex[:, :], Any)
        proxy = x.vindex
        assert_type(proxy[index, index], Any)

    assert_type(pi.oindex([[1, 2], [3, 4]], slice(None), 0), ir.RV)
    assert_type(pi.vindex([[1, 2], [3, 4]], [0, 1], 0), ir.RV)


if TYPE_CHECKING:
    pi.oindex()  # type: ignore
    pi.vindex()  # type: ignore
    pi.oindex([1, 2], object())  # type: ignore
    pi.vindex([1, 2], object())  # type: ignore
    pi.vindex([1, 2], axis=0)  # type: ignore
