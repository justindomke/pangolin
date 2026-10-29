import pytest

from pangolin import ir
from pangolin.interface import Interface


def test_sum_method_and_transpose_property_match_explicit_calls():
    pi = Interface()
    x = pi.constant([[1, 2], [3, 4]])
    assert ir.rv_equal(x.sum(axis=0), pi.sum(x, axis=0))
    assert ir.rv_equal(x.T, pi.transpose(x))


def test_index_proxy_preserves_identity_and_is_lazy():
    pi = Interface()
    x = pi.constant([1, 2])
    before = ir.RV._n
    proxy = x.vindex
    assert proxy.rv is x
    assert proxy.interface is pi
    assert ir.RV._n == before
    assert ir.rv_equal(proxy[0], pi.vindex(x, 0))


@pytest.mark.parametrize(
    "name", ["config", "overload", "bind", "coerce", "normal", "rv", "exposed"]
)
def test_interface_helpers_do_not_leak_onto_rv(name):
    x = Interface().constant(1)
    assert not hasattr(x, name)


def test_saved_bound_method_observes_overload():
    pi = Interface(broadcasting="off")
    x = pi.constant(1)
    add = x.__add__
    with pi.overload(broadcasting="simple"):
        result = add([1, 2])
    assert result.shape == (2,)
    assert result.interface is pi
    with pytest.raises(ValueError):
        add([1, 2])
