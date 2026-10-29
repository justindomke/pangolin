import pytest

from pangolin.ir import Constant, RV


def test_constant_RV():
    x = RV(Constant(2))
    assert x.shape == ()
    assert str(x) == "2"
    assert repr(x) == "RV(Constant(2))"


def test_rv_has_no_container_length():
    x = RV(Constant([1, 2]))

    with pytest.raises(TypeError):
        len(x)


def test_ir_rvs_have_consecutive_creation_numbers():
    first = RV(Constant(1))
    second = RV(Constant(2))

    assert second._n == first._n + 1
