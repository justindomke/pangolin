from pangolin.ir.constant import Constant
from pangolin.ir import RV

def test_constant_zero_dims():
    a = Constant(1)
    assert a.get_shape() == ()

def test_constant_one_dim():
    a = Constant([1,2,3])
    assert a.get_shape() == (3,)

def test_constant_two_dims():
    a = Constant([[1,2,3],[4,5,6]])
    assert a.get_shape() == (2,3)

def test_constant_equality():
    a = Constant(1)
    b = Constant(1)
    c = Constant(2)
    d = Constant(2)
    assert a == b
    assert c == d
    assert a != c

def test_wrong_number_of_args():
    try:
        d = Constant(2)
        tmp = RV(Constant(1))
        x = RV(d,tmp)
        assert False
    except ValueError as e:
        assert str(e) == "Constant got 1 arguments but expected 0."
