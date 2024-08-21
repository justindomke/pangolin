from pangolin.ir.constant import Constant
from pangolin.ir.op import RV

def test_constant_RV():
    x = RV(Constant(2))
    assert x.shape == ()
    assert str(x) == '2'
    assert repr(x) == 'RV(Constant(2))'

# def test_op_RV_creation():
#     d = Constant(2)
#     x = d()
#     assert x.shape == ()
#     assert str(x) == '2'
#     assert repr(x) == 'RV(Constant(2))'