from pangolin.interface import normal, makerv, OperatorRV
from pangolin import ir

def test_wrong_number_of_args():
    for args in [(),(0,),(0,1,2),(0,1,2,3)]:
        try:
            x = normal(*args)
            assert False
        except ValueError as e:
            assert str(e) == f"Normal op got {len(args)} arguments but expected 2."
        except TypeError as e: # wrong number of args
            if len(args) == 0:
                assert str(e) == f"normal() missing 2 required positional arguments: 'loc' and 'scale'"
            elif len(args) == 1:
                assert str(
                    e) == f"normal() missing 1 required positional argument: 'scale'"
            else:
                assert str(e) == f"normal() takes 2 positional arguments but {len(args)} were given"

def test_normal_rv_rv():
    x = makerv(0)
    y = makerv(1)
    z = normal(x,y)
    assert z.op == ir.Normal()
    assert z.parents == (x,y)
    assert isinstance(z,OperatorRV)

def test_normal_rv_scalar():
    x = makerv(0)
    y = 1
    z = normal(x,y)
    assert z.op == ir.Normal()
    assert z.parents[0] == x
    assert z.parents[1].op == ir.Constant(1)
    assert isinstance(z,OperatorRV)

def test_normal_scalar_rv():
    x = 0
    y = makerv(1)
    z = normal(x, y)
    assert z.op == ir.Normal()
    assert z.parents[0].op == ir.Constant(0)
    assert z.parents[1] == y
    assert isinstance(z, OperatorRV)

def test_normal_scalar_scalar():
    x = 0
    y = 1
    z = normal(x, y)
    assert z.op == ir.Normal()
    assert z.parents[0].op == ir.Constant(0)
    assert z.parents[1].op == ir.Constant(1)
    assert isinstance(z, OperatorRV)

