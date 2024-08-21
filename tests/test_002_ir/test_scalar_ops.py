from pangolin.ir.scalar_ops import *
from pangolin.ir import RV, Constant

def test_sameness():
    assert Normal() == Normal()
    assert Normal() != StudentT()

def test_hashing():
    assert hash(Normal()) == hash(Normal())
    assert hash(NormalPrec()) == hash(NormalPrec())

def test_wrong_number_of_args():
    for args in [(),(0,),(0,1,2),(0,1,2,3)]:
        try:
            d = Normal()
            args = tuple(RV(Constant(a)) for a in args) # convert to RVs
            x = RV(d,*args)
            assert False
        except ValueError as e:
            assert str(e) == f"Normal op got {len(args)} arguments but expected 2."