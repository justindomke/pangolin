from cleanpangolin.ir.scalar_ops import *

def test_sameness():
    assert Normal() == Normal()
    assert Normal() != StudentT()

def test_hashing():
    assert hash(Normal()) == hash(Normal())
    assert hash(NormalPrec()) == hash(NormalPrec())

def test_wrong_number_of_args():
    for args in [(),(0,),(0,1,2),(0,1,2,3)]:
        try:
            x = Normal()(*args)
            assert False
        except ValueError as e:
            assert str(e) == f"Normal op got {len(args)} arguments but expected 2."