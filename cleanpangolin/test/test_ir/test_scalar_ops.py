from cleanpangolin.ir.scalar_ops import *

def test_sameness():
    assert normal == normal
    assert Normal() == Normal()
    assert normal != normal_prec

def test_hashing():
    assert hash(normal) == hash(normal)
    assert hash(Normal()) == hash(Normal())
    assert hash(normal) != hash(normal_prec)

def test_wrong_number_of_args():
    for args in [(),(0,),(0,1,2),(0,1,2,3)]:
        try:
            x = normal(*args)
            assert False
        except ValueError as e:
            assert str(e) == f"Normal op got {len(args)} arguments but expected 2."