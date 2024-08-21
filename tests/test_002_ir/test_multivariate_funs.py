from pangolin.ir.multivariate_funs import *

# MatMul
# Inv
# Softmax
# Sum

def test_equality():
    assert MatMul() == MatMul()
    assert Inv() == Inv()
    assert Softmax() == Softmax()
    assert Sum(0) == Sum(0)
    assert Sum(1) == Sum(1)
    assert MatMul() != Inv()
    assert MatMul() != Softmax()
    assert MatMul() != Sum(0)
    assert Sum(0) != Sum(1)

