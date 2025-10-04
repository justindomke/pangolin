from pangolin.ir import *

# MatMul
# Inv
# Softmax
# Sum


def test_equality():
    assert Matmul() == Matmul()
    assert Inv() == Inv()
    assert Softmax() == Softmax()
    assert Sum(0) == Sum(0)
    assert Sum(1) == Sum(1)
    assert Matmul() != Inv()
    assert Matmul() != Softmax()
    assert Matmul() != Sum(0)
    assert Sum(0) != Sum(1)
