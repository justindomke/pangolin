from pangolin.interface import makerv, sum
import numpy as np

def test_sum():
    x = makerv(np.random.randn(5,3))
    y = sum(x,axis=0)
    assert y.shape == (3,)
    z = sum(x,axis=1)
    assert z.shape == (5,)