import numpyro.handlers

from pangolin.interface import *
from pangolin.inference.numpyro import (
    sample_flat,
    sample,
    E,
)
from pangolin.inference.numpyro.handlers import get_numpyro_val
import numpy as np
from pangolin import ir
from util import inf_until_match


def test_composite_val_handler_nonrandom():
    def f(x):
        a = x + 2
        b = x * x
        return a+b

    composite_f = composite(f)

    x = makerv(1.5)
    y = composite_f(x)
    assert isinstance(y.op, ir.Composite)

    expected = f(1.5)
    out = get_numpyro_val(y.op, 1.5, is_observed=False)

    assert np.allclose(expected, out)


def test_composite_val_handler_random():
    @composite
    def f(x):
        a = x + 2
        b = x * x
        return normal(a**b, 1e-10)

    x = makerv(1.5)
    y = f(x)
    assert isinstance(y.op, ir.Composite)

    def model():
        val = get_numpyro_val(y.op, 1.5, is_observed=False)
        return numpyro.sample('y', val)

    with numpyro.handlers.seed(rng_seed=0):
        out = model()

    expected = (1.5+2)**(1.5**2)

    assert np.allclose(expected, out)

def test_composite_deterministic():
    @composite
    def f(x):
        a = x + 2
        b = x * x
        return a+b

    x = makerv(1.5)
    y = f(x)
    assert isinstance(y.op, ir.Composite)

    expected = (1.5 + 2) + (1.5**2)

    [ys] = sample_flat([y], [], [], niter=100)
    assert np.allclose(ys[-1], expected, rtol=1e-3, atol=1e-3)

    ys = sample(y, None, None, niter=100)
    assert np.allclose(ys[-1], expected, rtol=1e-3, atol=1e-3)



def test_composite_random():
    @composite
    def f(x):
        a = x + 2
        b = x * x
        return normal(a**b, 1e-5)

    x = makerv(1.5)
    y = f(x)
    assert isinstance(y.op, ir.Composite)

    expected = (1.5 + 2) ** (1.5**2)

    [ys] = sample_flat([y], [], [], niter=100)
    print(f"{ys=}")
    assert np.allclose(ys[-1], expected, rtol=1e-3, atol=1e-3)

    ys = sample(y, None, None, niter=100)
    assert np.allclose(ys[-1], expected, rtol=1e-3, atol=1e-3)


def test_composite_simple_const_rv():
    x = makerv(0.5)
    noise = makerv(1e-3)

    @composite
    def f(last):
        return normal(last, noise)  # +1

    y = f(x)

    print_upstream(y)

    def testfun(E_y):
        print(f"{E_y=}")
        return np.abs(E_y - 0.5) < 0.1

    inf_until_match(E, y, [], [], testfun)
