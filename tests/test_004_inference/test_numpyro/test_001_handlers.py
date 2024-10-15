from pangolin import ir
from numpyro import distributions as numpyro_dist
import pangolin as pg
import numpyro
import numpy as np

from pangolin.inference.numpyro.handlers import get_numpyro_rv, get_numpyro_val

def test_get_numpyro_val_constant():
    var = pg.makerv(2)
    x = get_numpyro_val(var.op, is_observed=False)
    assert x == 2

def test_get_numpyro_rv_constant():
    var = pg.makerv(2)
    x = get_numpyro_rv(var.op, "name", None)
    print(f"{x=}")


def test_get_numpyro_rv_normal():
    with numpyro.handlers.seed(rng_seed=0):
        x = get_numpyro_rv(pg.ir.Normal(), "x", None, 3.3, 1e-10)

    print(f"{x=}")


def test_get_numpyro_val_add():
    op = ir.Add()
    out = get_numpyro_val(op, 2.5, 3.1, is_observed=False)
    assert np.allclose(out, 2.5+3.1)

def test_get_numpyro_val_mul():
    op = ir.Mul()
    out = get_numpyro_val(op, 2.5, 3.1, is_observed=False)
    assert np.allclose(out, 2.5*3.1)