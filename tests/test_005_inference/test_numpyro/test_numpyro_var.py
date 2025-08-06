from pangolin import ir
from pangolin.inference.numpyro.handlers import (
    get_numpyro_rv, get_numpyro_val
)
from numpyro import distributions as numpyro_dist
import numpyro

def test_numpyro_var_add():
    assert get_numpyro_val(ir.Add(), 2, 3, is_observed=False) == 5

def test_numpyro_var_normal():
    out = get_numpyro_val(ir.Normal(), 2, 3, is_observed=False)
    assert out.loc == 2
    assert out.scale == 3
    assert isinstance(out, numpyro_dist.Normal)

def test_numpyro_var_normal_prec():
    out = get_numpyro_val(ir.NormalPrec(), 2, 2, is_observed=False)
    assert out.loc == 2
    assert out.scale == 1 / 4  # exact because using powers of 2 yay floating point
    assert isinstance(out, numpyro_dist.Normal)