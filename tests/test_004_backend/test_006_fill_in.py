import numpy as np
from pangolin import ir

from pangolin.backend import fill_in
import jax
import numpyro.distributions
from jax import numpy as jnp


def test_simple():
    loc = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))
    x = ir.RV(ir.Normal(), loc, scale)
    y = ir.RV(ir.Add(), x, scale)

    [y_val] = fill_in([x], [2.0], [y])
    assert np.allclose(y_val, 3.0)


def test_pair():
    loc = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))
    x1 = ir.RV(ir.Normal(), loc, scale)
    x2 = ir.RV(ir.Normal(), loc, scale)
    y1 = ir.RV(ir.Add(), x1, x2)
    y2 = ir.RV(ir.Mul(), x1, x2)

    [y1_val, y2_val] = fill_in([x1, x2], [2.0, 3.0], [y1, y2])
    assert np.allclose(y1_val, 5.0)
    assert np.allclose(y2_val, 6.0)

    [y1_val, y2_val] = fill_in([x2, x1], [3.0, 2.0], [y1, y2])
    assert np.allclose(y1_val, 5.0)
    assert np.allclose(y2_val, 6.0)

    [y2_val, y1_val] = fill_in([x1, x2], [2.0, 3.0], [y2, y1])
    assert np.allclose(y1_val, 5.0)
    assert np.allclose(y2_val, 6.0)

    [y2_val, y1_val] = fill_in([x2, x1], [3.0, 2.0], [y2, y1])
    assert np.allclose(y1_val, 5.0)
    assert np.allclose(y2_val, 6.0)
