import numpy as np
from pangolin import ir
from pangolin.ir import RV
from pangolin.jax_backend import fill_in
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


def inf_until_match(inf, vars, given, vals, testfun, niter_start=1000, niter_max=100000):
    from time import time

    niter = niter_start
    while niter <= niter_max:
        t0 = time()
        out = inf(vars, given, vals, num_samples=niter)
        t1 = time()
        # print(f"{niter=} {t1 - t0}")
        if testfun(out):
            assert True
            return
        else:
            niter *= 2
    assert False


# def test_random():
#     # x ~ Normal(0, 1); w ~ Normal(x, 1); no observations
#     # deferred=True: x is MCMC'd (or forward-sampled), w is filled in per draw
#     loc = RV(ir.Constant(0.0))
#     scale = RV(ir.Constant(1.0))
#     x = RV(ir.Normal(), loc, scale)
#     w = RV(ir.Normal(), x, scale)

#     def myfun(key):
#         return fill_in([x], [jnp.array(2.0)], [w], key)

#     key = jax.random.PRNGKey(0)

#     niter = 1000
#     while niter < 1e9:
#         out = jax.vmap(myfun)(jax.random.split(key, niter))

#         if np.abs(np.mean(out) - 2) < 0.1 and np.abs(np.var(out) - 1) < 0.1:
#             assert True
#             return
#     assert False

#     # out[0] ~ Normal(2.0, 1.0) — mean over many keys should be ≈ 2.0
