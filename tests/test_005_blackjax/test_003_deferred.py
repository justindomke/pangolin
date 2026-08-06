from pangolin import ir
from pangolin.ir import RV
from pangolin.jax_backend import fill_in
from jax import numpy as jnp
import jax


def test_sanity():
    # x ~ Normal(0, 1); w ~ Normal(x, 1); no observations
    # deferred=True: x is MCMC'd (or forward-sampled), w is filled in per draw
    loc = RV(ir.Constant(0.0))
    scale = RV(ir.Constant(1.0))
    x = RV(ir.Normal(), loc, scale)
    w = RV(ir.Normal(), x, scale)
    out = fill_in([x], [jnp.array(2.0)], [w], key=jax.random.PRNGKey(0))
    # out[0] ~ Normal(2.0, 1.0) — mean over many keys should be ≈ 2.0
