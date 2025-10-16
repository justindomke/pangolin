from pangolin import blackjax
from pangolin import ir
from jax import numpy as jnp


def test_simple():
    loc = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))
    x = ir.RV(ir.Normal(), loc, scale)
    [x_samps] = blackjax.sample_flat([x], [], [], niter=1000)
    assert x_samps.shape == (1000,)
    assert jnp.abs(jnp.mean(x_samps) - 0) < 0.05
    assert jnp.abs(jnp.std(x_samps) - 1) < 0.05


def test_conditioning():
    loc = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))
    x = ir.RV(ir.Normal(), loc, scale)
    y = ir.RV(ir.Normal(), x, scale)
    [x_samps] = blackjax.sample_flat([x], [y], [1], niter=1000)
    assert x_samps.shape == (1000,)
    assert jnp.abs(jnp.mean(x_samps) - 0.5) < 0.05
    assert jnp.abs(jnp.var(x_samps) - 0.5) < 0.05


def test_nonrandom():
    loc = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))
    x = ir.RV(ir.Normal(), loc, scale)
    y = ir.RV(ir.Add(), x, x)
    [x_samps, y_samps] = blackjax.sample_flat([x, y], [], [], niter=100)
    assert x_samps.shape == (100,)
    assert y_samps.shape == (100,)
    assert jnp.allclose(y_samps, x_samps * 2)


def test_nonrandom_conditioning():
    loc = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))
    x = ir.RV(ir.Normal(), loc, scale)
    z = ir.RV(ir.Normal(), x, scale)
    y1 = ir.RV(ir.Add(), x, x)
    y2 = ir.RV(ir.Mul(), x, x)
    [x_samps, y1_samps, y2_samps] = blackjax.sample_flat(
        [x, y1, y2], [z], [1.0], niter=1000
    )
    assert x_samps.shape == y1_samps.shape == y2_samps.shape == (1000,)
    assert jnp.abs(jnp.mean(x_samps) - 0.5) < 0.05
    assert jnp.abs(jnp.var(x_samps) - 0.5) < 0.05
    assert jnp.allclose(y1_samps, x_samps * 2)
    assert jnp.allclose(y2_samps, x_samps**2)


def test_nonrandom_from_given():
    loc = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))
    x = ir.RV(ir.Normal(), loc, scale)
    z = ir.RV(ir.Normal(), x, scale)
    y = ir.RV(ir.Mul(), z, scale)
    [x_samps, y_samps] = blackjax.sample_flat([x, y], [z], [1.0], niter=1000)
    assert x_samps.shape == y_samps.shape == (1000,)
    assert jnp.abs(jnp.mean(x_samps) - 0.5) < 0.05
    assert jnp.abs(jnp.var(x_samps) - 0.5) < 0.05
    assert jnp.allclose(y_samps, 1.0)


def test_given_in_output():
    loc = ir.RV(ir.Constant(0))
    scale = ir.RV(ir.Constant(1))
    x = ir.RV(ir.Normal(), loc, scale)
    z = ir.RV(ir.Normal(), x, scale)
    y = ir.RV(ir.Mul(), z, scale)
    [y_samps, z_samps] = blackjax.sample_flat([y, z], [z], [2.3], niter=100)
    assert y_samps.shape == z_samps.shape == (100,)
    assert jnp.allclose(z_samps, 2.3)
    assert jnp.allclose(y_samps, 2.3)
