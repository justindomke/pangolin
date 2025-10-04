from pangolin import ir
from pangolin.inference.numpyro import vmap
from pangolin.inference.numpyro.vmap import (
    handle_vmap_nonrandom,
    vmap_nesting,
    get_all_plate_sizes,
)
from jax import numpy as jnp

# handle_vmap_nonrandom(op: ir.VMap, *numpyro_parents, is_observed):


def test_handle_nonrandom_exp():
    op = ir.VMap(ir.Exp(), [None], 5)
    x = jnp.array(1.0)
    y = handle_vmap_nonrandom(op, x, is_observed=False)
    expected = jnp.exp(1.0) * jnp.ones(5)
    assert jnp.allclose(y, expected)


def test_handle_nonrandom_exp_2d():
    op = ir.VMap(ir.VMap(ir.Exp(), [None], 5), [0])
    x = jnp.array([1.0, 2.0])
    y = vmap.handle_vmap_nonrandom(op, x, is_observed=False)
    expected = jnp.exp(jnp.array([1.0, 2.0]))[:, None] * jnp.ones((2, 5))
    assert jnp.allclose(y, expected)


def test_handle_nonrandom_add():
    op = ir.VMap(ir.Add(), [None, 0])
    x = jnp.array(1.0)
    y = jnp.array([2.0, 3.0, 4.0])
    z = handle_vmap_nonrandom(op, x, y, is_observed=False)
    expected = x + y
    assert jnp.allclose(z, expected)


def test_handle_nonrandom_add_2d():
    op = ir.VMap(ir.VMap(ir.Add(), [None, 0]), [None, 0])
    x = jnp.array(1.0)
    y = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    z = handle_vmap_nonrandom(op, x, y, is_observed=False)
    expected = x + y
    assert jnp.allclose(z, expected)


def test_vmap_nesting():
    assert vmap_nesting(ir.Exp()) == 0
    assert vmap_nesting(ir.VMap(ir.Exp(), [None], 5)) == 1
    assert vmap_nesting(ir.VMap(ir.VMap(ir.Exp(), [None], 5), [0])) == 2


def test_get_all_plate_sizes():
    op = ir.Exp()
    assert get_all_plate_sizes(op, jnp.array(1.0)) == ()

    op = ir.VMap(ir.Exp(), [None], 5)
    assert get_all_plate_sizes(op, jnp.array(1.0)) == (5,)

    op = ir.VMap(ir.VMap(ir.Exp(), [None], 5), [0])
    assert get_all_plate_sizes(op, jnp.array([1.0, 2.0])) == (5, 2)
