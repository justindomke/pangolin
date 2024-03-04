from pangolin.interface import (
    normal,
    normal_scale,
    makerv,
    exp,
    Constant,
    vmap,
    VMapDist,
    viz_upstream,
    Loop,
    LoopRV,
    print_upstream,
)
from pangolin import dag
import numpy as np  # type: ignore
from pangolin.transforms.normal_normal import normal_normal
from pangolin.transforms.constant_op import constant_op


def test_shapes():
    var = makerv(np.random.randn(3))
    loop_var = LoopRV(None, vmap_var=var, loop_dim=0, loop=Loop())
    assert loop_var.shape == ()

    var = makerv(np.random.randn(3, 4))
    loop_var = LoopRV(None, vmap_var=var, loop_dim=0, loop=Loop())
    assert loop_var.shape == (4,)
    loop_var = LoopRV(None, vmap_var=var, loop_dim=1, loop=Loop())
    assert loop_var.shape == (3,)

    var = makerv(np.random.randn(3, 4, 5))
    loop_var = LoopRV(None, vmap_var=var, loop_dim=0, loop=Loop())
    assert loop_var.shape == (4, 5)
    loop_var = LoopRV(None, vmap_var=var, loop_dim=1, loop=Loop())
    assert loop_var.shape == (3, 5)
    loop_var = LoopRV(None, vmap_var=var, loop_dim=2, loop=Loop())
    assert loop_var.shape == (3, 4)


def test_generating():
    loop = Loop()
    x = makerv(np.random.randn(3, 4))
    y = makerv(np.random.randn(3, 4))
    x_loop = LoopRV(None, vmap_var=x, loop_dim=0, loop=loop)
    y_loop = LoopRV(None, vmap_var=y, loop_dim=0, loop=loop)
    z_loop = x_loop + y_loop

    assert x_loop.shape == (4,)
    assert y_loop.shape == (4,)
    assert z_loop.shape == (4,)

    z = z_loop.vmap_var
    assert z.shape == (3, 4)

    print_upstream(z)
