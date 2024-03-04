from pangolin.interface import (
    normal,
    normal_scale,
    makerv,
    exp,
    Constant,
    vmap,
    VMapDist,
    viz_upstream,
)
from pangolin import dag
import numpy as np  # type: ignore
from pangolin.transforms.normal_normal import normal_normal
from pangolin.transforms.constant_op import constant_op


def test_vmap_everything():
    a = makerv([1.1, 2.2])
    b = makerv([3.3, 4.4])
    c = makerv([5.5, 6.6])
    z = vmap(normal_scale, 0)(a, b)
    x = vmap(normal_scale, 0)(z, c)
    x_val = np.array([4.4, 5.5])

    [new_x, new_z], [new_x2], [new_x_val] = normal_normal.apply_to_node(
        x, [x, z], [x], [x_val]
    )
    assert new_x.cond_dist == VMapDist(normal_scale, (0, 0), 2)
    assert new_z.cond_dist == VMapDist(normal_scale, (0, 0), 2)
    assert new_x.shape == (2,)
    assert new_z.shape == (2,)
    assert new_x != new_z
    assert new_x in dag.upstream_nodes(new_z)


def test_vmap_only_inside():
    a = makerv(1.1)
    b = makerv(2.2)
    c = makerv(3.3)
    z = vmap(normal_scale, None, axis_size=2)(a, b)
    x = vmap(normal_scale, (0, None))(z, c)
    x_val = np.array([4.4, 5.5])

    [new_x, new_z], [new_x2], [new_x_val] = normal_normal.apply_to_node(
        x, [x, z], [x], [x_val]
    )
    assert new_x.cond_dist == VMapDist(normal_scale, (None, None), 2)
    assert new_z.cond_dist == VMapDist(normal_scale, (0, None), 2)
    assert new_x in dag.upstream_nodes(new_z)
    assert new_x2 is new_x
    assert new_x_val is x_val


def test_vmap_half_inside():
    a = makerv(1.1)
    b = makerv([2.2, 3.3])
    c = makerv(4.4)
    z = vmap(normal_scale, (None, 0))(a, b)
    x = vmap(normal_scale, (0, None))(z, c)
    x_val = np.array([4.4, 5.5])

    [new_x, new_z], [new_x2], [new_x_val] = normal_normal.apply_to_node(
        x, [x, z], [x], [x_val]
    )

    assert new_x.cond_dist == VMapDist(normal_scale, (None, 0), 2)
    assert new_z.cond_dist == VMapDist(normal_scale, (0, 0), 2)
    assert new_x2 is new_x
    assert new_x_val is x_val


# def test_vmapped_constant():
#     print("making new regenerator")
#     a = vmap(lambda: makerv(2.0), None, axis_size=3)()
#     b = vmap(lambda: makerv(3.0), None, axis_size=3)()
#     c = vmap(lambda ai, bi: ai + bi)(a, b)
#     [new_c], [], [] = constant_op.apply_to_node(c, [c], [], [])
#
#     assert new_c.cond_dist == VMapDist(Constant(5.0), (), 3)
