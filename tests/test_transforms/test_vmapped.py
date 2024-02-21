from pangolin.interface import normal, normal_scale, makerv, exp, Constant, vmap, VMapDist
from pangolin import dag
from pangolin.transforms.transforms_util import replace
import numpy as np

from pangolin import transforms
from pangolin.transforms import (
    normal_normal,
    normal_normal_ez,
    constant_op,
    constant_op_ez,
    apply_transforms,
    InapplicableTransform,
    vmap_local_transform,
    vmap_local_transform_ez,
)

from pytest import mark

vmapped_normal_normal = vmap_local_transform(normal_normal)
vmapped_normal_normal_ez = vmap_local_transform_ez(normal_normal_ez)
vmapped_constant_op = vmap_local_transform(constant_op)
vmapped_constant_op_ez = vmap_local_transform_ez(constant_op_ez)


@mark.parametrize("vmapped_tform", [vmapped_normal_normal, vmapped_normal_normal_ez])
def test_vmap_everything(vmapped_tform):
    a = makerv([1.1, 2.2])
    b = makerv([3.3, 4.4])
    c = makerv([5.5, 6.6])
    z = vmap(normal_scale, 0)(a, b)
    x = vmap(normal_scale, 0)(z, c)

    replacements = vmapped_tform.apply_to_node(x, [x])
    new_x = replacements[x]
    new_z = replacements[z]
    # check
    assert new_x.cond_dist == VMapDist(normal_scale, (0, 0), 2)
    assert new_z.cond_dist == VMapDist(normal_scale, (0, 0), 2)
    assert new_x.shape == (2,)
    assert new_z.shape == (2,)
    assert new_x != new_z
    assert new_x in dag.upstream_nodes(new_z)


@mark.parametrize("vmapped_tform", [vmapped_normal_normal, vmapped_normal_normal_ez])
def test_vmap_only_inside(vmapped_tform):
    a = makerv(1.1)
    b = makerv(2.2)
    c = makerv(3.3)
    z = vmap(normal_scale, None, axis_size=2)(a, b)
    x = vmap(normal_scale, (0, None))(z, c)

    replacements = vmapped_tform.apply_to_node(x, [x])
    new_x = replacements[x]
    new_z = replacements[z]

    # print(f"{replacements.values()=}")
    # print_upstream(tuple(replacements.values()))
    # viz_upstream(tuple(replacements.values())).render("graph")

    assert new_x.cond_dist == VMapDist(normal_scale, (None, None), 2)
    assert new_z.cond_dist == VMapDist(normal_scale, (0, None), 2)

    assert new_x in dag.upstream_nodes(new_z)


@mark.parametrize("vmapped_tform", [vmapped_normal_normal, vmapped_normal_normal_ez])
def test_vmap_half_inside(vmapped_tform):
    a = makerv(1.1)
    b = makerv([2.2, 3.3])
    c = makerv(4.4)
    z = vmap(normal_scale, (None, 0))(a, b)
    x = vmap(normal_scale, (0, None))(z, c)

    replacements = vmapped_tform.apply_to_node(x, [x])
    new_x = replacements[x]
    new_z = replacements[z]

    # viz_upstream(tuple(replacements.values())).render("graph")

    assert new_x.cond_dist == VMapDist(normal_scale, (None, 0), 2)
    assert new_z.cond_dist == VMapDist(normal_scale, (0, 0), 2)


@mark.parametrize("vmapped_tform", [vmapped_constant_op_ez, vmapped_constant_op])
def test_vmapped_constant(vmapped_tform):
    print("making new regenerator")
    a = vmap(lambda: makerv(2.0), None, axis_size=3)()
    b = vmap(lambda: makerv(3.0), None, axis_size=3)()
    c = vmap(lambda ai, bi: ai + bi)(a, b)
    replacements = vmapped_tform.apply_to_node(c, [])

    assert a not in replacements
    assert b not in replacements
    new_c = replacements[c]

    assert new_c.cond_dist == VMapDist(Constant(5.0), (), 3)
