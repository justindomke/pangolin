from pangolin.interface import normal, normal_scale, makerv, exp, Constant, vmap, VMapDist
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

    replacements = normal_normal.apply_to_node(x, [x], [x_val])
    new_x = replacements[x]
    new_z = replacements[z]
    # check
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

    replacements = normal_normal.apply_to_node(x, [x], [x_val])
    new_x = replacements[x]
    new_z = replacements[z]

    # print(f"{replacements.values()=}")
    # print_upstream(tuple(replacements.values()))
    # viz_upstream(tuple(replacements.values())).render("graph")

    assert new_x.cond_dist == VMapDist(normal_scale, (None, None), 2)
    assert new_z.cond_dist == VMapDist(normal_scale, (0, None), 2)

    assert new_x in dag.upstream_nodes(new_z)


def test_vmap_half_inside():
    a = makerv(1.1)
    b = makerv([2.2, 3.3])
    c = makerv(4.4)
    z = vmap(normal_scale, (None, 0))(a, b)
    x = vmap(normal_scale, (0, None))(z, c)
    x_val = np.array([4.4, 5.5])

    replacements = normal_normal.apply_to_node(x, [x], [x_val])
    new_x = replacements[x]
    new_z = replacements[z]

    # viz_upstream(tuple(replacements.values())).render("graph")

    assert new_x.cond_dist == VMapDist(normal_scale, (None, 0), 2)
    assert new_z.cond_dist == VMapDist(normal_scale, (0, 0), 2)


def test_vmapped_constant():
    print("making new regenerator")
    a = vmap(lambda: makerv(2.0), None, axis_size=3)()
    b = vmap(lambda: makerv(3.0), None, axis_size=3)()
    c = vmap(lambda ai, bi: ai + bi)(a, b)
    replacements = constant_op.apply_to_node(c, [], [])

    assert a not in replacements
    assert b not in replacements
    new_c = replacements[c]

    assert new_c.cond_dist == VMapDist(Constant(5.0), (), 3)
