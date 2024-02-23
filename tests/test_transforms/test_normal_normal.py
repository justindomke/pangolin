from pangolin.interface import normal, normal_scale, makerv, exp
from pangolin import dag
from pangolin.transforms.transforms_util import replace
import numpy as np  # type: ignore

from pangolin import transforms
from pangolin.transforms.transforms import apply_transforms, InapplicableTransform
from pangolin.transforms.normal_normal import normal_normal


def test_apply_to_node1():
    a = makerv(1.1)
    b = makerv(2.2)
    c = makerv(3.3)
    z = normal_scale(a, b)
    x = normal_scale(z, c)

    [new_x], [new_x2], [new_x_val] = normal_normal.apply_to_node(x, [x], [x], [1.0])
    # new_x = replacements[x]
    # new_z = replacements[z]
    # new_z = new_x2.parents[0]

    assert new_x.cond_dist == normal_scale
    # assert new_z.cond_dist == normal_scale
    # assert new_x != new_z
    # assert new_x in dag.upstream_nodes(new_z)

    try:
        normal_normal.apply_to_node(x, [x], [], [])
        assert False, "failed to raise error"
    except InapplicableTransform:
        pass


def test_apply_to_node2():
    a = makerv(1.1)
    b = makerv(2.2)
    c = makerv(3.3)
    z = normal_scale(a, b)
    x = normal_scale(z, c)
    x_val = np.array(1.0)

    [new_x, new_z], [new_x2], [new_x_val] = normal_normal.apply_to_node(
        x, [x, z], [x], [x_val]
    )

    assert new_x.cond_dist == normal_scale
    assert new_z.cond_dist == normal_scale
    assert new_x != new_z
    assert new_x in dag.upstream_nodes(new_z)
    assert new_x2 == new_x
    assert new_x_val == x_val

    try:
        normal_normal.apply_to_node(x, [x, z], [], [])
        assert False, "failed to raise error"
    except InapplicableTransform:
        pass


def test_call1():
    a = makerv(1.1)
    b = makerv(2.2)
    c = makerv(3.3)
    z = normal_scale(a, b)
    x = normal_scale(z, c)
    x_val = np.array(2.0)

    [new_x], [new_x2], [new_x_val] = normal_normal([x], [x], [x_val])
    assert new_x != x
    assert new_x2 == new_x
    assert new_x_val == x_val

    [new_x, new_z], [new_x2], [new_x_val] = normal_normal([x, z], [x], [x_val])
    assert new_x != x
    assert new_x2 == new_x
    assert new_x_val == x_val
    assert new_z != new_x
    assert new_x in dag.upstream_nodes(new_z)

    try:
        normal_normal([x, z], [], [x_val])
        assert False, "failed to raise error"
    except InapplicableTransform:
        pass


def test_apply_transforms1():
    a = makerv(1.1)
    b = makerv(2.2)
    c = makerv(3.3)
    z = normal_scale(a, b)
    x = normal_scale(z, c)
    x_val = np.array(2.0)

    new_x, [new_x2], [new_x_val] = apply_transforms([normal_normal], x, x, x_val)
    assert new_x != x
    assert new_x2 == new_x
    assert new_x_val == x_val

    [new_x, new_z], [new_x2], [new_x_val] = apply_transforms(
        [normal_normal], [x, z], x, x_val
    )
    assert new_x != x
    assert new_x2 == new_x
    assert new_x_val == x_val
    assert new_z != new_x
    assert new_x in dag.upstream_nodes(new_z)

    # without observing x, nothing should happen
    [new_x2, new_z2], [], [] = apply_transforms([normal_normal], [x, z], None, None)
    assert new_x2 == x
    assert new_z2 == z
