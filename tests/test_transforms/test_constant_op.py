from pangolin.interface import normal, normal_scale, makerv, exp, Constant
from pangolin.transforms.constant_op import constant_op

from pangolin.transforms.transforms import (
    apply_transforms,
    InapplicableTransform,
)


def test_apply_to_node1():
    a = makerv(1.25)
    b = makerv(2.0)
    c = a + b

    replacements = constant_op.apply_to_node(c, [], [])
    new_c = replacements[c]
    assert new_c.cond_dist == Constant(3.25)


def test_call1():
    a = makerv(1.25)
    b = makerv(2.0)
    c = a + b
    x = normal(c, 3.0)

    [new_c], [], [] = constant_op([c], [], [])
    assert new_c.cond_dist == Constant(3.25)

    [new_c, new_x], [], [] = constant_op([c, x], [], [])
    assert new_c.cond_dist == Constant(3.25)
    assert new_x.cond_dist == normal_scale

    [new_x], [], [] = constant_op([x], [], [])
    assert new_x.parents[0].cond_dist == Constant(3.25)
    assert new_x.cond_dist == normal_scale


def test_apply_transforms1():
    a = makerv(1.25)
    b = makerv(2.0)
    c = a + b
    x = normal_scale(c, 3.0)

    [new_x, new_c], [], [] = apply_transforms([constant_op], [x, c], None, None)
    assert new_c.cond_dist == Constant(3.25)
    assert new_x.parents[0] == new_c


def test_apply_many():
    a = makerv(1.0)
    x = makerv(0.0)
    for i in range(10):
        x += a
    new_x, _, _ = apply_transforms([constant_op], x, None, None)
    assert new_x.cond_dist == Constant(10.0)
