from pangolin.interface import normal, normal_scale
from pangolin import dag
from pangolin.transforms.transforms_util import replace
import numpy as np


def test_replace1():
    x = normal(1.1, 2.2)
    new_x = normal(3.3, 4.4)
    [x2] = replace([x], [x], [new_x])
    assert x2.cond_dist == new_x.cond_dist
    assert x2.parents[0].cond_dist.value == np.array(3.3)
    assert x2.parents[1].cond_dist.value == np.array(4.4)


def test_replace2():
    x = normal(1.1, 2.2)
    y = normal(x, 3.3)
    new_x = normal(4.4, 5.5)
    [x2, y2] = replace([x, y], [x], [new_x])
    assert x2.cond_dist == new_x.cond_dist
    assert y2.parents[0] == x2
    assert y2.parents[1].cond_dist.value == np.array(3.3)


def test_replace3():
    x = normal(1.1, 2.2)
    y = normal(x, 3.3)
    new_x = normal(4.4, 5.5)
    new_y = normal(x, 1)  # points to x — illegal
    try:
        replace([x, y], [x, y], [new_x, new_y])
        assert False, "failed to raise Assertion error for illegal inputs"
    except AssertionError as e:
        assert True


def test_replace4():
    # no replacements, exactly same nodes should be returned!
    x = normal(1.1, 2.2)
    y = normal(x, 1)
    z = normal(y, x**2)
    [x2, y2, z2] = replace([x, y, z], [], [])
    assert x2 == x
    assert y2 == y
    assert z2 == z


def test_replace5():
    # x and y start out independent
    # x not replaced, but y gets reference to "new" dependent x
    # old
    # x = normal(1.1,2.2)       <---
    # y = normal(3.3,4.4)       <---
    # new
    # x = normal(1.1,2.2)       <---
    # new_x = normal(5.5,6.6)
    # new_y = normal(new_x,7.7) <---
    x = normal(1.1, 2.2)
    y = normal(3.3, 4.4)
    new_x = normal(5.5, 6.6)
    new_y = normal(new_x, 7.7)
    [x2, y2] = replace([x, y], [y], [new_y])
    # old nodes: 1.1, 2.2, 3.3, 4.4, x,y
    assert len(dag.upstream_nodes([x, y])) == 6
    assert x2 == x
    # new nodes: 1.1, 2.2, x, 5.5, 6.6, new_x, 7.7, new_y
    assert len(dag.upstream_nodes([x2, y2])) == 8
    assert y2.cond_dist == normal_scale
    assert y2.parents[0] == new_x
    assert y2.parents[1].cond_dist.value == np.array(7.7)
