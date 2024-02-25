from pangolin.interface import (
    normal,
    normal_scale,
    makerv,
    exp,
    Constant,
    bernoulli,
    binomial,
    vmap,
    plate,
    VMapDist,
)
import numpy as np  # type: ignore

from pangolin import transforms
from pangolin.transforms.transforms import (
    apply_transforms,
    InapplicableTransform,
)
from pangolin.transforms.constant_op import constant_op
from pangolin.transforms.bernoulli_to_binomial import (
    bernoulli_to_binomial_regenerator,
    bernoulli_to_binomial_observer,
)
from pangolin.transforms.bernoulli_to_binomial import bernoulli_to_binomial as tform


def test_regenerator():
    x = vmap(bernoulli, None, axis_size=10)(0.2)
    parents = x.parents
    x_val = np.array([1] * 3 + [0] * 7)
    is_observed = x_val
    has_observed_descendent = False
    pars_included = (False,)

    new_x = bernoulli_to_binomial_regenerator(
        x, parents, is_observed, has_observed_descendent, pars_included
    )

    assert new_x.cond_dist == binomial
    # assert new_val == sum(x_val)


def test_observer():
    x = vmap(bernoulli, None, axis_size=10)(0.2)
    parents = x.parents
    x_val = np.array([1] * 3 + [0] * 7)
    is_observed = x_val
    has_observed_descendent = False
    pars_included = (False,)

    new_val = bernoulli_to_binomial_observer(is_observed)
    assert new_val == sum(x_val)


def test_apply_to_node1():
    x = vmap(bernoulli, None, axis_size=10)(makerv(0.2))
    x_val = np.array([1] * 3 + [0] * 7)

    # interface.print_upstream(x)
    # dag.upstream_nodes(x)

    [new_x], [new_x2], [new_x_val] = tform.apply_to_node(x, [x], [x], [x_val])

    print(f"{new_x=}")
    print(f"{new_x2=}")

    assert new_x.cond_dist == Constant(x_val)
    assert new_x2.cond_dist == binomial
    assert new_x_val == np.array(3)


def test_call1():
    x = vmap(bernoulli, None, axis_size=10)(makerv(0.2))
    x_val = np.array([1] * 3 + [0] * 7)

    [new_x], [new_x2], [new_x_val] = tform([x], [x], [x_val])

    assert new_x.cond_dist == Constant(x_val)
    assert new_x2.cond_dist == binomial
    assert new_x_val == np.array(sum(x_val))


def test_call2():
    x = vmap(bernoulli, None, axis_size=10)(makerv(0.2))

    try:
        tform([x], [], [])
        assert False, "failed to raise InapplicableTransform error"
    except InapplicableTransform:
        pass


def test_vmapped_call2():
    params = makerv([0.2, 0.3, 0.4])
    x = plate(params)(lambda param: plate(N=10)(lambda: bernoulli(param)))
    # equivalent but more confusing
    # x = vmap(lambda param: vmap(bernoulli, None, 10)(param))(params)

    x_val = np.round(np.random.rand(3, 10))

    [new_x], [new_x2], [new_x_val] = tform([x], [x], [x_val])

    assert new_x.cond_dist == Constant(x_val)
    assert new_x2.cond_dist == VMapDist(
        binomial,
        (
            None,
            0,
        ),
        3,
    )
    assert np.array_equal(new_x_val, np.sum(x_val, axis=1))
