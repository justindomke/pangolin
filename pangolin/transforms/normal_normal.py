from .. import interface
from .transforms import InapplicableTransform, Transform
from .local_transforms import LocalTransform


def normal_normal_extractor(node):
    if len(node.parents) != 2:
        raise InapplicableTransform("doesn't have two parents")
    loc = node.parents[0]
    return node, loc


def normal_normal_regenerator(
    targets, parents_of_targets, is_observed, has_observed_descendent, pars_included
):
    node, loc = targets
    node_parents, loc_parents = parents_of_targets

    # print(f"{is_observed=}")
    # print(f"{has_observed_descendent=}")

    if node.cond_dist != interface.normal_scale:
        raise InapplicableTransform(
            f"cond_dist not normal_scale, instead " f"{type(node.cond_dist)}"
        )
    if loc.cond_dist != interface.normal_scale:
        raise InapplicableTransform(
            f"loc not normal_scale, instead " f"{type(loc.cond_dist)}"
        )

    # print(f"{is_observed=}")

    if not has_observed_descendent[0]:
        raise InapplicableTransform("node not observed, no point in transforming")
    if has_observed_descendent[1]:
        raise InapplicableTransform("parent observed, no point in transforming")

    (a, b) = loc_parents
    (old_z, c) = node_parents

    new_x = interface.normal_scale(a, (b**2 + c**2) ** 0.5)
    # not totally obvious this is the most stable way of doing things...
    adj = 1 + c**2 / b**2
    new_mean = a + (new_x - a) / adj
    new_std = b * (1 - 1 / adj) ** 0.5
    new_z = interface.normal_scale(new_mean, new_std)
    return (new_x, new_z)


normal_normal = LocalTransform(normal_normal_extractor, normal_normal_regenerator)
"""
Reverse the order of two chained normal distributions when only bottom has an 
observed descendant. (Satisfies `pangolin.transforms.transforms.Transform` protocol)
"""
