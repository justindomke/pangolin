from .. import interface
from .transforms import InapplicableTransform
from .local_transforms import LocalTransform


def normal_normal_extractor(node):
    if len(node.parents) != 2:
        raise InapplicableTransform("doesn't have two parents")
    loc = node.parents[0]
    return node, loc


def normal_normal_regenerator(
    targets, parents_of_targets, *, pars_included, has_observed_descendent
):
    node, loc = targets
    node_parents, loc_parents = parents_of_targets

    if node.cond_dist != interface.normal_scale:
        raise InapplicableTransform(
            f"cond_dist not normal_scale, instead " f"{type(node.cond_dist)}"
        )
    if loc.cond_dist != interface.normal_scale:
        raise InapplicableTransform(
            f"loc not normal_scale, instead " f"{type(loc.cond_dist)}"
        )

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
    return new_x, new_z


normal_normal = LocalTransform(normal_normal_extractor, normal_normal_regenerator)
