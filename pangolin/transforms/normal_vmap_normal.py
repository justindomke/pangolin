from .. import interface
from .transforms import InapplicableTransform, Transform
from .local_transforms import LocalTransform
import numpy as np


def normal_vmap_normal_extractor(node):
    if len(node.parents) != 2:
        raise InapplicableTransform("doesn't have two parents")
    loc = node.parents[0]
    return node, loc


def normal_vmap_normal_regenerator(
    targets, parents_of_targets, is_observed, has_observed_descendent, pars_included
):
    node, loc = targets
    node_parents, loc_parents = parents_of_targets

    # print(f"{is_observed=}")
    # print(f"{has_observed_descendent=}")

    if loc.cond_dist != interface.normal_scale:
        raise InapplicableTransform(
            f"cond_dist not normal_scale, instead " f"{type(node.cond_dist)}"
        )

    if not (
        isinstance(node.cond_dist, interface.VMapDist)
        and node.cond_dist.base_cond_dist == interface.normal_scale
        and node.cond_dist.in_axes == (None, None)
    ):
        raise InapplicableTransform(
            f"loc not normal_scale, instead " f"{type(loc.cond_dist)}"
        )

    # print(f"{is_observed=}")

    if not has_observed_descendent[0]:
        raise InapplicableTransform("node not observed, no point in transforming")
    if has_observed_descendent[1]:
        raise InapplicableTransform("parent observed, no point in transforming")

    (mu_z, sigma_z) = loc_parents
    (old_z, sigma_x) = node_parents

    if sigma_x.shape == ():
        sigma_x = sigma_x * np.ones(node.cond_dist.axis_size)

    new_std = (sigma_x**2 + sigma_z**2) ** 0.5

    new_x = interface.vmap(interface.normal_scale, (None, 0))(mu_z, new_std)
    # not totally obvious this is the most stable way of doing things...
    adj = 1 / sigma_z**2 + interface.sum(1 / sigma_x**2)
    new_mean = (mu_z / sigma_z**2 + interface.sum(new_x / sigma_x**2)) / adj
    new_std = (1 / adj) ** 0.5
    new_z = interface.normal_scale(new_mean, new_std)
    return (new_x, new_z)


normal_vmap_normal = LocalTransform(
    normal_vmap_normal_extractor, normal_vmap_normal_regenerator
)
"""
Reverse the order of two chained normal distributions when only bottom has an 
observed descendant. (Satisfies `pangolin.transforms.transforms.Transform` protocol)
"""
