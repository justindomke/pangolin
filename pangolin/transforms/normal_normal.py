from transforms import InapplicableTransform
from .. import interface
from local_transforms import LocalTransform


def normal_normal_extractor(node):
    if len(node.parents) != 2:
        raise InapplicableTransform("doesn't have two parents")
    return node, node.parents[0]


def normal_normal_transformer(
    cond_dist,
    loc_cond_dist,
    *,
    pars_included,
    obs_below,
):
    if cond_dist != interface.normal_scale:
        raise InapplicableTransform(f"cond_dist not normal_scale, instead {cond_dist}")
    if loc_cond_dist != interface.normal_scale:
        raise InapplicableTransform(f"loc not normal_scale, instead {loc_cond_dist}")

    if not obs_below[0]:
        raise InapplicableTransform("node not observed, no point in transforming")
    if obs_below[1]:
        raise InapplicableTransform("parent observed, no point in transforming")

    def regenerate(node_parents, loc_parents):
        (a, b) = loc_parents
        (old_z, c) = node_parents
        # (a, b), (old_z, c) = info_nodes

        new_x = interface.normal_scale(a, (b**2 + c**2) ** 0.5)
        # not totally obvious this is the most stable way of doing things...
        adj = 1 + c**2 / b**2
        new_mean = a + (new_x - a) / adj
        new_std = b * (1 - 1 / adj) ** 0.5
        new_z = interface.normal_scale(new_mean, new_std)
        return new_x, new_z

    return regenerate


normal_normal_tform = LocalTransform(normal_normal_extractor, normal_normal_transformer)
