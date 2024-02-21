from .. import interface
from .transforms import InapplicableTransform
from .local_transforms import LocalTransform, LocalTransformEZ

from .. import inference_numpyro


# Thoughts:
# it's awkward to need to extract the parents here and pass them to regenerate
# just so the cond_dists can be examined by the transformer
# but kinda seems needed to work inside vmap... parents could be vmapped constant and
# this code still works...

def constant_op_extractor(node):
    return node, *node.parents


def constant_op_transformer(
        node_cond_dist,
        *par_cond_dists,
        pars_included,
        has_observed_descendent
):
    if isinstance(node_cond_dist, interface.Constant):
        raise InapplicableTransform("node is constant")

    if node_cond_dist.random:
        raise InapplicableTransform("cond_dist is random")
    for p_cond_dist in par_cond_dists:
        if not isinstance(p_cond_dist, interface.Constant):
            raise InapplicableTransform("parent not constant")

    parent_vals = [d.value for d in par_cond_dists]

    def regenerate(parents, *parents_parents):
        print(f"{parents=}")
        assert len(parents) == len(parents_parents)
        new_val = inference_numpyro.evaluate(node_cond_dist, *parent_vals)
        new_node = interface.makerv(new_val)
        return new_node, *parents

    return regenerate


constant_op = LocalTransform(constant_op_extractor, constant_op_transformer)


def constant_op_extract(node):
    return node


def constant_op_regenerator(
        node,
        parents,
        *,
        has_observed_descendent,
        pars_included
):
    if isinstance(node.cond_dist, interface.Constant):
        raise InapplicableTransform("node is constant")

    if node.cond_dist.random:
        raise InapplicableTransform("cond_dist is random")
    for p in parents:
        if not isinstance(p.cond_dist, interface.Constant):
            raise InapplicableTransform(f"parent not constant ({p.cond_dist})")

    parent_vals = [p.cond_dist.value for p in parents]

    new_val = inference_numpyro.evaluate(node.cond_dist, *parent_vals)
    new_node = interface.makerv(new_val)
    return new_node


constant_op_ez = LocalTransformEZ(constant_op_extract,
                                  constant_op_regenerator)
