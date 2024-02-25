from .. import interface
from .transforms import InapplicableTransform
from .local_transforms import LocalTransform
from .. import inference_numpyro


def constant_op_extractor(node):
    return node


def constant_op_regenerator(
    node, parents, is_observed, has_observed_descendent, pars_included
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


constant_op = LocalTransform(constant_op_extractor, constant_op_regenerator)
"""
If deterministic functions are applied to Constant RVs, pre-compute them.
(Satisfies `pangolin.transforms.transforms.Transform` protocol)
"""
