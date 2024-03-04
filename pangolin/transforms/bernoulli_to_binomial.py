from .. import interface
from .transforms import InapplicableTransform
from .local_transforms import LocalTransform

np = interface.np
from .. import inference_numpyro


def bernoulli_to_binomial_extractor(node):
    return node


def bernoulli_to_binomial_regenerator(
    node, parents, is_observed, has_observed_descendent, pars_included
):
    if not isinstance(node.cond_dist, interface.VMapDist):
        raise InapplicableTransform("node not vmapped")
    if node.cond_dist.in_axes != (None,):
        raise InapplicableTransform("node in_axes not None")
    if node.cond_dist.base_cond_dist != interface.bernoulli:
        raise InapplicableTransform("base dist not bernoulli")
    if not is_observed:
        raise InapplicableTransform("not observed")

    n = interface.makerv(node.cond_dist.axis_size)
    # p = node.parents[0]
    p = parents[0]

    new_node = interface.binomial(n, p)
    print(f"{new_node=}")

    return new_node


def bernoulli_to_binomial_observer(val):
    return np.array(sum(val))


bernoulli_to_binomial = LocalTransform(
    bernoulli_to_binomial_extractor,
    bernoulli_to_binomial_regenerator,
    bernoulli_to_binomial_observer,
)
"""
Look for vmapped bernoulli RVs with observations and convert them into binomial RVs 
with observations. For any variables that might reference that RV, it is replaced 
with a Constant. (Satisfies the `pangolin.transforms.transforms.Transform` protocol.) 
"""
