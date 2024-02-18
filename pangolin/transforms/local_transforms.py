from .. import interface
from .. import dag

CondDist = interface.CondDist
VMapDist = interface.VMapDist
RV = interface.RV


def check_observed_descendents(nodes, observed_nodes):
    """
    which nodes in `nodes` have descendents in `observed_nodes` if you aren't allowed
    to traverse links BETWEEN nodes in `nodes`?
    """

    # first, find all the upstream nodes if you can't go into `nodes`
    block_condition = None

    def link_block_condition(node1, node2):
        return node1 in nodes and node2 in nodes

    blocked_upstream = dag.upstream_nodes(
        observed_nodes, block_condition, link_block_condition
    )

    return tuple(node in blocked_upstream for node in nodes)


################################################################################
# The transform class
################################################################################


class LocalTransform:
    """
    A local transform is a transform created from an `extractor` and a `transformer`
    """

    def __init__(self, extractor, transformer):
        self.extractor = extractor
        self.transformer = transformer

    def apply_to_node(self, node, observed_vars):
        # extract variables
        nodes = self.extractor(node)

        # parents of nodes
        nodes_parents = tuple(tuple(n.parents) for n in nodes)

        # who has an observed descendent
        obs_below = check_observed_descendents(nodes, observed_vars)
        # which parents are included in the extracted vars
        pars_included = tuple(tuple(p in nodes for p in n.parents) for n in nodes)

        # try to get the transformation
        cond_dists = tuple(n.cond_dist for n in nodes)

        tform = self.transformer(
            *cond_dists,
            pars_included=pars_included,
            obs_below=obs_below,
        )

        # apply the transformation
        new_nodes = tform(*nodes_parents)

        assert len(new_nodes) == len(nodes)

        replacements = dict(tuple(zip(nodes, new_nodes)))
        return replacements
