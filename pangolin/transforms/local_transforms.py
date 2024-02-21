from .. import interface
from .. import dag
from .transforms import InapplicableTransform
from .transforms_util import replace_with_given

CondDist = interface.CondDist
VMapDist = interface.VMapDist
RV = interface.RV

from jax import tree_map, tree_util


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

    # return tuple(node in blocked_upstream for node in nodes)
    return tree_map(lambda node: node in blocked_upstream, nodes)


################################################################################
# The transform class
################################################################################


class LocalTransform:
    """
    A local transform is a transform created from an `extractor` and a `regenerator`
    """

    def __init__(self, extractor, regenerator):
        self.extractor = extractor
        self.regenerator = regenerator

    def apply_to_node(self, node, observed_vars):
        # extract variables
        nodes = self.extractor(node)
        parents = tree_map(lambda x: x.parents, nodes)

        flat_nodes, tree1 = tree_util.tree_flatten(nodes)
        pars_included = tree_map(lambda x: x in flat_nodes, parents)
        has_observed_descendent = check_observed_descendents(nodes, observed_vars)

        new_nodes = self.regenerator(
            nodes,
            parents,
            pars_included=pars_included,
            has_observed_descendent=has_observed_descendent,
        )

        flat_new_nodes, tree2 = tree_util.tree_flatten(new_nodes)
        assert tree1 == tree2

        replacements = dict(tuple(zip(flat_nodes, flat_new_nodes)))
        return replacements

    def __call__(self, vars, given, vals):
        assert isinstance(vars, list)
        for var in vars:
            assert isinstance(var, interface.RV)
        assert isinstance(given, list)
        for var in given:
            assert isinstance(var, interface.RV)

        for node in dag.upstream_nodes(vars + given):
            try:
                replacements = self.apply_to_node(node, observed_vars=given)
                new_vars, new_given = replace_with_given(
                    vars, given, replacements.keys(), replacements.values()
                )
                return new_vars, new_given, vals
            except InapplicableTransform as e:
                continue
        raise InapplicableTransform("No nodes found to apply local transform")
