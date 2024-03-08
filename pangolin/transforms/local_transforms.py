from .. import interface
from .. import dag
from .transforms import InapplicableTransform, Transform
from .transforms_util import replace_with_given_old, replace_with_given
from ..interface import CondDist, VMapDist, RV, makerv
from jax import tree_map, tree_util  # type: ignore
from .. import util
import numpy as np
from ..util import tree_map_with_none_as_leaf, tree_map_preserve_none
from typing import Sequence


def check_observed_descendents(nodes, observed_nodes):
    """
    which nodes in `nodes` have descendents in `observed_nodes` if you aren't allowed
    to traverse links BETWEEN nodes in `nodes`?
    """

    # first, find all the upstream nodes if you can't go into `nodes`
    block_condition = None

    flat_nodes, _ = tree_util.tree_flatten(nodes)

    def link_block_condition(node1, node2):
        return node1 in flat_nodes and node2 in flat_nodes

    blocked_upstream = dag.upstream_nodes(
        observed_nodes, block_condition, link_block_condition
    )

    # return tuple(node in blocked_upstream for node in nodes)
    return tree_map(lambda node: node in blocked_upstream, nodes)


def vmap_regenerator(base_regenerator):
    """
    utility to turn a regenerator function into a vmapped version
    """

    def new_regenerator(
        vars, parents_of_vars, is_observed, has_observed_descendent, pars_included
    ):
        def check_vmap_dist(var):
            if not isinstance(var.cond_dist, VMapDist):
                raise InapplicableTransform(f"dist not vmapped {var.cond_dist}")

        tree_map(check_vmap_dist, vars)

        # all vars that point TO EACH OTHER must map over axis 0 only
        def check_axes(var, included):
            for in_axis, in_included in zip(var.cond_dist.in_axes, included):
                if in_axis != 0 and in_included:
                    raise InapplicableTransform(
                        "regenerated parent not mapped over axis 0"
                    )

        tree_map(check_axes, vars, pars_included)

        vars_in_axes = tree_map(lambda node: 0, vars)
        parents_in_axes = tree_map(lambda node: node.cond_dist.in_axes, vars)
        in_axes = vars_in_axes, parents_in_axes

        flat_vars, _ = tree_util.tree_flatten(vars)
        axis_sizes = tuple(
            var.cond_dist.axis_size
            for var in flat_vars
            if var.cond_dist.axis_size is not None
        )

        if len(axis_sizes) > 0:
            assert all(axis_size == axis_sizes[0] for axis_size in axis_sizes)
            axis_size = axis_sizes[0]
        else:
            axis_size = None

        def myfun(vars, info_vars):
            return base_regenerator(
                vars, info_vars, is_observed, has_observed_descendent, pars_included
            )

        # print(f"{in_axes=}")
        # print(f"{vars=}")
        # print(f"{parents_of_vars=}")
        # print(f"{is_observed=}")

        return interface.vmap(myfun, in_axes, axis_size)(vars, parents_of_vars)

    return new_regenerator


################################################################################
# The transform class
################################################################################


def is_rv_non_rv_pair(x):
    return isinstance(x, tuple) and isinstance(x[0], RV) and not isinstance(x[1], RV)


def is_leaf(x):
    """
    A method to find leaves in pytrees that might contain var/val pairs as well as
    just RVs. This prevents jax.tree_util methods from recusing inside the pair.
    """
    return isinstance(x, RV) or is_rv_non_rv_pair(x)


def default_observer(observations):
    return tree_map(lambda obs: None, observations)


class LocalTransform:
    """
    A local transform is a transform created from an `extractor` (which extracts
    nodes from the graph without looking at distributions) and a `regenerator` (which
    tries to transform those nodes using distributions but without looking at the
    graph). It is designed this way to facilitate working with `vmap`—if an extractor
    and regenerator are created following the rules, then it will automatically work
    even when things are vectorized.
    """

    def __init__(self, extractor, regenerator, observer=None):
        """
        Create a LocalTransform by providing two functions that obey the rules below.
        """

        self.extractor = extractor
        """
        `nodes = extractor(var)` takes a single `RV` as input (`var`) and returns a
        pytree of `RV`s (`nodes`) that would be replaced if this transformation were to go
        ahead.
        * This function should *not* look at the `cond_dist` for any random
        variables
        * This function is free to examine the graph upstream of `var` using `.parents`
        and return any `RV`s in the graph organized into a pytree in whatever way is
        convenient.
        * This function would *often* include `var` in `nodes` but this is not required.
        * If this transformation is inapplicable based solely on the graph structure,
        then this function should raise an `InapplicableTransformation` exception.
        """
        self.regenerator = regenerator
        """
        `new_nodes = regenerator(nodes, parents, has_observed_descendent, pars_included)`
        takes the `RV`s returned by the extractor and returns new `RV`s that should
        replace them.
        * `nodes` is a pytree of `RV`s as returned by `extractor`. The function is free
        to examine the `cond_dist` for each of these but should not examine the graph
        structure.
        * `parents` is a pytree of the parents, with each `RV` in `nodes` replaced by
        a tuple of `RV`s in parents.
        * `has_observed_descendent` is a pytree with the same structure as `nodes`
        indicating if each has an observed descendent.
        * `pars_included` is a pytree with the same structure as `parents` with a boolean
        for each indicating if that parent aslo appears in `nodes`.
        * If this transformation is inapplicable based on the `cond_dist`s present in
        each node and/or which parents are included or have observed descendents,
        then this function should raise an `InapplicableTransformation` exception.
        """
        if observer is None:
            observer = default_observer

        self.observer = observer

    def apply_to_node(self, node, vars, given, vals):
        """
        Given a particular `RV` `node` and a list of observed `RV`s, try to apply the
        transformation. If it doesn't work, and the RV is vmapped, then try vmapping
        the regenerator and applying it that way instead.
        """
        # TODO: better testing of inputs, things returned from the local transform

        nodes = self.extractor(node)
        parents = tree_map(lambda x: x.parents, nodes)

        def lookup_val(x):
            return vals[given.index(x)] if x in given else None

        is_observed = tree_map(lambda x: x in given, nodes)
        observations = tree_map(lookup_val, nodes)
        has_observed_descendent = check_observed_descendents(nodes, given)
        flat_nodes, tree1 = tree_util.tree_flatten(nodes)
        pars_included = tree_map(lambda x: x in flat_nodes, parents)

        inputs = [nodes, parents, is_observed, has_observed_descendent, pars_included]

        try:
            new_nodes = self.regenerator(*inputs)
            new_vals = self.observer(observations)
        except InapplicableTransform as e:
            # if failed, try vmapping
            if isinstance(node.cond_dist, VMapDist):
                new_nodes = vmap_regenerator(self.regenerator)(*inputs)
                new_vals = util.map_inside_tree(self.observer, observations)
            else:
                raise e

        vars, given, vals = replace_with_given(
            vars, given, vals, nodes, new_nodes, new_vals
        )

        return vars, given, vals

    def __call__(self, vars, given, vals):
        util.assert_is_sequence_of(vars, interface.RV)
        util.assert_is_sequence_of(given, interface.RV)

        for node in dag.upstream_nodes(vars + given):
            try:
                return self.apply_to_node(node, vars, given, vals)
            except InapplicableTransform as e:
                continue
        raise InapplicableTransform("No nodes found to apply local transform")
