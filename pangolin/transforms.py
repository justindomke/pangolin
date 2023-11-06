from . import interface
from . import dag
from typing import Sequence, Sized, Union, Optional
from abc import ABC, abstractmethod, abstractproperty
from pangolin import inference_numpyro

CondDist = interface.CondDist
VMapDist = interface.VMapDist
RV = interface.RV

from jax.tree_util import tree_map


class InapplicableTransformationError(Exception):
    pass


tuple_of_RVs = tuple[RV, ...]


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


class Transformation(ABC):
    @abstractmethod
    def regenerate(self, *nodes_parents: tuple[RV, ...]):
        # each parent should be either a RV for a parent (if not regenerated) or RVs for grandparents (if regenerated)
        # return new RV for this node and for regenerated parents
        pass


class TransformationRule(ABC):
    def extract(self, var) -> tuple[RV, ...]:
        """
        The `extract` method takes a single `RV` and gets all the `RV`s that would be
        changed in this transformation rule were applied. This would typically include
        the node itself and might include parents but can in principle include anything.

        There are a few rules that this method must follow:
        * It cannot examine `var.cond_dist` because the same function can be called
        by `VMappedTransformationRule`
        * It cannot look at shapes, for the same reason
        * But it can look at `node.cond_dist.is_random` because this is presered by
        `VMapDist`
        """
        raise NotImplementedError

    # must define get_transform(self, *cond_dists) -> Transformation
    @abstractmethod
    def get_transform(self, *cond_dists: CondDist, pars_included, obs_below):
        """
        This method checks if you can apply this transformation rule.
        The inputs are:
        * `cond_dists`: The `CondDist`s for all the `RV`s from `extract`
        * `pars_included`: A tuple of tuples of bools. This states, for each parent of
        each variable, was that parent extracted?

        If the transformation rule cannot be applied, this should raise a
        `InapplicableTransformationError`. Otherwise it should return a `Transformation`
        """
        raise NotImplementedError()

    def apply(self, node: RV, observed_vars=()):
        # extract variables
        nodes = self.extract(node)

        # parents of nodes
        nodes_parents = tuple(tuple(n.parents) for n in nodes)

        # who has an observed descendent
        obs_below = check_observed_descendents(nodes, observed_vars)
        # which parents are included in the extracted vars
        pars_included = tuple(tuple(p in nodes for p in n.parents) for n in nodes)

        # try to get the transformation
        cond_dists = [n.cond_dist for n in nodes]
        tform = self.get_transform(
            *cond_dists,
            pars_included=pars_included,
            obs_below=obs_below,
        )

        # apply the transformation
        new_nodes = tform.regenerate(*nodes_parents)

        assert len(new_nodes) == len(nodes)

        replacements = dict(tuple(zip(nodes, new_nodes)))
        return replacements


class NonCenteredTransformation(Transformation):
    def regenerate(self, *nodes_parents: tuple[RV, ...]):
        assert len(nodes_parents) == 1, f"should be 1 instead {len(nodes_parents)}"
        assert (
            len(nodes_parents[0]) == 2
        ), f"should be 2 instead {len(nodes_parents[0])}"

        ((loc, scale),) = nodes_parents

        if not isinstance(loc, interface.RV):
            raise Exception("loc is not RV")
        if not isinstance(scale, interface.RV):
            raise Exception("scale is not RV")

        dummy = interface.normal_scale(0, 1)
        new_node = loc + scale * dummy
        return (new_node,)


class NonCenteredNormalTransformationRule(TransformationRule):
    def extract(self, node):
        return (node,)

    def get_transform(
        self, *cond_dists: CondDist, pars_included=None, obs_below=None
    ) -> NonCenteredTransformation:
        if len(cond_dists) != 1 or not isinstance(cond_dists[0], CondDist):
            raise Exception("calling error")

        if cond_dists[0] != interface.normal_scale:
            raise InapplicableTransformationError(f"cond_dist not normal_scale.")

        return NonCenteredTransformation()


class NormalNormalTransformation(Transformation):
    def regenerate(self, *nodes_parents: tuple[RV, ...]) -> tuple[RV, RV]:
        (old_z, c), (a, b) = nodes_parents

        new_x = interface.normal_scale(a, (b**2 + c**2) ** 0.5)
        adj = 1 + c**2 / b**2
        new_mean = a + (new_x - a) / adj
        new_std = b * (1 - 1 / adj) ** 0.5
        new_z = interface.normal_scale(new_mean, new_std)
        return new_x, new_z


class NormalNormalTransformationRule(TransformationRule):
    """
    z = normal(a,b)
    x = normal(z,c)
    transformed into reverse order
    """

    def extract(self, node: RV) -> tuple[RV, RV]:
        if len(node.parents) != 2:
            raise InapplicableTransformationError("doesn't have two parents")
        return node, node.parents[0]

    def get_transform(
        self, *cond_dists: CondDist, pars_included=None, obs_below
    ) -> NormalNormalTransformation:
        cond_dist, loc_cond_dist = cond_dists

        if cond_dist != interface.normal_scale:
            raise InapplicableTransformationError(
                f"cond_dist not normal_scale. instead {cond_dist}"
            )
        if loc_cond_dist != interface.normal_scale:
            raise InapplicableTransformationError(
                f"loc cond_dist not normal_scale. instead {loc_cond_dist}"
            )

        if not obs_below[0]:
            raise InapplicableTransformationError(
                "node not observed, no point in transforming"
            )
        if obs_below[1]:
            raise InapplicableTransformationError(
                "parent observed, no point in transforming"
            )

        return NormalNormalTransformation()


class BetaBinomialTransformation(Transformation):
    # node is binomial
    # parents = [n,p]
    # n is regenerated
    # p is not regenerated

    def regenerate(self, *nodes_parents: tuple[RV, ...]) -> tuple[RV, RV]:
        assert len(nodes_parents) == 2
        n, old_p = nodes_parents[0]
        a, b = nodes_parents[1]
        # a, b = p_parents
        new_node = interface.beta_binomial(n, a, b)
        new_p = interface.beta(
            a + new_node, b + n - new_node
        )  # use OLD node so graph structure is preserved
        return new_node, new_p


class BetaBinomialTransformationRule(TransformationRule):
    def extract(self, node: RV) -> tuple[RV, RV]:
        if len(node.parents) != 2:
            raise InapplicableTransformationError("doesn't have two parents")
        return node, node.parents[1]

    def get_transform(self, *cond_dists: CondDist, pars_included, obs_below):
        if cond_dists[0] != interface.binomial:
            raise InapplicableTransformationError("not binomial")

        if cond_dists[1] != interface.beta:
            raise InapplicableTransformationError("p not beta")

        if not obs_below[0]:
            raise InapplicableTransformationError(
                "node not observed, no point in transforming"
            )
        if obs_below[1]:
            raise InapplicableTransformationError(
                "parent observed, no point in transforming"
            )

        return BetaBinomialTransformation()


class ConstantOpTransformation(Transformation):
    # node is any deterministic cond_dist
    # parents are constants
    # only node is regenerated

    def __init__(self, cond_dist):
        assert not cond_dist.is_random
        self.cond_dist = cond_dist

    def regenerate(self, *nodes_parents: tuple[RV, ...]) -> tuple[RV, RV]:
        # assert len(nodes_parents) == 1
        parents = nodes_parents[0]
        assert len(parents) == len(nodes_parents[1:])
        parent_vals = [p.cond_dist.value for p in parents]
        new_val = inference_numpyro.evaluate(self.cond_dist, *parent_vals)
        new_node = interface.makerv(new_val)
        return (new_node,) + parents


class ConstantOpTransformationRule(TransformationRule):
    def extract(self, node):
        return (node,) + tuple(node.parents)

    def get_transform(self, *cond_dists: CondDist, pars_included, obs_below):
        node_cond_dist = cond_dists[0]
        par_cond_dists = cond_dists[1:]

        if isinstance(node_cond_dist, interface.Constant):
            raise InapplicableTransformationError("node is constant")

        if node_cond_dist.is_random:
            raise InapplicableTransformationError("cond_dist is random")
        for p_cond_dist in par_cond_dists:
            if not isinstance(p_cond_dist, interface.Constant):
                raise InapplicableTransformationError("parent not constant")

        return ConstantOpTransformation(node_cond_dist)


class VMappedTransformationRule(TransformationRule):
    def __init__(self, base_rule):
        self.base_rule = base_rule

    def extract(self, node):
        nodes = self.base_rule.extract(node)
        return nodes

    def get_transform(self, *cond_dists: CondDist, pars_included, obs_below):
        if any(not isinstance(cond_dist, VMapDist) for cond_dist in cond_dists):
            raise InapplicableTransformationError("dist not vmapped")

        # check: for all the cond_dists that point TO EACH OTHER
        # they must map over axis 0 only
        for cond_dist, included in zip(cond_dists, pars_included):
            for in_axis, included in zip(cond_dist.in_axes, included):
                if in_axis != 0 and included:
                    raise InapplicableTransformationError(
                        "regenerated parent not mapped over axis 0"
                    )

        base_cond_dists = tuple(cond_dist.base_cond_dist for cond_dist in cond_dists)  # type: ignore
        in_axes = tuple(cond_dist.in_axes for cond_dist in cond_dists)  # type: ignore

        axis_sizes = tuple(
            cond_dist.axis_size  # type: ignore
            for cond_dist in cond_dists
            if cond_dist.axis_size is not None  # type: ignore
        )

        if len(axis_sizes) > 0:
            assert all(axis_size == axis_sizes[0] for axis_size in axis_sizes)
            axis_size = axis_sizes[0]
        else:
            axis_size = None

        base_tform = self.base_rule.get_transform(
            *base_cond_dists, pars_included=pars_included, obs_below=obs_below
        )

        return VMappedTransformation(base_tform, in_axes, axis_size)


class VMappedTransformation(Transformation):
    def __init__(self, base_tform, in_axes, axis_size):
        self.base_tform = base_tform
        self.in_axes = in_axes
        self.axis_size = axis_size

    def regenerate(self, *nodes_parents: tuple[RV, ...]):
        return interface.vmap(self.base_tform.regenerate, self.in_axes, self.axis_size)(
            *nodes_parents
        )


def apply_transformation_rules(vars, rules, observed_vars=(), max_iter=-1):
    assert isinstance(vars, list)
    for var in vars:
        assert isinstance(var, interface.RV)

    # create a mapping of "old" (current) RVs to "new" RVs
    # old_to_new = {var: var for var in dag.upstream_nodes(vars)}

    old_to_new = {}
    for old_var in dag.upstream_nodes(vars):
        # create a new RV identical to the old one
        new_var = RV(old_var.cond_dist, *[old_to_new[p] for p in old_var.parents])
        old_to_new[old_var] = new_var

    # function to destructively modify the targets of old_to_new
    def apply_replacements(replacements):
        # should this be below the next block? does it matter?
        my_new_nodes = dag.upstream_nodes(list(old_to_new.values()))

        for old_var in old_to_new:
            new_var = old_to_new[old_var]
            if new_var in replacements:
                old_to_new[old_var] = replacements[new_var]

        # make any nodes that point to previous point to new instead
        for my_node in my_new_nodes:
            my_node.parents = tuple(
                replacements[p] if p in replacements else p for p in my_node.parents
            )

    def run_loop():
        new_nodes = list(old_to_new.values())
        up = dag.upstream_nodes(new_nodes)

        # compute observed vars anew
        new_observed_vars = tuple(old_to_new[var] for var in observed_vars)

        for node in up:
            for rule in rules:
                # print(f"trying to apply {rule} to {node}")
                try:
                    replacements = rule.apply(node, observed_vars=new_observed_vars)
                    apply_replacements(replacements)

                    # print(f"applied {type(rule).__name__} to {node}")
                    return True
                except InapplicableTransformationError as e:
                    # print(f"{e=}")
                    continue
        return False

    i = 0
    while run_loop():
        i += 1
        if i == max_iter:
            break

    return [old_to_new[var] for var in vars]


################################################################################
# Simpler way of doing Tforms
################################################################################


class InapplicableTform(Exception):
    pass


class Tform:
    def extract(self, var) -> tuple[RV, ...]:
        """
        The `extract` method takes a single `RV` and gets all the `RV`s that would be
        changed in this transformation rule were applied. This would typically include
        the node itself and might include parents but can in principle include anything.

        There are a few rules that this method must follow:
        * It cannot examine `var.cond_dist` because the same function can be called
        by `VMappedTransformationRule`
        * It cannot look at shapes, for the same reason
        * But it can look at `node.cond_dist.is_random` because this is presered by
        `VMapDist`
        """
        raise NotImplementedError

    # must define get_transform(self, *cond_dists) -> Transformation
    def get_transform(self, *cond_dists, pars_included, obs_below):
        """
        This method checks if you can apply this transformation rule.
        The inputs are:
        * `cond_dists`: The `CondDist`s for all the `RV`s from `extract`
        * `pars_included`: A tuple of tuples of bools. This states, for each parent of
        each variable, was that parent extracted?

        If the transformation rule cannot be applied, this should raise a
        `InapplicableTransformationError`. Otherwise it should return a function that
        can be called on a bunch of random variables and will give new random variables

        """
        raise NotImplementedError()

    def apply(self, node: RV, observed_vars=()):
        # extract variables
        nodes = self.extract(node)

        # parents of nodes
        nodes_parents = tuple(tuple(n.parents) for n in nodes)

        # who has an observed descendent
        obs_below = check_observed_descendents(nodes, observed_vars)
        # which parents are included in the extracted vars
        pars_included = tuple(tuple(p in nodes for p in n.parents) for n in nodes)

        # try to get the transformation
        cond_dists = tuple(n.cond_dist for n in nodes)
        tform = self.get_transform(
            *cond_dists,
            pars_included=pars_included,
            obs_below=obs_below,
        )

        # apply the transformation
        new_nodes = tform(*nodes_parents)

        assert len(new_nodes) == len(nodes)

        replacements = dict(tuple(zip(nodes, new_nodes)))
        return replacements


class NonCenteredNormalTform(Tform):
    def extract(self, node):
        return (node,)

    def get_transform(self, cond_dist, *, pars_included, obs_below):
        if cond_dist != interface.normal_scale:
            raise InapplicableTform(f"cond_dist not normal_scale.")

        def regenerate(node_parents):
            (loc, scale) = node_parents
            dummy = interface.normal_scale(0, 1)
            new_node = loc + scale * dummy
            return (new_node,)

        return regenerate


class NormalNormalTform(Tform):
    """
    z = normal(a,b)
    x = normal(z,c)
    transformed into reverse order
    """

    def extract(self, node):
        if len(node.parents) != 2:
            raise InapplicableTform("doesn't have two parents")
        return node, node.parents[0]

    def get_transform(self, cond_dist, loc_cond_dist, *, pars_included, obs_below):
        if cond_dist != interface.normal_scale:
            raise InapplicableTform(f"cond_dist not normal_scale, instead {cond_dist}")
        if loc_cond_dist != interface.normal_scale:
            raise InapplicableTform(f"loc not normal_scale, instead {loc_cond_dist}")

        if not obs_below[0]:
            raise InapplicableTform("node not observed, no point in transforming")
        if obs_below[1]:
            raise InapplicableTform("parent observed, no point in transforming")

        def regenerate(node_parents, loc_parents):
            (a, b) = loc_parents
            (old_z, c) = node_parents

            new_x = interface.normal_scale(a, (b**2 + c**2) ** 0.5)
            # not totally obvious this is the most stable way of doing things...
            adj = 1 + c**2 / b**2
            new_mean = a + (new_x - a) / adj
            new_std = b * (1 - 1 / adj) ** 0.5
            new_z = interface.normal_scale(new_mean, new_std)
            return new_x, new_z

        return regenerate


class BetaBinomialTform(Tform):
    def extract(self, node):
        if len(node.parents) != 2:
            raise InapplicableTform("doesn't have two parents")
        return node, node.parents[1]

    def get_transform(self, cond_dist, p_cond_dist, *, pars_included, obs_below):
        if cond_dist != interface.binomial:
            raise InapplicableTform("not binomial")

        if p_cond_dist != interface.beta:
            raise InapplicableTform("p not beta")

        if not obs_below[0]:
            raise InapplicableTform("node not observed, no point in transforming")
        if obs_below[1]:
            raise InapplicableTform("parent observed, no point in transforming")

        def regenerate(node_parents, p_parents):
            n, old_p = node_parents
            a, b = p_parents
            new_node = interface.beta_binomial(n, a, b)
            new_p = interface.beta(a + new_node, b + n - new_node)
            return new_node, new_p

        return regenerate


class ConstantOpTform(Tform):
    def extract(self, node):
        return (node,) + tuple(node.parents)

    def get_transform(self, node_cond_dist, *par_cond_dists, pars_included, obs_below):
        if isinstance(node_cond_dist, interface.Constant):
            raise InapplicableTform("node is constant")

        if node_cond_dist.is_random:
            raise InapplicableTform("cond_dist is random")
        for p_cond_dist in par_cond_dists:
            if not isinstance(p_cond_dist, interface.Constant):
                raise InapplicableTform("parent not constant")

        def regenerate(parents, *grandparents):
            assert len(parents) == len(grandparents)
            parent_vals = [p.cond_dist.value for p in parents]
            new_val = inference_numpyro.evaluate(node_cond_dist, *parent_vals)
            new_node = interface.makerv(new_val)
            return (new_node,) + parents

        return regenerate


class VMappedTform(Tform):
    def __init__(self, base_tform):
        self.base_tform = base_tform

    def extract(self, node):
        nodes = self.base_tform.extract(node)
        return nodes

    def get_transform(self, *cond_dists, pars_included, obs_below):
        if any(not isinstance(cond_dist, VMapDist) for cond_dist in cond_dists):
            raise InapplicableTform("dist not vmapped")

        # check: for all the cond_dists that point TO EACH OTHER
        # they must map over axis 0 only
        for cond_dist, included in zip(cond_dists, pars_included):
            # if not isinstance(cond_dist, VMapDist):
            #    raise InapplicableTform("dist not vmapped")
            for in_axis, included in zip(cond_dist.in_axes, included):
                if in_axis != 0 and included:
                    raise InapplicableTform("regenerated parent not mapped over axis 0")

        base_cond_dists = tuple(cond_dist.base_cond_dist for cond_dist in cond_dists)
        in_axes = tuple(cond_dist.in_axes for cond_dist in cond_dists)

        axis_sizes = tuple(
            cond_dist.axis_size  # type: ignore
            for cond_dist in cond_dists
            if cond_dist.axis_size is not None  # type: ignore
        )

        if len(axis_sizes) > 0:
            assert all(axis_size == axis_sizes[0] for axis_size in axis_sizes)
            axis_size = axis_sizes[0]
        else:
            axis_size = None

        base_tform = self.base_tform.get_transform(
            *base_cond_dists, pars_included=pars_included, obs_below=obs_below
        )

        def regenerate(*nodes_parents):
            return interface.vmap(base_tform, in_axes, axis_size)(*nodes_parents)

        return regenerate


def apply_tforms(vars, tforms, observed_vars=(), max_iter=-1):
    assert isinstance(vars, list)
    for var in vars:
        assert isinstance(var, interface.RV)

    # create a mapping of "old" (current) RVs to "new" RVs
    # old_to_new = {var: var for var in dag.upstream_nodes(vars)}

    old_to_new = {}
    for old_var in dag.upstream_nodes(vars):
        # create a new RV identical to the old one
        new_var = RV(old_var.cond_dist, *[old_to_new[p] for p in old_var.parents])
        old_to_new[old_var] = new_var

    # function to destructively modify the targets of old_to_new
    def apply_replacements(replacements):
        # should this be below the next block? does it matter?
        my_new_nodes = dag.upstream_nodes(list(old_to_new.values()))

        for old_var in old_to_new:
            new_var = old_to_new[old_var]
            if new_var in replacements:
                old_to_new[old_var] = replacements[new_var]

        # make any nodes that point to previous point to new instead
        for my_node in my_new_nodes:
            my_node.parents = tuple(
                replacements[p] if p in replacements else p for p in my_node.parents
            )

    def run_loop():
        new_nodes = list(old_to_new.values())
        up = dag.upstream_nodes(new_nodes)

        # compute observed vars anew
        new_observed_vars = tuple(old_to_new[var] for var in observed_vars)

        for node in up:
            for tform in tforms:
                # print(f"trying to apply {rule} to {node}")
                try:
                    replacements = tform.apply(node, observed_vars=new_observed_vars)
                    apply_replacements(replacements)

                    # print(f"applied {type(rule).__name__} to {node}")
                    return True
                except InapplicableTform as e:
                    # print(f"{e=}")
                    continue
        return False

    i = 0
    while run_loop():
        i += 1
        if i == max_iter:
            break

    return [old_to_new[var] for var in vars]


# def simplify(vars, observed_vars=()):
#     # first, create some standard rules
#     rules = []
#     rules.append(NormalNormalTransformationRule())
#     rules.append(BetaBinomialTransformationRule())
#     rules.append(ConstantOpTransformationRule())
#
#     # now add vmapped versions of all those rules
#     new_rules = rules
#     for reps in range(3):
#         new_rules = [VMappedTransformationRule(rule) for rule in new_rules]
#         rules += new_rules
#
#     return apply_transformation_rules(vars, rules, observed_vars)


def simplify(vars, observed_vars=(), max_iter=-1):
    # first, create some standard rules
    tforms = []
    tforms.append(NormalNormalTform())
    tforms.append(BetaBinomialTform())
    tforms.append(ConstantOpTform())

    # now add vmapped versions of all those rules
    new_tforms = tforms
    for reps in range(3):
        new_tforms = [VMappedTform(tform) for tform in new_tforms]
        tforms += new_tforms

    return apply_tforms(vars, tforms, observed_vars, max_iter)
