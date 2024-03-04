from .. import util
from .. import dag
from ..ir import RV, Constant, makerv
from jax import tree_util


def replace(vars, replacements):
    """
    Given some set of `RV`s, replace some old ones with new ones
    rules: nodes in `new` cannot point to nodes in `old`
    """

    old = replacements.keys()
    new = replacements.values()

    for n in new:
        if any(p in old for p in n.parents):
            assert False, "new nodes shouldn't point to replaced nodes"

    all_vars = dag.upstream_nodes(vars)
    # replacements = dict(zip(old, new))

    old_to_new = {}
    for var in all_vars:
        if var in replacements:
            new_var = replacements[var]
        else:
            new_pars = tuple(old_to_new[p] for p in var.parents)
            if new_pars == var.parents:
                new_var = var
            else:
                new_var = RV(var.cond_dist, *new_pars)

        old_to_new[var] = new_var
    return [old_to_new[v] for v in vars]


def replace_with_given_old(vars, given, vals, replacements):
    all_vars = vars + given
    new_all_vars = replace(all_vars, replacements)
    return new_all_vars[: len(vars)], new_all_vars[len(vars) :], vals


# replace_with_given = replace_with_given_old


def replace_with_given(vars, given, vals, nodes, new_nodes, new_vals):
    """
    Take a query triplet defined by (vars, given vals) and replace each node in
    **nodes** with the corresponding node in **new_nodes** and the corresponding
    value in **new_vals** (if not None).

    If the value corresponding to a node is None, then that node is just replaced.

    If the value *is* none, then the variable is replaced by the corresponding
    new_node in **given** but replaced everywhere else by a new Constant. This makes
    it possible to use transformations that change observed nodes without messing up
    the user's query.
    """

    util.assert_all_leaves_instance_of_with_none(vars, RV)
    util.assert_all_leaves_instance_of_with_none(given, RV)
    util.assert_all_leaves_instance_of_with_none(nodes, RV)
    util.assert_all_leaves_instance_of_with_none(new_nodes, RV)

    # TODO: not efficient at all! shouldn't call into replace_with_given_old but instead
    # just do the work directly
    def process(old_node, new_node, new_val):
        nonlocal vars, given, vals
        if new_val is None:
            vars, given, vals = replace_with_given_old(
                vars, given, vals, {old_node: new_node}
            )
        else:
            # create new constant RV to represent old node
            idx = given.index(old_node)
            old_val = vals[idx]
            new_constant_node = makerv(old_val)
            vars, given, vals = replace_with_given_old(
                vars, given, vals, {old_node: new_constant_node}
            )

            # remove previous given value, replace it with new one
            given = util.replace_in_sequence(given, idx, new_node)
            vals = util.replace_in_sequence(vals, idx, new_val)

    # I think no need to preserve None since nodes always non-None
    tree_util.tree_map(process, nodes, new_nodes, new_vals)
    return vars, given, vals


# def replace_with_given(vars, given, vals, replacements):
#     """
#     Given some set of `RV`s, replace some old ones with new ones
#     rules: nodes in `new` cannot point to nodes in `old`
#     """
#
#     assert len(given) == len(vals)
#
#     old = replacements.keys()
#     new = replacements.values()
#
#     for n in new:
#         if isinstance(n, tuple):
#             assert len(n) == 2
#             n, val = n
#         if any(p in old for p in n.parents):
#             assert False, "new nodes shouldn't point to replaced nodes"
#
#     all_vars = dag.upstream_nodes(vars + given)
#
#     old_to_new = {}
#     old_to_new_given = {}
#     old_to_new_val = {}
#     for var in all_vars:
#         if var in replacements:
#             rep = replacements[var]
#             if isinstance(rep, tuple):
#                 # if a tuple is provided, then:
#                 # 1st output is new GIVEN variable
#                 # 2nd output is new VALUE for that given variable
#                 # create new constant to represent old variable
#                 assert len(rep) == 2
#                 assert var in given
#                 new_given, new_val = rep
#                 old_to_new_given[var] = new_given
#                 old_to_new_val[var] = new_val
#                 new_var = makerv(vals[given.index(var)])
#             else:
#                 new_var = rep
#         else:
#             new_pars = tuple(old_to_new[p] for p in var.parents)
#             if new_pars == var.parents:
#                 new_var = var
#             else:
#                 new_var = RV(var.cond_dist, *new_pars)
#         old_to_new[var] = new_var
#
#     new_vars = [old_to_new[var] for var in vars]
#     new_given = [
#         old_to_new_given[v] if v in old_to_new_given else old_to_new[v] for v in given
#     ]
#     new_vals = [
#         old_to_new_val[v] if v in old_to_new_val else vals[given.index(v)] for v in given
#     ]
#     return new_vars, new_given, new_vals


def bin_vars(vars, filter, signature):
    binned_vars = {}
    for var in vars:
        if filter(var):
            sig = signature(var)
            if sig not in binned_vars:
                binned_vars[sig] = [var]
            else:
                binned_vars[sig].append(var)
    return binned_vars
