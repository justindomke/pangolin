from .. import util
from .. import dag
from ..ir import RV, Constant, makerv


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


def replace_with_given(vars, given, vals, replacements):
    """
    Given some set of `RV`s, replace some old ones with new ones
    rules: nodes in `new` cannot point to nodes in `old`
    """

    assert len(given) == len(vals)

    old = replacements.keys()
    new = replacements.values()

    for n in new:
        if isinstance(n, tuple):
            assert len(n) == 2
            n, val = n
        if any(p in old for p in n.parents):
            assert False, "new nodes shouldn't point to replaced nodes"

    all_vars = dag.upstream_nodes(vars + given)

    old_to_new = {}
    old_to_new_given = {}
    old_to_new_val = {}
    for var in all_vars:
        if var in replacements:
            rep = replacements[var]
            if isinstance(rep, tuple):
                # if a tuple is provided, then:
                # 1st output is new GIVEN variable
                # 2nd output is new VALUE for that given variable
                # create new constant to represent old variable
                assert len(rep) == 2
                assert var in given
                new_given, new_val = rep
                old_to_new_given[var] = new_given
                old_to_new_val[var] = new_val
                new_var = makerv(vals[given.index(var)])
            else:
                new_var = rep
        else:
            new_pars = tuple(old_to_new[p] for p in var.parents)
            if new_pars == var.parents:
                new_var = var
            else:
                new_var = RV(var.cond_dist, *new_pars)
        old_to_new[var] = new_var

    new_vars = [old_to_new[var] for var in vars]
    new_given = [
        old_to_new_given[v] if v in old_to_new_given else old_to_new[v] for v in given
    ]
    new_vals = [
        old_to_new_val[v] if v in old_to_new_val else vals[given.index(v)] for v in given
    ]
    return new_vars, new_given, new_vals


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
