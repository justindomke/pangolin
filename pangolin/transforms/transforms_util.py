from .. import util
from .. import dag
from ..ir import RV


def replace(vars, old, new):
    """
    Given some set of `RV`s, replace some old ones with new ones
    rules: nodes in `new` cannot point to nodes in `old`
    """

    for n in new:
        if any(p in old for p in n.parents):
            assert False, "new nodes shouldn't point to replaced nodes"

    all_vars = dag.upstream_nodes(vars)
    replacements = dict(zip(old, new))

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


def replace_with_given(vars, given, old, new):
    all_vars = vars + given
    new_all_vars = replace(all_vars, old, new)
    return new_all_vars[: len(vars)], new_all_vars[len(vars) :]


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
