from .. import dag
from .transforms import InapplicableTransform
from .transforms_util import bin_vars, replace_with_given


def duplicate_deterministic(vars, given, vals):
    """
    Search for duplicate deterministic RVs with same CondDist and same parents. If
    they exist, merge them together.
    """

    all_vars = dag.upstream_nodes(vars + given)

    def signature(var):
        return (var.cond_dist, *var.parents)

    def filter(var):
        return not var.cond_dist.random

    binned_vars = bin_vars(all_vars, filter, signature)

    for sig in binned_vars:
        old = binned_vars[sig]
        if len(old) > 1:
            new = [old[0]] * len(old)
            print(f"replacing {old} with {new}")
            new_vars, new_given = replace_with_given(vars, given, old, new)
            return new_vars, new_given, vals
    raise InapplicableTransform("no duplicate deterministic nodes")
