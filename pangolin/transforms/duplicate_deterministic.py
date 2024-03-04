from .. import dag
from .transforms import InapplicableTransform
from .transforms_util import bin_vars, replace_with_given_old


def duplicate_deterministic(vars, given, vals):
    """
    Search for duplicate deterministic RVs with same CondDist and same parents. If
    they exist, merge them together. (Satisfies
    `pangolin.transforms.transforms.Transform` protocol)

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
            replacements = dict(zip(old, new))
            return replace_with_given_old(vars, given, vals, replacements)
    raise InapplicableTransform("no duplicate deterministic nodes")
