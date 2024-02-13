from . import dag


class Inference:
    pass

def upstream_with_descendent(requested_vars, given_vars):
    """
    First, find all vars that are upstream (inclusive) of `requested_vars`
    Then, find all the vars that:
    1. Are *downstream* (inclusive) of that set
    2. Have a descendant in `given_nodes`
    3. Have `var.cond_dist.is_random=True`
    """
    upstream_observed_vars = dag.upstream_with_descendent(requested_vars, given_vars)
    return list(x for x in upstream_observed_vars if x.cond_dist.random)
