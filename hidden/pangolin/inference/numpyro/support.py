from numpyro import distributions as dist
from pangolin import dag, util, ir


# op_class_to_support = util.WriteOnceDefaultDict(
#     default_factory=lambda key: dist.constraints.real_vector
# )
op_class_to_support = util.WriteOnceDict(
    default_factory=lambda key: dist.constraints.real_vector
)
op_class_to_support[ir.Exponential] = dist.constraints.positive
op_class_to_support[ir.Dirichlet] = dist.constraints.simplex
op_class_to_support[ir.Bernoulli] = dist.constraints.boolean
op_class_to_support[ir.BernoulliLogit] = dist.constraints.boolean


def get_support(op: ir.Op):
    """
    Get support. Only used inside by numpyro_vmap_var_random

    """
    op_class = type(op)
    return op_class_to_support[type(op)]
    # if op in op_class_to_support:
    #    return op_class_to_support[op_class]
    # else:
    #    raise Exception("unsupported op class")
    # elif isinstance(op, ir.Truncated):
    #     if op.lo is not None and op.hi is not None:
    #         return dist.constraints.interval(op.lo, op.hi)
    #     elif op.lo is not None:
    #         assert op.hi is None
    #         return dist.constraints.greater_than(op.lo)
    #     elif op.hi is not None:
    #         assert op.lo is None
    #         return dist.constraints.less_than(op.hi)
    #     else:
    #         assert False, "should be impossible"
