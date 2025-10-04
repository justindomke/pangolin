from pangolin import ir


def is_continuous(op: ir.Op):
    continuous_dists = (
        ir.Normal,
        ir.Beta,
        ir.Cauchy,
        ir.Exponential,
        ir.Dirichlet,
        ir.Gamma,
        ir.Lognormal,
        ir.MultiNormal,
        ir.Poisson,
        ir.StudentT,
        ir.Uniform,
    )
    discrete_dists = (
        ir.Bernoulli,
        ir.BernoulliLogit,
        ir.BetaBinomial,
        ir.Binomial,
        ir.Categorical,
        ir.Multinomial,
    )

    if not op.random:
        raise ValueError("is_continuous only handles random ops")
    elif isinstance(op, ir.VMap):
        return is_continuous(op.base_op)
    elif isinstance(op, ir.Composite):
        return is_continuous(op.ops[-1])
    elif isinstance(op, ir.Autoregressive):
        return is_continuous(op.base_op)
    elif isinstance(op, continuous_dists):
        return True
    elif isinstance(op, discrete_dists):
        return False
    else:
        raise NotImplementedError(f"is_continuous doesn't not know to handle {op}")
