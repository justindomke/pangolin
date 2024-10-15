# from . import interface, dag, util, inference
import jax.tree_util
import functools
from jax import numpy as jnp
from numpyro import distributions as dist
from typing import Sequence
import numpyro
from jax import lax
from numpyro.distributions import util as dist_util
from jax.scipy import special as jspecial
from jax import nn as jnn
# import numpy as np
from pangolin.ir.rv import RV
from pangolin import dag, util, ir
from pangolin.interface.interface import OperatorRV
from numpy.typing import ArrayLike
from pangolin.interface import RV_or_array
from pangolin.inference import inference_util
import numpy as np

"""
NumPyro's support for discrete latent variables seems to be rather buggy.
There are sampling methods like [numpyro.infer.DiscreteHMCGibbs](https://num.pyro.ai/en/latest/mcmc.html#id12) 
and [numpyro.infer.MixedHMC](https://num.pyro.ai/en/latest/mcmc.html#id15). These *often* seem to
work, but I've not been able to get them to work *reliably* and *consistently*. Thus, I have limited
support here to cases where discrete variables can be automatically integrated out. This means, in
practice, that you can only have *non-observed* discrete latent variables that are simple (e.g.
Bernoulli) and not "complex" (e.g. autoregressive). *Observed* discrete latent variables can be of
any type.
"""

def get_model_flat(vars: list[RV], given: list[RV], vals: list[RV_or_array]):
    """
    Given a "flat" specification of an inference problem, get a numpyro model.

    The basic algorithm here is quite simple:
    1. Get all variables upstream of `vars` and `given`
    2. Define a new model function that goes through those variables in order, and does the
    appropriate numpyro operations for each

    Parameters
    ----------
    vars
        A flat sequence of random variables to define a model over
    given
        A flat sequence of random variables to condition on
    vals
        A flat sequence of constants (should have `len(vals)==len(given)` and matching shapes for
        each element)

    Returns
    -------
    model
        a new numpyro model, which can be used just like any other numpyro model
    names
        the names of the random variables in the model corresponding to `vars`
    """
    if not isinstance(vars, list):
        raise ValueError("vars must be list")
    if not isinstance(given, list):
        raise ValueError("given must be list")
    if not isinstance(vals, list):
        raise ValueError("vals must be list")
    if not all(isinstance(a, RV) for a in vars):
        raise ValueError("all elements of vars must be RVs")
    if not all(isinstance(a, RV) for a in given):
        raise ValueError("all elements of given must be RVs")
    if len(given) != len(vals):
        raise ValueError(f"length of given ({len(given)}) does not match length of vals ({len(vals)})")

    vals = [jnp.array(a) for a in vals]

    for var, val in zip(given, vals):
        if not util.is_numeric_numpy_array(val):
            raise ValueError("given val {val} not numeric")
        if var.shape != val.shape:
            raise ValueError("given var {var} with shape {var.shape} does not match corresponding given val {val} with shape {val.shape}")


    all_vars = dag.upstream_nodes(tuple(vars) + tuple(given))

    name_to_var = {}
    var_to_name = {}
    varnum = 0
    for var in all_vars:
        name = f"v{varnum}"
        varnum += 1
        name_to_var[name] = var
        var_to_name[var] = name

    def model():
        var_to_numpyro_rv = {}
        name_to_numpyro_rv = {}
        for var in all_vars:
            assert isinstance(var, RV)
            name = var_to_name[var]
            numpyro_pars = [var_to_numpyro_rv[p] for p in var.parents]

            if var in given:
                obs = vals[given.index(var)]
            else:
                obs = None

            # if isinstance(var.op, ir.VMap):
            #    d = deal_with_vmap(var.op, *numpyro_pars, is_observed=obs is not None)
            # else:
            d = numpyro_var(var.op, *numpyro_pars, is_observed=obs is not None)

            if var.op.random:
                assert isinstance(d,
                                  dist.Distribution), "numpyo handler failed to return distribution for random op"
                numpyro_rv = numpyro.sample(name, d, obs=obs)
            else:
                assert isinstance(d,
                                  jnp.ndarray), f"numpyo handler failed to return jax.numpy array for nonrandom op {var=} {d=}"
                numpyro_rv = numpyro.deterministic(name, d)

            var_to_numpyro_rv[var] = numpyro_rv
            name_to_numpyro_rv[name] = numpyro_rv
        return name_to_numpyro_rv

    return model, var_to_name


broadcastable_op_classes = (
ir.Normal, ir.NormalPrec, ir.Bernoulli, ir.BernoulliLogit, ir.Beta, ir.BetaBinomial, ir.Binomial,
ir.Categorical, ir.Cauchy, ir.Exponential, ir.Gamma, ir.LogNormal, ir.Poisson, ir.StudentT,
ir.Uniform)

# ir.Dirichlet?
# ir.MultiNormal?

# handlers that don't need to look at the op itself, just the type
simple_handlers = {
    ir.Normal: dist.Normal,
    ir.NormalPrec: lambda loc, prec: dist.Normal(loc, 1 / prec ** 2),
    ir.Bernoulli: dist.Bernoulli,
    ir.BernoulliLogit: dist.BernoulliLogits,
    ir.Beta: dist.Beta,
    ir.BetaBinomial: lambda n, a, b: dist.BetaBinomial(a, b, n),  # numpyro has a different order
    ir.Binomial: dist.Binomial,
    ir.Categorical: dist.Categorical,
    ir.Cauchy: dist.Cauchy,
    ir.Exponential: dist.Exponential,
    ir.Dirichlet: dist.Dirichlet,
    ir.Gamma: dist.Gamma,
    ir.LogNormal: dist.LogNormal,
    ir.Multinomial: dist.Multinomial,
    ir.MultiNormal: dist.MultivariateNormal,
    ir.Poisson: dist.Poisson,
    ir.StudentT: dist.StudentT,
    ir.Uniform: dist.Uniform,
    ir.Add: lambda a, b: a + b,
    ir.Sub: lambda a, b: a - b,
    ir.Mul: lambda a, b: a * b,
    ir.Div: lambda a, b: a / b,
    ir.Pow: lambda a, b: a ** b,
    ir.Abs: jnp.abs,
    ir.Arccos: jnp.arccos,
    ir.Arccosh: jnp.arccosh,
    ir.Arcsin: jnp.arcsin,
    ir.Arcsinh: jnp.arcsinh,
    ir.Arctan: jnp.arctan,
    ir.Arctanh: jnp.arctanh,
    ir.Cos: jnp.cos,
    ir.Cosh: jnp.cosh,
    ir.Exp: jnp.exp,
    ir.InvLogit: dist.transforms.SigmoidTransform(),
    ir.Log: jnp.log,
    ir.Loggamma: jspecial.gammaln,
    ir.Logit: jspecial.logit,
    ir.Sin: jnp.sin,
    ir.Sinh: jnp.sinh,
    ir.Step: lambda x: jnp.heaviside(x, 0.5),
    ir.Tan: jnp.tan,
    ir.Tanh: jnp.tanh,
    ir.MatMul: jnp.matmul,
    ir.Inv: jnp.linalg.inv,
    ir.Softmax: jnn.softmax,
}

numpyro_handlers = util.WriteOnceDict()


def register_handler(op_class):
    def register(handler):
        numpyro_handlers[op_class] = handler
        return handler

    return register


@register_handler(ir.Constant)
def handle_constant(op: ir.Constant, *, is_observed):
    # return op.value
    return jnp.array(op.value)  # return a jax array, not a numpy array


@register_handler(ir.Index)
def handle_index(op: ir.Index, val, *indices, is_observed):
    stuff = []
    i = 0
    for my_slice in op.slices:
        if my_slice:
            stuff.append(my_slice)
        else:
            stuff.append(indices[i])
            i += 1
    stuff = tuple(stuff)
    return val[stuff]


@register_handler(ir.Sum)
def handle_sum(op: ir.Sum, val, *, is_observed):
    return jnp.sum(val, axis=op.axis)


op_class_to_support = util.WriteOnceDefaultDict(
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


def get_vmap_dummy(op):
    if not isinstance(op, ir.VMap):
        def dummy(*args):
            return args

        return dummy
    else:
        dummy = get_vmap_dummy(op.base_op)
        return jax.vmap(dummy, op.in_axes, axis_size=op.axis_size)


def vmap_numpyro_pars(op: ir.VMap, *numpyro_pars):
    dummy = get_vmap_dummy(op)
    return dummy(*numpyro_pars)


@register_handler(ir.VMap)
def handle_vmap(op: ir.VMap, *numpyro_parents, is_observed):
    # if is simple/broadcastable, use broadcasting vmap
    # elif is nonrandom, use nonrandom
    # elif is observed or is non-discrete use class vmap
    # else, can't handle it

    print(f"{op=}")
    print(f"{vmap_base_is_broadcastable(op)=}")

    if vmap_base_is_broadcastable(op):
        print("a")
        return numpyro_vmap_var_broadcast(op, *numpyro_parents, is_observed=is_observed)
    elif not op.random:
        print("b")
        return numpyro_vmap_var_nonrandom(op, *numpyro_parents, is_observed=is_observed)
    elif is_observed or is_continuous(op):
        print("c")
        return numpyro_vmap_var_random(op, *numpyro_parents, is_observed=is_observed)
    else:
        raise ValueError(
            "NumPyro backend can't handle vmaps over ops that are (1) random (2) discrete (3) non-observed and (4) not 'simple' (basic dists)")


def numpyro_vmap_var_plate(op: ir.VMap, *numpyro_pars, evidence):
    """
    Do a vmap using plates.

    Parameters
    ----------
    op
    numpyro_pars
    evidence

    Returns
    -------
    mapped var
    """

    parents_shapes = [x.shape for x in numpyro_pars]

    remaining_shapes, axis_size = ir.vmap.get_sliced_shapes(
        parents_shapes, op.in_axes, op.axis_size
    )

    plate_name = "i" + str(np.random.randint(100000)) + str(np.random.randint(100000))

    with numpyro.plate(plate_name, axis_size) as i:
        my_numpyro_pars = []
        for x, axis in zip(numpyro_pars, op.in_axes, strict=True):
            slices = [slice(None)]*axis
            new_x = x[*slices,i]
            my_numpyro_pars.append(new_x)
        my_evidence = evidence[i]



def numpyro_vmap_var_broadcast(op: ir.VMap, *numpyro_pars, is_observed):
    assert isinstance(op, ir.VMap)
    assert op.random

    assert vmap_base_is_broadcastable(op)

    numpyro_pars = vmap_numpyro_pars(op, *numpyro_pars)

    while isinstance(op, ir.VMap):
        op = op.base_op

    d = numpyro_var(op, *numpyro_pars, is_observed=is_observed)

    return d


def numpyro_vmap_var_random(op: ir.VMap, *numpyro_parents, is_observed):
    assert isinstance(op, ir.VMap)
    assert op.random

    class NewDist(dist.Distribution):
        @property
        def support(self):
            # TODO:
            # should be a more elegant solution here...
            my_op = op
            while isinstance(my_op, ir.VMap):
                my_op = my_op.base_op
            return get_support(my_op)

        def __init__(self, *args, validate_args=False):
            self.args = args

            # TODO: infer correct batch_shape?
            batch_shape = ()
            parents_shapes = [p.shape for p in args]
            event_shape = op.get_shape(*parents_shapes)

            super().__init__(
                batch_shape=batch_shape,
                event_shape=event_shape,
                validate_args=validate_args,
            )

        def sample(self, key, sample_shape=()):
            assert numpyro.util.is_prng_key(key)
            assert sample_shape == () or sample_shape == (1,)

            def base_sample(key, *args):
                var = numpyro_var(op.base_op, *args, is_observed=is_observed)
                return var.sample(key)

            keys = jax.random.split(key, self.event_shape[0])
            in_axes = (0,) + op.in_axes
            axis_size = op.axis_size
            args = (keys,) + self.args
            if sample_shape == (1,):
                return jnp.array([jax.vmap(base_sample, in_axes, axis_size=axis_size)(*args)])
            elif sample_shape == ():
                return jax.vmap(base_sample, in_axes, axis_size=axis_size)(*args)
            else:
                assert False

        @dist_util.validate_sample
        def log_prob(self, value):
            def base_log_prob(val, *args):
                var = numpyro_var(op.base_op, *args, is_observed=is_observed)
                return var.log_prob(val)

            in_axes = (0,) + op.in_axes
            axis_size = op.axis_size
            args = (value,) + self.args
            ls = jax.vmap(base_log_prob, in_axes, axis_size=axis_size)(*args)
            return jnp.sum(ls)

    return NewDist(*numpyro_parents)


def numpyro_vmap_var_nonrandom(op: ir.VMap, *numpyro_parents, is_observed):
    assert isinstance(op, ir.VMap)
    assert not op.random

    def base_var(*args):
        return numpyro_var(op.base_op, *args, is_observed=is_observed)

    in_axes = op.in_axes
    axis_size = op.axis_size
    args = numpyro_parents
    return jax.vmap(base_var, in_axes=in_axes, axis_size=axis_size)(*args)


def vmap_base_is_broadcastable(op: ir.VMap):
    while isinstance(op, ir.VMap):
        op = op.base_op
    return isinstance(op, broadcastable_op_classes)


def is_continuous(op: ir.Op):
    continuous_dists = (
    ir.Normal, ir.Beta, ir.Cauchy, ir.Exponential, ir.Dirichlet, ir.Gamma, ir.LogNormal,
    ir.MultiNormal, ir.Poisson, ir.StudentT, ir.Uniform)
    discrete_dists = (
    ir.Bernoulli, ir.BernoulliLogit, ir.BetaBinomial, ir.Binomial, ir.Categorical, ir.Multinomial)

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


@register_handler(ir.Composite)
def numpyro_composite_var(op: ir.Composite, *numpyro_parents, is_observed):
    vals = list(numpyro_parents)
    assert len(numpyro_parents) == op.num_inputs
    for my_cond_dist, my_par_nums in zip(op.ops, op.par_nums, strict=True):
        my_parents = [vals[i] for i in my_par_nums]
        new_val = numpyro_var(my_cond_dist, *my_parents, is_observed=is_observed)
        vals.append(new_val)
    return vals[-1]


@register_handler(ir.Autoregressive)
def numpyro_autoregressive_var(op, *numpyro_parents, is_observed):
    if op.random:
        return numpyro_autoregressive_var_random(op, *numpyro_parents, is_observed=is_observed)
    else:
        return numpyro_autoregressive_var_nonrandom(op, *numpyro_parents, is_observed=is_observed)


def handle_autoregressive_inputs(op: ir.Autoregressive, *numpyro_parents):
    for in_axis in op.in_axes:
        assert in_axis in [0,
                           None], "NumPyro only supports Autoregressive with in_axis of 0 or None"

    mapped_parents = tuple(p for (p, in_axis) in zip(numpyro_parents, op.in_axes) if in_axis == 0)
    unmapped_parents = tuple(p for (p, in_axis) in zip(numpyro_parents, op.in_axes) if
                             in_axis is None)

    def merge_args(mapped_args):
        ret = []
        i_mapped = 0
        i_unmapped = 0
        for in_axis in op.in_axes:
            if in_axis == 0:
                ret.append(mapped_args[i_mapped])
                i_mapped += 1
            elif in_axis is None:
                ret.append(unmapped_parents[i_unmapped])
                i_unmapped += 1
            else:
                assert False
        assert i_mapped == len(mapped_args)
        assert i_unmapped == len(unmapped_parents)
        return tuple(ret)

    return mapped_parents, merge_args


def numpyro_autoregressive_var_nonrandom(op: ir.Autoregressive, numpyro_init, *numpyro_parents,
                                         is_observed):
    # numpyro.contrib.control_flow.scan exists but seems very buggy/limited
    assert isinstance(op, ir.Autoregressive)
    assert not op.random

    mapped_parents, merge_args = handle_autoregressive_inputs(op, *numpyro_parents)
    # print(f"{mapped_parents=}")
    # print(f"{numpyro_parents=}")
    # print(f"{merge_args(mapped_parents)=}")
    assert merge_args(mapped_parents) == numpyro_parents

    print(f"{numpyro_init=}")

    def myfun(carry, x):
        # inputs = (carry,) + x
        inputs = (carry,) + merge_args(x)
        # print(f"{inputs=}")
        # print(f"{[a.shape for a in inputs]=}")
        y = numpyro_var(op.base_op, *inputs, is_observed=is_observed)
        return y, y

    # myfun(numpyro_init, tuple(p[0] for p in mapped_parents))

    # carry, ys = jax.lax.scan(myfun, numpyro_init, numpyro_parents, length=op.length)
    carry, ys = jax.lax.scan(myfun, numpyro_init, mapped_parents, length=op.length)
    return ys


def numpyro_autoregressive_var_random(op: ir.Autoregressive, numpyro_init, *numpyro_parents,
                                      is_observed):
    # numpyro.contrib.control_flow.scan exists but seems very buggy/limited
    assert isinstance(op, ir.Autoregressive)
    assert op.random

    mapped_parents, merge_args = handle_autoregressive_inputs(op, *numpyro_parents)

    class NewDist(dist.Distribution):  # NUMPYRO dist
        @property
        def support(self):
            return get_support(op.base_op)

        def __init__(self, *args, validate_args=False):
            self.args = args

            # TODO: infer correct batch_shape?
            batch_shape = ()
            parents_shapes = [p.shape for p in args]
            event_shape = op.get_shape(*parents_shapes)

            super().__init__(
                batch_shape=batch_shape,
                event_shape=event_shape,
                validate_args=validate_args,
            )

        def sample(self, key, sample_shape=()):
            assert numpyro.util.is_prng_key(key)
            # assert sample_shape == (), f"sample shape is {sample_shape} expected ()"
            assert sample_shape in ((), (1,))
            single_samp = (sample_shape == (1,))

            # print(f"{sample_shape=} {single_samp=}")

            def base_sample(carry, key_and_x):
                # print(f"{key_and_x=}")
                key = key_and_x[0]
                x = key_and_x[1:]
                # inputs = (carry,) + x
                inputs = (carry,) + merge_args(x)
                var = numpyro_var(op.base_op, *inputs, is_observed=is_observed)
                y = var.sample(key)
                return y, y

            keys = jax.random.split(key, op.length)

            # base_sample(numpyro_init, (keys[0],) + mapped_parents)

            carry, ys = jax.lax.scan(
                base_sample, numpyro_init, (keys,) + mapped_parents, length=op.length
            )
            if single_samp:
                return jnp.array([ys])
            else:
                return ys

        @dist_util.validate_sample
        def log_prob(self, value):
            def base_log_prob(carry, val_and_x):
                val = val_and_x[0]
                x = val_and_x[1:]
                # inputs = (carry,) + x
                inputs = (carry,) + merge_args(x)
                var = numpyro_var(op.base_op, *inputs, is_observed=is_observed)
                return val, var.log_prob(val)

            carry, ls = jax.lax.scan(
                base_log_prob, numpyro_init, (value,) + mapped_parents, length=op.length
            )
            return jnp.sum(ls)

    return NewDist(numpyro_init, *numpyro_parents)


# this "should" work to eliminate the need for simple_handlers. But for some reason it doesn't...
# for op_type in simple_handlers:
#     print(f"{op_type=}")
#     simple_handler = simple_handlers[op_type]
#     # create a handler that just ignores the op
#     numpyro_handlers[op_type] = lambda op, *args: simple_handler(*args)


def numpyro_var(op, *numpyro_parents, is_observed):
    # each ir op type has a unique handler
    op_type = type(op)
    if op_type in simple_handlers:
        out = simple_handlers[op_type](*numpyro_parents)
    else:
        out = numpyro_handlers[op_type](op, *numpyro_parents, is_observed=is_observed)
    return out


def generate_seed(size=()):
    import numpy as np  # import here to prevent accidental use elsewhere in a jax shop
    info = np.iinfo(np.int32)
    return np.random.randint(info.min, info.max, size=size)


def generate_key():
    seed = generate_seed()
    return jax.random.PRNGKey(seed)


def ancestor_sample_flat(vars, *, niter=None):
    model, names = get_model_flat(vars, [], [])

    def base_sample(seed):
        with numpyro.handlers.seed(rng_seed=seed):
            out = model()
            return [out[names[var]] for var in vars]

    if niter is None:
        my_seed = generate_key()
        return base_sample(my_seed)
    else:
        seeds = generate_seed(niter)
        return jax.vmap(base_sample)(seeds)


def sample_flat(vars: list[RV], given: list[RV], vals: list[RV_or_array], *, niter=10000):
    # assert isinstance(vars, Sequence)
    # assert isinstance(given, Sequence)
    # assert isinstance(vals, Sequence)
    assert len(given) == len(vals)

    # TODO: activate ancestor sampling
    if len(given) == 0:
        # print("ancestor sampling")
        return ancestor_sample_flat(vars, niter=niter)

    # TODO:z
    # raise an exception if no random vars
    # make sure vals is actually an array (not RV!)

    if any(not v.op.random for v in given):
        nonrandom_ops = [v.op for v in given if not v.op.random]
        raise ValueError(f"Cannot condition on RV with non-random op(s) {nonrandom_ops}")

    vals = [jnp.array(val) for val in vals]

    model, names = get_model_flat(vars, given, vals)

    def infer(kernel):
        mcmc = numpyro.infer.MCMC(
            kernel,
            num_warmup=niter,
            num_samples=niter,
            progress_bar=False,
        )
        key = generate_key()

        # numpyro gives some annoying future warnings
        import warnings

        with warnings.catch_warnings(action="ignore", category=FutureWarning):  # type: ignore
           mcmc.run(key)

        # numpyro_allocation_error = False
        #
        # try:
        #     mcmc.run(key)
        # except ValueError as e:
        #     if str(e).startswith("Ran out of free dims during allocation"):
        #         numpyro_allocation_error = True
        #     else:
        #         raise e
        #
        # if numpyro_allocation_error:
        #     raise ValueError("NumPyro raised a allocation error.\n"
        #                      "This usually indicates that it wasn't able to integrate out all "
        #                      "discrete latent variables.")

        # wierd trick to get samples for deterministic sites
        latent_samples = mcmc.get_samples()
        if latent_samples == {}:
            predictive = numpyro.infer.Predictive(model, num_samples=niter, infer_discrete=True)
        else:
            predictive = numpyro.infer.Predictive(model, latent_samples, infer_discrete=True)
        predictive_samples = predictive(key)
        # print(f"{predictive_samples=}")
        # merge
        samples = {**latent_samples, **predictive_samples}
        return samples

    # kernel = numpyro.infer.NUTS(model)
    # samples = infer(kernel)

    try:
        kernel = numpyro.infer.DiscreteHMCGibbs(
            numpyro.infer.NUTS(model), modified=True
        )
        # kernel = numpyro.infer.MixedHMC(
        #     numpyro.infer.HMC(model), modified=False
        # )
        samples = infer(kernel)
    except AssertionError as e:
        # print(f"FALLING BACK TO NUTS: {e}")
        if str(e) != "Cannot detect any discrete latent variables in the model.":
            raise e
        kernel = numpyro.infer.NUTS(model)
        # with warnings.simplefilter(action='ignore', category=FutureWarning):
        samples = infer(kernel)

    return [samples[names[var]] for var in vars]


# sample = inference_util.get_non_flat_sampler(sample_flat)
calc = inference_util.Calculate(sample_flat)
sample = calc.sample
E = calc.E
var = calc.var
std = calc.std
