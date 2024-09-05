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
#import numpy as np
from pangolin.ir.rv import RV
from pangolin import dag, util, ir
from pangolin.interface.interface import OperatorRV
from numpy.typing import ArrayLike
from pangolin.interface import RV_or_array
from pangolin.inference import inference_util


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
        raise ValueError("length of given does not match length of vals")

    vals = [jnp.array(a) for a in vals]

    for var, val in zip(given, vals):
        if var.shape != val.shape:
            raise ValueError("given var has shape not matching given val")
        if not util.is_numeric_numpy_array(val):
            raise ValueError("given val not numeric")

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

            d = numpyro_var(var.op, *numpyro_pars)

            if var.op.random:
                assert isinstance(d, dist.Distribution), "numpyo handler failed to return distribution for random op"
                if var in given:
                    numpyro_rv = numpyro.sample(name, d, obs=vals[given.index(var)])
                else:
                    numpyro_rv = numpyro.sample(name, d)
            else:
                assert isinstance(d, jnp.ndarray), f"numpyo handler failed to return jax.numpy array for nonrandom op {var=} {d=}"
                numpyro_rv = numpyro.deterministic(name, d)

            var_to_numpyro_rv[var] = numpyro_rv
            name_to_numpyro_rv[name] = numpyro_rv
        return name_to_numpyro_rv

    return model, var_to_name


# handlers that don't need to look at the op itself, just the type
simple_handlers = {
    ir.Normal: dist.Normal,
    ir.NormalPrec: lambda loc, prec: dist.Normal(loc, 1 / prec**2),
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
    ir.Pow: lambda a, b: a**b,
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
def handle_constant(op: ir.Constant):
    # return op.value
    return jnp.array(op.value) # return a jax array, not a numpy array


@register_handler(ir.Index)
def handle_index(op: ir.Index, val, *indices):
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
def handle_sum(op: ir.Sum, val):
    return jnp.sum(val, axis=op.axis)


@register_handler(ir.VMap)
def handle_vmap(op: ir.VMap, *numpyro_parents):
    if op.random:
        return numpyro_vmap_var_random(op, *numpyro_parents)
    else:
        return numpyro_vmap_var_nonrandom(op, *numpyro_parents)


op_class_to_support = util.WriteOnceDefaultDict(
    default_factory=lambda key: dist.constraints.real_vector
)
op_class_to_support[ir.Exponential] = dist.constraints.positive
op_class_to_support[ir.Dirichlet] = dist.constraints.simplex


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


def numpyro_vmap_var_random(op: ir.VMap, *numpyro_parents):
    assert isinstance(op, ir.VMap)
    assert op.random

    class NewDist(dist.Distribution):
        @property
        def support(self):
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
                var = numpyro_var(op.base_op, *args)
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
                var = numpyro_var(op.base_op, *args)
                return var.log_prob(val)

            in_axes = (0,) + op.in_axes
            axis_size = op.axis_size
            args = (value,) + self.args
            ls = jax.vmap(base_log_prob, in_axes, axis_size=axis_size)(*args)
            return jnp.sum(ls)

    return NewDist(*numpyro_parents)


def numpyro_vmap_var_nonrandom(op: ir.VMap, *numpyro_parents):
    assert isinstance(op, ir.VMap)
    assert not op.random

    def base_var(*args):
        return numpyro_var(op.base_op, *args)

    in_axes = op.in_axes
    axis_size = op.axis_size
    args = numpyro_parents
    return jax.vmap(base_var, in_axes=in_axes, axis_size=axis_size)(*args)


@register_handler(ir.Composite)
def numpyro_composite_var(op: ir.Composite, *numpyro_parents):
    vals = list(numpyro_parents)
    assert len(numpyro_parents) == op.num_inputs
    for my_cond_dist, my_par_nums in zip(op.ops, op.par_nums, strict=True):
        my_parents = [vals[i] for i in my_par_nums]
        new_val = numpyro_var(my_cond_dist, *my_parents)
        vals.append(new_val)
    return vals[-1]


@register_handler(ir.Autoregressive)
def numpyro_autoregressive_var(op, *numpyro_parents):
    if op.random:
        return numpyro_autoregressive_var_random(op, *numpyro_parents)
    else:
        return numpyro_autoregressive_var_nonrandom(op, *numpyro_parents)


def handle_autoregressive_inputs(op: ir.Autoregressive, *numpyro_parents):
    for in_axis in op.in_axes:
        assert in_axis in [0,None], "NumPyro only supports Autoregressive with in_axis of 0 or None"

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

def numpyro_autoregressive_var_nonrandom(op: ir.Autoregressive, numpyro_init, *numpyro_parents):
    # numpyro.contrib.control_flow.scan exists but seems very buggy/limited
    assert isinstance(op, ir.Autoregressive)
    assert not op.random

    mapped_parents, merge_args = handle_autoregressive_inputs(op, *numpyro_parents)
    #print(f"{mapped_parents=}")
    #print(f"{numpyro_parents=}")
    #print(f"{merge_args(mapped_parents)=}")
    assert merge_args(mapped_parents) == numpyro_parents

    print(f"{numpyro_init=}")

    def myfun(carry, x):
        #inputs = (carry,) + x
        inputs = (carry,) + merge_args(x)
        #print(f"{inputs=}")
        #print(f"{[a.shape for a in inputs]=}")
        y = numpyro_var(op.base_op, *inputs)
        return y, y

    #myfun(numpyro_init, tuple(p[0] for p in mapped_parents))

    #carry, ys = jax.lax.scan(myfun, numpyro_init, numpyro_parents, length=op.length)
    carry, ys = jax.lax.scan(myfun, numpyro_init, mapped_parents, length=op.length)
    return ys


def numpyro_autoregressive_var_random(op: ir.Autoregressive, numpyro_init, *numpyro_parents):
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
            #assert sample_shape == (), f"sample shape is {sample_shape} expected ()"
            assert sample_shape in ((),(1,))
            single_samp = (sample_shape == (1,))

            #print(f"{sample_shape=} {single_samp=}")

            def base_sample(carry, key_and_x):
                #print(f"{key_and_x=}")
                key = key_and_x[0]
                x = key_and_x[1:]
                #inputs = (carry,) + x
                inputs = (carry,) + merge_args(x)
                var = numpyro_var(op.base_op, *inputs)
                y = var.sample(key)
                return y, y


            keys = jax.random.split(key, op.length)

            #base_sample(numpyro_init, (keys[0],) + mapped_parents)

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
                #inputs = (carry,) + x
                inputs = (carry,) + merge_args(x)
                var = numpyro_var(op.base_op, *inputs)
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


def numpyro_var(op, *numpyro_parents):
    # each ir op type has a unique handler
    op_type = type(op)
    if op_type in simple_handlers:
        out = simple_handlers[op_type](*numpyro_parents)
    else:
        out = numpyro_handlers[op_type](op, *numpyro_parents)
    # print(f"{op=}")
    # print(f"{[type(p) for p in numpyro_parents]=}")
    # print(f"{type(out)=}")
    return out

def generate_seed(size=()):
    import numpy as np # import here to prevent accidental use elsewhere in a jax shop
    info = np.iinfo(np.int32)
    return np.random.randint(info.min, info.max, size=size)


def generate_key():
    seed = generate_seed()
    # print(f"{seed=}")
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
        # seeds = jax.random.split(seed, niter)
        seeds = generate_seed(niter)
        return jax.vmap(base_sample)(seeds)


def sample_flat(vars: list[RV], given: list[RV], vals: list[RV_or_array], *, niter=10000):
    # assert isinstance(vars, Sequence)
    # assert isinstance(given, Sequence)
    # assert isinstance(vals, Sequence)
    assert len(given) == len(vals)

    # TODO: activate ancestor sampling
    if len(given) == 0:
        #print("ancestor sampling")
        return ancestor_sample_flat(vars, niter=niter)

    # TODO:
    # raise an exception if no random vars

    if any(not v.op.random for v in given):
       raise ValueError("Cannot condition on RV with non-random op")

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

        # wierd trick to get samples for deterministic sites
        latent_samples = mcmc.get_samples()
        #print(f"{latent_samples=}")
        predictive = numpyro.infer.Predictive(model, latent_samples)
        predictive_samples = predictive(key)
        #print(f"{predictive_samples=}")
        # merge
        samples = {**latent_samples, **predictive_samples}
        return samples

    try:
        kernel = numpyro.infer.DiscreteHMCGibbs(
            numpyro.infer.NUTS(model), modified=True
        )
        samples = infer(kernel)
    except AssertionError as e:
        if str(e) != "Cannot detect any discrete latent variables in the model.":
            raise e
        kernel = numpyro.infer.NUTS(model)
        # with warnings.simplefilter(action='ignore', category=FutureWarning):
        samples = infer(kernel)

    # try:
    #     kernel = numpyro.infer.NUTS(model)
    #     # with warnings.simplefilter(action='ignore', category=FutureWarning):
    #     samples = infer(kernel)
    # except ValueError as e:
    #     print("sampling failed with default—switching to MixedHMC")
    #     assert str(e) == "No sample sites in posterior samples to infer `num_samples`."
    #     # kernel = numpyro.infer.MixedHMC(
    #     #     numpyro.infer.HMC(model, trajectory_length=1.2), num_discrete_updates=20
    #     # )
    #     kernel = numpyro.infer.DiscreteHMCGibbs(
    #         numpyro.infer.NUTS(model), modified=True
    #     )
    #     samples = infer(kernel)

    return [samples[names[var]] for var in vars]


#sample = inference_util.get_non_flat_sampler(sample_flat)
calc = inference_util.Calculate(sample_flat)
sample = calc.sample
E = calc.E
var = calc.var
std = calc.std


# def numpyro_var(cond_dist, *numpyro_parents):
#     """given a Pangolin cond_dist and a Numpyro parents, get new numpyro dist"""
#
#     numpyro_parents = [
#         p if isinstance(p, dist.Distribution) else jnp.array(p) for p in numpyro_parents
#     ]
#
#     if cond_dist in cond_dist_to_numpyro_dist:
#         d = cond_dist_to_numpyro_dist[cond_dist]
#         return d(*numpyro_parents)
#     elif isinstance(cond_dist, interface.Constant):  # Constants
#         return cond_dist.value
#     elif isinstance(cond_dist, interface.Sum):  # Sums
#         [a] = numpyro_parents
#         return jnp.sum(a, axis=cond_dist.axis)
#     elif isinstance(cond_dist, interface.Index):  # Indexes
#         return numpyro_index_var(cond_dist, *numpyro_parents)
#     elif isinstance(cond_dist, interface.VMapDist):  # VMaps
#         return numpyro_vmap_var(cond_dist, *numpyro_parents)
#     elif isinstance(cond_dist, interface.Mixture):  # Mixtures
#         return numpyro_mixture_var(cond_dist, *numpyro_parents)
#     elif isinstance(cond_dist, interface.Truncated):  # Truncated dists
#         return numpyro_truncated_var(cond_dist, *numpyro_parents)
#     elif isinstance(cond_dist, interface.Composite):
#         return numpyro_composite_var(cond_dist, *numpyro_parents)
#     elif isinstance(cond_dist, interface.Autoregressive):
#         return numpyro_autoregressive_var(cond_dist, *numpyro_parents)
#     else:
#         raise NotImplementedError(f"unsupported cond_dist {cond_dist} {type(cond_dist)}")
#
#
# def numpyro_truncated_var(cond_dist: interface.Truncated, *numpyro_parents):
#     return dist.TruncatedDistribution(
#         numpyro_var(cond_dist.base_dist, *numpyro_parents),
#         low=cond_dist.lo,
#         high=cond_dist.hi,
#     )
#
# def numpyro_composite_var(cond_dist: interface.Composite, *numpyro_parents):
#     vals = list(numpyro_parents)
#     assert len(numpyro_parents) == cond_dist.num_inputs
#     for my_cond_dist, my_par_nums in zip(cond_dist.cond_dists, cond_dist.par_nums):
#         my_parents = [vals[i] for i in my_par_nums]
#         new_val = numpyro_var(my_cond_dist, *my_parents)
#         vals.append(new_val)
#     return vals[-1]
#
#
#
# def numpyro_mixture_var(cond_dist, *numpyro_parents):
#     assert isinstance(cond_dist, interface.Mixture)
#
#     class NewDist(dist.Distribution):
#         @property
#         def support(self):
#             my_cond_dist = cond_dist.vmap_dist.base_dist
#             while isinstance(my_cond_dist, interface.VMapDist):
#                 my_cond_dist = my_cond_dist.base_cond_dist
#             return cond_dist_to_support[my_cond_dist]
#
#         def __init__(self, *args, validate_args=False):
#             self.args = args
#
#             # TODO: infer correct batch_shape?
#             batch_shape = ()
#             parents_shapes = [p.shape for p in args]
#             event_shape = cond_dist.get_shape(*parents_shapes)
#
#             super().__init__(
#                 batch_shape=batch_shape,
#                 event_shape=event_shape,
#                 validate_args=validate_args,
#             )
#
#         def sample(self, key, sample_shape=()):
#             assert numpyro.util.is_prng_key(key)
#             assert sample_shape == ()
#
#             def base_sample(key, *args):
#                 var = numpyro_var(cond_dist.vmap_dist.base_cond_dist, *args)
#                 return var.sample(key)
#
#             mixing_args = self.args[: cond_dist.num_mixing_args]
#             vmap_args = self.args[cond_dist.num_mixing_args :]
#
#             vmap_shape = cond_dist.vmap_dist.get_shape(*(i.shape for i in vmap_args))
#
#             # seems silly to reproduce functionality from vmapdist...
#             key, subkey = jax.random.split(key)
#             keys = jax.random.split(subkey, vmap_shape[0])
#             in_axes = (0,) + cond_dist.vmap_dist.in_axes
#             axis_size = cond_dist.vmap_dist.axis_size
#             args = (keys,) + vmap_args
#             vec_sample = jax.vmap(base_sample, in_axes, axis_size=axis_size)(*args)
#
#             mix_var = numpyro_var(cond_dist.mixing_dist, *mixing_args)
#             mix_sample = mix_var.sample(key)
#             return vec_sample[mix_sample]
#
#         @dist_util.validate_sample
#         def log_prob(self, value):
#             def base_log_prob(val, *args):
#                 var = numpyro_var(cond_dist.vmap_dist.base_cond_dist, *args)
#                 return var.log_prob(val)
#
#             mixing_args = self.args[: cond_dist.num_mixing_args]
#             vmap_args = self.args[cond_dist.num_mixing_args :]
#
#             vmap_shape = cond_dist.vmap_dist.get_shape(*(i.shape for i in vmap_args))
#
#             in_axes = (None,) + cond_dist.vmap_dist.in_axes
#             axis_size = cond_dist.vmap_dist.axis_size
#             args = (value,) + vmap_args
#             vec_ls = jax.vmap(base_log_prob, in_axes, axis_size=axis_size)(*args)
#
#             def weight_log_prob(val, *args):
#                 var = numpyro_var(cond_dist.mixing_dist, *args)
#                 return var.log_prob(val)
#
#             idx = jnp.arange(vmap_shape[0])
#             log_weights = jax.vmap(weight_log_prob, [0] + [None] * len(mixing_args))(
#                 idx, *mixing_args
#             )
#
#             return jax.scipy.special.logsumexp(vec_ls + log_weights)
#
#     return NewDist(*numpyro_parents)
#
#
#
#
# def generate_seed(size=()):
#     info = np.iinfo(int)
#     return np.random.randint(info.min, info.max, size=size)
#
#
# def generate_key():
#     seed = generate_seed()
#     # print(f"{seed=}")
#     return jax.random.PRNGKey(seed)
#
#
# def ancestor_sample_flat(vars, *, niter=None):
#     model, names = get_model_flat(vars, [], [])
#
#     def base_sample(seed):
#         with numpyro.handlers.seed(rng_seed=seed):
#             out = model()
#             return [out[var] for var in vars]
#
#     if niter is None:
#         my_seed = generate_key()
#         return base_sample(my_seed)
#     else:
#         # seeds = jax.random.split(seed, niter)
#         seeds = generate_seed(niter)
#         return jax.vmap(base_sample)(seeds)
#
#
# def sample_flat(vars, given, vals, *, niter=10000):
#     assert isinstance(vars, Sequence)
#     assert isinstance(given, Sequence)
#     assert isinstance(vals, Sequence)
#     assert len(given) == len(vals)
#
#     if len(given) == 0:
#         return ancestor_sample_flat(vars, niter=niter)
#
#     vals = [jnp.array(val) for val in vals]
#
#     model, names = get_model_flat(vars, given, vals)
#     nuts_kernel = numpyro.infer.NUTS(model)
#     mcmc = numpyro.infer.MCMC(
#         nuts_kernel,
#         num_warmup=niter,
#         num_samples=niter,
#         progress_bar=False,
#     )
#     key = generate_key()
#     mcmc.run(key)
#     # mcmc.print_summary(exclude_deterministic=False)
#
#     # wierd trick to get samples for deterministic sites
#     latent_samples = mcmc.get_samples()
#     predictive = numpyro.infer.Predictive(model, latent_samples)
#     predictive_samples = predictive(key)
#
#     # merge
#     samples = {**latent_samples, **predictive_samples}
#
#     return [samples[names[var]] for var in vars]
#
#
# class DiagNormal(dist.Distribution):
#     """
#     Test case: try to implement a new distribution. This isn't useful, just here to try to understand numpyro better.
#     """
#
#     # arg_constraints = {
#     #     "loc": dist.constraints.real_vector,
#     #     "scale": dist.constraints.real_vector,
#     # }
#
#     support = dist.constraints.real_vector
#
#     # reparametrized_params = ["loc", "scale"]
#
#     def __init__(self, loc, scale, *, validate_args=False):
#         assert jnp.ndim(loc) == 1
#         assert jnp.ndim(scale) == 1
#
#         # self.loc, self.scale = dist_util.promote_shapes(loc, scale)
#
#         self.loc = loc
#         self.scale = scale
#
#         print(f"{self.loc=}")
#         print(f"{self.scale=}")
#
#         batch_shape = ()
#         event_shape = jnp.shape(loc)
#
#         super().__init__(
#             batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args
#         )
#
#     def sample(self, key, sample_shape=()):
#         assert numpyro.util.is_prng_key(key)
#         eps = jax.random.normal(
#             key, shape=sample_shape + self.batch_shape + self.event_shape
#         )
#         return self.loc + eps * self.scale
#
#     @dist_util.validate_sample
#     def log_prob(self, value):
#         normalize_term = jnp.log(jnp.sqrt(2 * jnp.pi) * self.scale)
#         value_scaled = (value - self.loc) / self.scale
#         return jnp.sum(-0.5 * value_scaled**2 - normalize_term)
