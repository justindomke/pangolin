from . import interface, dag, util, inference
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
import numpy as np

cond_dist_to_numpyro_dist = {
    interface.normal_scale: dist.Normal,
    interface.bernoulli: dist.Bernoulli,
    interface.bernoulli_logit: dist.BernoulliLogits,
    interface.beta: dist.Beta,
    interface.beta_binomial: lambda n, a, b: dist.BetaBinomial(a, b, n),
    interface.binomial: dist.Binomial,
    interface.categorical: dist.Categorical,
    interface.cauchy: dist.Cauchy,
    interface.exponential: dist.Exponential,
    interface.dirichlet: dist.Dirichlet,
    interface.gamma: dist.Gamma,
    interface.multinomial: dist.Multinomial,
    interface.multi_normal_cov: dist.MultivariateNormal,
    interface.student_t: dist.StudentT,
    interface.uniform: dist.Uniform,
    interface.add: lambda a, b: a + b,
    interface.sub: lambda a, b: a - b,
    interface.mul: lambda a, b: a * b,
    interface.div: lambda a, b: a / b,
    interface.pow: lambda a, b: a**b,
    interface.abs: jnp.abs,
    interface.arccos: jnp.arccos,
    interface.arccosh: jnp.arccosh,
    interface.arcsin: jnp.arcsin,
    interface.arcsinh: jnp.arcsinh,
    interface.arctan: jnp.arctan,
    interface.arctanh: jnp.arctanh,
    interface.cos: jnp.cos,
    interface.cosh: jnp.cosh,
    interface.exp: jnp.exp,
    interface.inv_logit: dist.transforms.SigmoidTransform(),
    interface.log: jnp.log,
    interface.loggamma: jspecial.gammaln,
    interface.logit: jspecial.logit,
    interface.sin: jnp.sin,
    interface.sinh: jnp.sinh,
    interface.step: lambda x: jnp.heaviside(x, 0.5),
    interface.matmul: jnp.matmul,
    interface.inv: jnp.linalg.inv,
    interface.softmax: jnn.softmax,
}


def numpyro_var(cond_dist, *numpyro_parents):
    """given a Pangolin cond_dist and a Numpyro parents, get new numpyro dist"""

    numpyro_parents = [
        p if isinstance(p, dist.Distribution) else jnp.array(p) for p in numpyro_parents
    ]

    if cond_dist in cond_dist_to_numpyro_dist:
        d = cond_dist_to_numpyro_dist[cond_dist]
        return d(*numpyro_parents)
    elif isinstance(cond_dist, interface.Constant):  # Constants
        return cond_dist.value
    elif isinstance(cond_dist, interface.Sum):  # Sums
        [a] = numpyro_parents
        return jnp.sum(a, axis=cond_dist.axis)
    elif isinstance(cond_dist, interface.Index):  # Indexes
        return numpyro_index_var(cond_dist, *numpyro_parents)
    elif isinstance(cond_dist, interface.VMapDist):  # VMaps
        return numpyro_vmap_var(cond_dist, *numpyro_parents)
    elif isinstance(cond_dist, interface.Mixture):  # Mixtures
        return numpyro_mixture_var(cond_dist, *numpyro_parents)
    elif isinstance(cond_dist, interface.Truncated):  # Truncated dists
        return numpyro_truncated_var(cond_dist, *numpyro_parents)
    elif isinstance(cond_dist, interface.Composite):
        return numpyro_composite_var(cond_dist, *numpyro_parents)
    elif isinstance(cond_dist, interface.Autoregressive):
        return numpyro_autoregressive_var(cond_dist, *numpyro_parents)
    else:
        raise NotImplementedError(f"unsupported cond_dist {cond_dist} {type(cond_dist)}")


def numpyro_index_var(cond_dist: interface.Index, val, *indices):
    stuff = []
    i = 0
    for my_slice in cond_dist.slices:
        if my_slice:
            stuff.append(my_slice)
        else:
            stuff.append(indices[i])
            i += 1
    stuff = tuple(stuff)
    return val[stuff]


def numpyro_truncated_var(cond_dist: interface.Truncated, *numpyro_parents):
    return dist.TruncatedDistribution(
        numpyro_var(cond_dist.base_dist, *numpyro_parents),
        low=cond_dist.lo,
        high=cond_dist.hi,
    )

def numpyro_composite_var(cond_dist: interface.Composite, *numpyro_parents):
    vals = list(numpyro_parents)
    assert len(numpyro_parents) == cond_dist.num_inputs
    for my_cond_dist, my_par_nums in zip(cond_dist.cond_dists, cond_dist.par_nums):
        my_parents = [vals[i] for i in my_par_nums]
        new_val = numpyro_var(my_cond_dist, *my_parents)
        vals.append(new_val)
    return vals[-1]

def numpyro_autoregressive_var(cond_dist, *numpyro_parents):
    if cond_dist.random:
        return numpyro_autoregressive_var_random(cond_dist, *numpyro_parents)
    else:
        return numpyro_autoregressive_var_nonrandom(cond_dist, *numpyro_parents)


def numpyro_autoregressive_var_nonrandom(cond_dist: interface.Autoregressive, numpyro_init, *numpyro_parents):
    # numpyro.contrib.control_flow.scan exists but seems very buggy/limited
    assert isinstance(cond_dist, interface.Autoregressive)
    assert not cond_dist.random
    def myfun(carry, x):
        inputs = (carry,) + x
        y = numpyro_var(cond_dist.base_cond_dist, *inputs)
        return y, y
    carry, ys = jax.lax.scan(myfun, numpyro_init, numpyro_parents, length=cond_dist.length)
    return ys


def numpyro_autoregressive_var_random(cond_dist: interface.Autoregressive, numpyro_init, *numpyro_parents):
    # numpyro.contrib.control_flow.scan exists but seems very buggy/limited
    assert isinstance(cond_dist, interface.Autoregressive)
    assert cond_dist.random

    class NewDist(dist.Distribution):
        @property
        def support(self):
            return get_support(cond_dist.base_cond_dist)

        def __init__(self, *args, validate_args=False):
            self.args = args

            # TODO: infer correct batch_shape?
            batch_shape = ()
            parents_shapes = [p.shape for p in args]
            event_shape = cond_dist.get_shape(*parents_shapes)

            super().__init__(
                batch_shape=batch_shape,
                event_shape=event_shape,
                validate_args=validate_args,
            )

        def sample(self, key, sample_shape=()):
            assert numpyro.util.is_prng_key(key)
            assert sample_shape == ()

            def base_sample(carry, key_and_x):
                key = key_and_x[0]
                x = key_and_x[1:]
                inputs = (carry,) + x
                var = numpyro_var(cond_dist.base_cond_dist, *inputs)
                y = var.sample(key)
                return y, y

            keys = jax.random.split(key,cond_dist.length)
            carry, ys = jax.lax.scan(base_sample, numpyro_init, (keys,)+numpyro_parents, length=cond_dist.length)
            return ys

        @dist_util.validate_sample
        def log_prob(self, value):
            def base_log_prob(carry, val_and_x):
                val = val_and_x[0]
                x = val_and_x[1:]
                inputs = (carry,) + x
                var = numpyro_var(cond_dist.base_cond_dist, *inputs)
                return val, var.log_prob(val)

            carry, ls = jax.lax.scan(base_log_prob, numpyro_init, (value,) + numpyro_parents, length=cond_dist.length)
            return jnp.sum(ls)

    return NewDist(numpyro_init, *numpyro_parents)


def numpyro_vmap_var(cond_dist, *numpyro_parents):
    if cond_dist.random:
        return numpyro_vmap_var_random(cond_dist, *numpyro_parents)
    else:
        return numpyro_vmap_var_nonrandom(cond_dist, *numpyro_parents)


cond_dist_to_support = {
    interface.normal_scale: dist.constraints.real_vector,
    interface.exponential: dist.constraints.positive,
    interface.dirichlet: dist.constraints.simplex,
}


def get_support(cond_dist):
    """
    Get support. Only used inside by numpyro_vmap_var_random

    """
    if cond_dist in cond_dist_to_support:
        return cond_dist_to_support[cond_dist]
    elif isinstance(cond_dist, interface.Truncated):
        if cond_dist.lo is not None and cond_dist.hi is not None:
            return dist.constraints.interval(cond_dist.lo, cond_dist.hi)
        elif cond_dist.lo is not None:
            assert cond_dist.hi is None
            return dist.constraints.greater_than(cond_dist.lo)
        elif cond_dist.hi is not None:
            assert cond_dist.lo is None
            return dist.constraints.less_than(cond_dist.hi)
        else:
            assert False, "should be impossible"


def numpyro_vmap_var_random(cond_dist, *numpyro_parents):
    assert isinstance(cond_dist, interface.VMapDist)
    assert cond_dist.random

    class NewDist(dist.Distribution):
        @property
        def support(self):
            my_cond_dist = cond_dist
            while isinstance(my_cond_dist, interface.VMapDist):
                my_cond_dist = my_cond_dist.base_cond_dist
            return get_support(my_cond_dist)

        def __init__(self, *args, validate_args=False):
            self.args = args

            # TODO: infer correct batch_shape?
            batch_shape = ()
            parents_shapes = [p.shape for p in args]
            event_shape = cond_dist.get_shape(*parents_shapes)

            super().__init__(
                batch_shape=batch_shape,
                event_shape=event_shape,
                validate_args=validate_args,
            )

        def sample(self, key, sample_shape=()):
            assert numpyro.util.is_prng_key(key)
            assert sample_shape == ()

            def base_sample(key, *args):
                var = numpyro_var(cond_dist.base_cond_dist, *args)
                return var.sample(key)

            keys = jax.random.split(key, self.event_shape[0])
            in_axes = (0,) + cond_dist.in_axes
            axis_size = cond_dist.axis_size
            args = (keys,) + self.args
            return jax.vmap(base_sample, in_axes, axis_size=axis_size)(*args)

        @dist_util.validate_sample
        def log_prob(self, value):
            def base_log_prob(val, *args):
                var = numpyro_var(cond_dist.base_cond_dist, *args)
                return var.log_prob(val)

            in_axes = (0,) + cond_dist.in_axes
            axis_size = cond_dist.axis_size
            args = (value,) + self.args
            ls = jax.vmap(base_log_prob, in_axes, axis_size=axis_size)(*args)
            return jnp.sum(ls)

    return NewDist(*numpyro_parents)


def numpyro_vmap_var_nonrandom(cond_dist, *numpyro_parents):
    assert isinstance(cond_dist, interface.VMapDist)
    assert not cond_dist.random

    def base_var(*args):
        return numpyro_var(cond_dist.base_cond_dist, *args)

    in_axes = cond_dist.in_axes
    axis_size = cond_dist.axis_size
    args = numpyro_parents
    return jax.vmap(base_var, in_axes=in_axes, axis_size=axis_size)(*args)


def numpyro_mixture_var(cond_dist, *numpyro_parents):
    assert isinstance(cond_dist, interface.Mixture)

    class NewDist(dist.Distribution):
        @property
        def support(self):
            my_cond_dist = cond_dist.vmap_dist.base_dist
            while isinstance(my_cond_dist, interface.VMapDist):
                my_cond_dist = my_cond_dist.base_cond_dist
            return cond_dist_to_support[my_cond_dist]

        def __init__(self, *args, validate_args=False):
            self.args = args

            # TODO: infer correct batch_shape?
            batch_shape = ()
            parents_shapes = [p.shape for p in args]
            event_shape = cond_dist.get_shape(*parents_shapes)

            super().__init__(
                batch_shape=batch_shape,
                event_shape=event_shape,
                validate_args=validate_args,
            )

        def sample(self, key, sample_shape=()):
            assert numpyro.util.is_prng_key(key)
            assert sample_shape == ()

            def base_sample(key, *args):
                var = numpyro_var(cond_dist.vmap_dist.base_cond_dist, *args)
                return var.sample(key)

            mixing_args = self.args[: cond_dist.num_mixing_args]
            vmap_args = self.args[cond_dist.num_mixing_args :]

            vmap_shape = cond_dist.vmap_dist.get_shape(*(i.shape for i in vmap_args))

            # seems silly to reproduce functionality from vmapdist...
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, vmap_shape[0])
            in_axes = (0,) + cond_dist.vmap_dist.in_axes
            axis_size = cond_dist.vmap_dist.axis_size
            args = (keys,) + vmap_args
            vec_sample = jax.vmap(base_sample, in_axes, axis_size=axis_size)(*args)

            mix_var = numpyro_var(cond_dist.mixing_dist, *mixing_args)
            mix_sample = mix_var.sample(key)
            return vec_sample[mix_sample]

        @dist_util.validate_sample
        def log_prob(self, value):
            def base_log_prob(val, *args):
                var = numpyro_var(cond_dist.vmap_dist.base_cond_dist, *args)
                return var.log_prob(val)

            mixing_args = self.args[: cond_dist.num_mixing_args]
            vmap_args = self.args[cond_dist.num_mixing_args :]

            vmap_shape = cond_dist.vmap_dist.get_shape(*(i.shape for i in vmap_args))

            in_axes = (None,) + cond_dist.vmap_dist.in_axes
            axis_size = cond_dist.vmap_dist.axis_size
            args = (value,) + vmap_args
            vec_ls = jax.vmap(base_log_prob, in_axes, axis_size=axis_size)(*args)

            def weight_log_prob(val, *args):
                var = numpyro_var(cond_dist.mixing_dist, *args)
                return var.log_prob(val)

            idx = jnp.arange(vmap_shape[0])
            log_weights = jax.vmap(weight_log_prob, [0] + [None] * len(mixing_args))(
                idx, *mixing_args
            )

            return jax.scipy.special.logsumexp(vec_ls + log_weights)

    return NewDist(*numpyro_parents)


def get_model_flat(vars, given, vals):
    assert isinstance(given, Sequence)
    assert isinstance(vals, Sequence)
    assert isinstance(vars, Sequence)

    for node in vars:
        assert isinstance(node, interface.RV)

    assert len(given) == len(vals)
    for var, val in zip(given, vals):
        assert var.shape == val.shape

    all_vars = dag.upstream_nodes(vars + given)

    # all_vars = inference.upstream_with_descendent(vars, given)
    # latent_vars = [node for node in random_vars if node not in given]

    names = {}
    varnum = 0
    for var in all_vars:
        name = f"v{varnum}"
        varnum += 1
        names[var] = name

    def model():
        numpyro_rvs = {}
        for var in all_vars:
            name = names[var]
            # print(f"{numpyro_rvs=}")
            numpyro_pars = [numpyro_rvs[p] for p in var.parents]
            d = numpyro_var(var.cond_dist, *numpyro_pars)

            if var in given:
                obs = vals[given.index(var)]
            else:
                obs = None

            # print(f"{var=} {obs=}")

            if var.cond_dist.random:
                numpyro_rv = numpyro.sample(name, d, obs=obs)
            else:
                numpyro_rv = numpyro.deterministic(name, d)

            numpyro_rvs[var] = numpyro_rv
        return numpyro_rvs

    return model, names


def generate_seed(size=()):
    info = np.iinfo(int)
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
            return [out[var] for var in vars]

    if niter is None:
        my_seed = generate_key()
        return base_sample(my_seed)
    else:
        # seeds = jax.random.split(seed, niter)
        seeds = generate_seed(niter)
        return jax.vmap(base_sample)(seeds)


def sample_flat(vars, given, vals, *, niter=10000):
    assert isinstance(vars, Sequence)
    assert isinstance(given, Sequence)
    assert isinstance(vals, Sequence)
    assert len(given) == len(vals)

    if len(given) == 0:
        return ancestor_sample_flat(vars, niter=niter)

    vals = [jnp.array(val) for val in vals]

    model, names = get_model_flat(vars, given, vals)
    nuts_kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(
        nuts_kernel,
        num_warmup=niter,
        num_samples=niter,
        progress_bar=False,
    )
    key = generate_key()
    mcmc.run(key)
    # mcmc.print_summary(exclude_deterministic=False)

    # wierd trick to get samples for deterministic sites
    latent_samples = mcmc.get_samples()
    predictive = numpyro.infer.Predictive(model, latent_samples)
    predictive_samples = predictive(key)

    # merge
    samples = {**latent_samples, **predictive_samples}

    return [samples[names[var]] for var in vars]


class DiagNormal(dist.Distribution):
    """
    Test case: try to implement a new distribution. This isn't useful, just here to try to understand numpyro better.
    """

    # arg_constraints = {
    #     "loc": dist.constraints.real_vector,
    #     "scale": dist.constraints.real_vector,
    # }

    support = dist.constraints.real_vector

    # reparametrized_params = ["loc", "scale"]

    def __init__(self, loc, scale, *, validate_args=False):
        assert jnp.ndim(loc) == 1
        assert jnp.ndim(scale) == 1

        # self.loc, self.scale = dist_util.promote_shapes(loc, scale)

        self.loc = loc
        self.scale = scale

        print(f"{self.loc=}")
        print(f"{self.scale=}")

        batch_shape = ()
        event_shape = jnp.shape(loc)

        super().__init__(
            batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert numpyro.util.is_prng_key(key)
        eps = jax.random.normal(
            key, shape=sample_shape + self.batch_shape + self.event_shape
        )
        return self.loc + eps * self.scale

    @dist_util.validate_sample
    def log_prob(self, value):
        normalize_term = jnp.log(jnp.sqrt(2 * jnp.pi) * self.scale)
        value_scaled = (value - self.loc) / self.scale
        return jnp.sum(-0.5 * value_scaled**2 - normalize_term)
