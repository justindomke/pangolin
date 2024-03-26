from . import interface, dag, util, inference
import jax.tree_util
import functools
from jax import numpy as jnp
from numpyro import distributions as dist
from typing import Sequence
import numpyro
from jax import lax
from numpyro.distributions import util as dist_util

cond_dists = {
    interface.normal_scale: dist.Normal,
    interface.exponential: dist.Exponential,
    interface.dirichlet: dist.Dirichlet,
    interface.add: lambda a, b: a + b,
    interface.pow: lambda a, b: a**b,
    interface.exp: lambda a: jnp.exp(a),
}

cond_dist_to_support = {
    interface.normal_scale: dist.constraints.real_vector,
    interface.exponential: dist.constraints.positive,
    interface.dirichlet: dist.constraints.simplex,
}


def numpyro_dist_class(cond_dist):
    # if isinstance(cond_dist, interface.VMapDist):

    assert cond_dist in cond_dists
    return cond_dists[cond_dist]


def numpyro_dist(var, numpyro_rvs):
    par_numpyro_rvs = (numpyro_rvs[p] for p in var.parents)

    if isinstance(var.cond_dist, interface.Constant):
        return var.cond_dist.value

    if isinstance(var.cond_dist, interface.VMapDist):
        return get_numpyro_vmapdist(var.cond_dist, *par_numpyro_rvs)

    dist_class = numpyro_dist_class(var.cond_dist)

    return dist_class(*par_numpyro_rvs)


class DiagNormal(dist.Distribution):
    """
    Test case: try to implement a new distribution
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


def get_model_flat(vars, given, vals):
    assert isinstance(given, Sequence)
    assert isinstance(vals, Sequence)
    assert isinstance(vars, Sequence)

    for node in vars:
        assert isinstance(node, interface.RV)

    assert len(given) == len(vals)
    for var, val in zip(given, vals):
        assert var.shape == val.shape

    all_vars = dag.upstream_nodes(vars)

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
            print(f"{numpyro_rvs=}")
            d = numpyro_dist(var, numpyro_rvs)

            if var.cond_dist.random:
                numpyro_rv = numpyro.sample(name, d)
            else:
                numpyro_rv = numpyro.deterministic(name, d)

            numpyro_rvs[var] = numpyro_rv

    return model, names


def get_numpyro_vmapdist(cond_dist, *par_numpyro_rvs):
    assert isinstance(cond_dist, interface.VMapDist)

    print(f"{par_numpyro_rvs=}")

    # # TODO: START HERE
    # if not cond_dist.random:
    #     if cond_dist in cond_dists:
    #         return cond_dists[cond_dist]
    #     else:
    #         assert isinstance(cond_dist, interface.VMapDist)

    class NewDist(dist.Distribution):
        # TODO: fix? somehow?
        # support = dist.constraints.real_vector

        @property
        def support(self):
            my_cond_dist = cond_dist
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
                # TODO: update to do more general lookup
                # dist_class = cond_dists[cond_dist.base_cond_dist]
                dist_class = numpyro_dist_class(cond_dist.base_cond_dist)
                dist = dist_class(*args)

                return dist.sample(key)

            keys = jax.random.split(key, self.event_shape[0])
            in_axes = (0,) + cond_dist.in_axes
            axis_size = cond_dist.axis_size
            args = (keys,) + self.args
            return jax.vmap(base_sample, in_axes, axis_size=axis_size)(*args)

        @dist_util.validate_sample
        def log_prob(self, value):
            def base_log_prob(val, *args):
                # TODO: update to do more general lookup
                # dist_class = cond_dists[cond_dist.base_cond_dist]
                dist_class = numpyro_dist_class(cond_dist.base_cond_dist)
                dist = dist_class(*args)
                return dist.log_prob(val)

            in_axes = (0,) + cond_dist.in_axes
            axis_size = cond_dist.axis_size
            args = (value,) + self.args
            ls = jax.vmap(base_log_prob, in_axes, axis_size=axis_size)(*args)
            return jnp.sum(ls)

    return NewDist(*par_numpyro_rvs)


def sample_flat(vars, given, vals):
    model, names = get_model_flat(vars, given, vals)
    nuts_kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=10000, num_samples=10000)
    key = jax.random.PRNGKey(0)
    mcmc.run(key)
    mcmc.print_summary(exclude_deterministic=False)

    # wierd trick to get samples for deterministic sites
    latent_samples = mcmc.get_samples()
    predictive = numpyro.infer.Predictive(model, latent_samples)
    predictive_samples = predictive(key)

    # merge
    samples = {**latent_samples, **predictive_samples}

    return [samples[names[var]] for var in vars]
