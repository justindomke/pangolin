from . import interface, dag, util, inference
import jax.tree_util
import functools
from jax import numpy as jnp
from numpyro import distributions as dist
from typing import Sequence
import numpyro
from jax import lax
from numpyro.distributions import util as dist_util


cond_dists = {interface.normal_scale: dist.Normal}


def get_dist(var, numpyro_rvs):
    assert var.cond_dist in cond_dists
    dist_class = cond_dists[var.cond_dist]
    par_numpyro_rvs = (numpyro_rvs[p] for p in var.parents)
    return dist_class(*par_numpyro_rvs)


def get_deterministic(var, numpyro_rvs):
    if var.cond_dist == interface.add:
        p1, p2 = var.parents
        return numpyro_rvs[p1] + numpyro_rvs[p2]
    else:
        raise Exception(f"can't handle deterministic {var}")


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

    print(f"{all_vars=}")

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

            if isinstance(var.cond_dist, interface.Constant):
                numpyro_rv = numpyro.deterministic(name, var.cond_dist.value)
            elif var.cond_dist.random:
                d = get_dist(var, numpyro_rvs)
                numpyro_rv = numpyro.sample(name, d)
            else:
                numpyro_rv = numpyro.deterministic(
                    name, get_deterministic(var, numpyro_rvs)
                )

            numpyro_rvs[var] = numpyro_rv

    return model, names


# class VMapDist(dist.Distribution):
#
#     def __init__(self, base_dist, **args, *, validate_args=None):
#         self.args = args
#         super().__init__()
