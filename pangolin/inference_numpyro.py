from . import interface, dag, util, inference
import jax.tree_util
import functools
from jax import numpy as jnp
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
import random


################################################################################
# First major primitive — evaluate conditional log probs on groups of nodes
################################################################################


def ancestor_log_prob_flat(vars, vals, given_vars, given_vals):
    """
    Compute the (conditional) joint log probability of `vars == vals` given
    `given_vars == given_vals`.
    * `vars` - a list of `RV`s
    * `vals` - a list of values for the `RV`s
    * `given_vars` - list of `RV`s to condition on
    * `given_vals` - list of values for the `RV`s to condition on
    """

    # convert things to arrays and make sure shapes match
    vals = util.assimilate_vals(vars, vals)
    given_vals = util.assimilate_vals(given_vars, given_vals)

    for var in vars:
        assert var.cond_dist.is_random, "all vars must be random"
    for var in given_vars:
        assert var.cond_dist.is_random, "all given vars must be random"

    upstream_vars = dag.upstream_nodes(
        vars, block_condition=lambda node: (node in given_vars) and (node not in vars)
    )

    input_vals = util.WriteOnceDict(zip(vars, vals))
    computed_vals = util.WriteOnceDict(zip(given_vars, given_vals))

    l = 0.0
    for node in upstream_vars:
        for p in node.parents:
            assert p in computed_vals, "bug: all parents should already be computed"
        parent_vals = [computed_vals[p] for p in node.parents]

        if node.cond_dist.is_random:
            assert node in vars, "user error: random node in tree not included"
            assert node in input_vals, "bug"

            val = input_vals[node]
            # TODO: update
            logp = log_prob(node.cond_dist, val, *parent_vals)
            l += logp
        else:
            assert node not in vars, "user error: non-random node included"
            # TODO: update
            val = evaluate(node.cond_dist, *parent_vals)
        if node not in given_vars:
            computed_vals[node] = val
    return l


################################################################################
# Second major primitive — sample groups of nodes
################################################################################


def ancestor_sample_flat(key, vars, given_vars, given_vals):
    """
    This samples a bunch of `RV`s using ancestor sampling.
    * `key` - a `jax.random.PRNGKey`
    * `vars` - a list of `RV`s
    * `given_vars` - list of `RV`s to condition on
    * `given_vals` - list of values for the `RV`s to condition on

    Notes:
    * If extra nodes above those listed need to be sampled, they
    will automatically be included
    * It's fine if the `RV`s are determinstic. In that case they
    are just evaluated.
    * If any variable in `given_vars` is downstream of any variable
    in `vars` this will still run, but won't give correct samples.
    """

    given_vals = util.assimilate_vals(given_vars, given_vals)

    # do NOT insist that vars or given_vars be random

    upstream_vars = dag.upstream_nodes(
        vars, block_condition=lambda node: node in given_vars
    )

    computed_vals = util.WriteOnceDict(zip(given_vars, given_vals))
    for node in upstream_vars:
        for p in node.parents:
            assert p in computed_vals, "bug: all parents should already be computed"
        parent_vals = [computed_vals[p] for p in node.parents]

        if node.cond_dist.is_random:
            key, subkey = jax.random.split(key)
            val = sample(node.cond_dist, subkey, *parent_vals)
        else:
            val = evaluate(node.cond_dist, *parent_vals)
        computed_vals[node] = val
    return [computed_vals[var] for var in vars]


################################################################################
# Inference workhorse— do MCMC
################################################################################


# should this be generic?
# should be in DAG


def sample_flat(requested_vars, given_vars, given_vals, *, niter):
    """
    Do MCMC using Numpyro.

    Inputs:
    * `requested_vars`: A list of `RV`s you want to sample
    * `given_vars`: A list of `RV`s you want to condition on
    * `given_vals`: A list of numpy arrays with values for `given_vars`
    """

    assert isinstance(given_vars, list)
    assert isinstance(given_vals, list)
    assert isinstance(requested_vars, list)

    for node in requested_vars:
        assert isinstance(node, interface.RV)

    # given_vals = util.assimilate_vals(given_vars, given_vals)

    assert len(given_vars) == len(given_vals)
    for var, val in zip(given_vars, given_vals):
        assert var.shape == val.shape

    # this variable splitting business is a bit of a mess (although seemingly correct)
    random_vars = inference.upstream_with_descendent(requested_vars, given_vars)
    latent_vars = [node for node in random_vars if node not in given_vars]

    if not latent_vars:  # could be that latent_vars == []
        # print("skipping MCMC...")
        latent_samps = []
    else:
        # print(f"{random_vars=}")

        def potential_fn(latent_vals):
            return -ancestor_log_prob_flat(
                latent_vars + given_vars,
                latent_vals + given_vals,
                given_vars,
                given_vals,
            )

        all_zeros = list(
            map(lambda my_node: 0.01 + jnp.zeros(my_node.shape), latent_vars)
        )

        test_logp = potential_fn(all_zeros)  # test call
        if jnp.isnan(test_logp):
            raise Exception("got nan for intial parameters")

        nuts_kernel = NUTS(potential_fn=potential_fn)

        key = jax.random.PRNGKey(random.randint(0, 10**9))
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=niter // 2,
            num_samples=niter,
            progress_bar=False,
        )
        mcmc.run(key, init_params=all_zeros)
        latent_samps = mcmc.get_samples()

    key = jax.random.PRNGKey(random.randint(0, 10**9))
    sample_fn = lambda key, latent_vals: ancestor_sample_flat(
        key, requested_vars, latent_vars + given_vars, latent_vals + given_vals
    )

    rng_key = jax.random.split(key, niter)
    requested_samps = jax.vmap(sample_fn)(rng_key, latent_samps)

    return requested_samps


################################################################################
# Functions to evaluate individual nodes
################################################################################


def log_prob(cond_dist, observed_val, *parent_vals):
    "compute log probabilities for a single cond_dist"
    assert cond_dist.is_random
    dist_class = type(cond_dist)
    if cond_dist in numpyro_dists:
        return numpyro_dists[cond_dist](*parent_vals).log_prob(observed_val)
    elif cond_dist in log_prob_funs:
        raise NotImplementedError("not activated")  # line below "should" work
        # return log_prob_funs[cond_dist](observed_val,*parent_vals)
    elif dist_class in class_log_prob_funs:
        return class_log_prob_funs[dist_class](cond_dist, observed_val, *parent_vals)
    else:
        raise NotImplementedError()


def sample(cond_dist, key, *parent_vals):
    "sample a single cond_dist"
    assert cond_dist.is_random
    dist_class = type(cond_dist)
    if cond_dist in numpyro_dists:
        return numpyro_dists[cond_dist](*parent_vals).sample(key)
    elif cond_dist in sample_funs:
        raise NotImplementedError("not activated")  # line below "should" work
        # return sample_funs[cond_dist](key,*parent_vals)
    elif dist_class in class_sample_funs:
        return class_sample_funs[dist_class](cond_dist, key, *parent_vals)
    else:
        raise NotImplementedError(f"no sample fun implemented for {cond_dist}")


def evaluate(cond_dist, *parent_vals):
    """
    evaluate a single cond_dist
    """
    assert not cond_dist.is_random
    dist_class = type(cond_dist)
    if cond_dist in evaluation_funs:
        return evaluation_funs[cond_dist](*parent_vals)
    elif dist_class in class_evaluation_funs:
        return class_evaluation_funs[dist_class](cond_dist, *parent_vals)
    else:
        raise NotImplementedError(f"no eval fun implemented for {cond_dist}")


################################################################################
# Functions to sample / log_prob / evaluate different cond_dists
################################################################################

numpyro_dists = {
    interface.uniform: dist.Uniform,
    interface.normal_scale: dist.Normal,
    interface.bernoulli: dist.Bernoulli,
    interface.bernoulli_logit: dist.BernoulliLogits,
    interface.categorical: dist.Categorical,
    interface.dirichlet: dist.Dirichlet,
    interface.binomial: dist.Binomial,
    interface.beta: dist.Beta,
    interface.exponential: dist.Exponential,
    interface.beta_binomial: dist.BetaBinomial,
    interface.multinomial: dist.Multinomial,
    interface.multi_normal_cov: dist.MultivariateNormal,
}


def log_prob_vmap(cond_dist, observed_val, *parent_vals):
    my_log_prob = functools.partial(log_prob, cond_dist.base_cond_dist)
    logps = jax.vmap(
        my_log_prob, in_axes=(0,) + cond_dist.in_axes, axis_size=cond_dist.axis_size
    )(observed_val, *parent_vals)
    return jnp.sum(logps)


def sample_vmap(cond_dist, rng_key, *parent_vals):
    # print(f"{cond_dist=}")
    # print(f"{rng_key=}")
    # print(f"{parent_vals=}")
    my_sample = functools.partial(sample, cond_dist.base_cond_dist)
    rng_keys = jax.random.split(rng_key, cond_dist.axis_size)
    samps = jax.vmap(
        my_sample, in_axes=(0,) + cond_dist.in_axes, axis_size=cond_dist.axis_size
    )(rng_keys, *parent_vals)
    # print(f"{samps.shape=}")
    return samps


log_prob_funs = {}  # none for now!

class_log_prob_funs = {interface.VMapDist: log_prob_vmap}

sample_funs = {}  # none for now

class_sample_funs = {interface.VMapDist: sample_vmap}


def eval_index(cond_dist: interface.Index, val, *indices):
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


def eval_vmap(cond_dist, *parent_vals):
    my_eval = functools.partial(evaluate, cond_dist.base_cond_dist)
    vals = jax.vmap(my_eval, in_axes=cond_dist.in_axes, axis_size=cond_dist.axis_size)(
        *parent_vals
    )
    return vals


evaluation_funs = {
    interface.add: lambda a, b: a + b,
    interface.sub: lambda a, b: a - b,
    interface.mul: lambda a, b: a * b,
    interface.div: lambda a, b: a / b,
    interface.pow: lambda a, b: a**b,
    interface.abs: jnp.abs,
    interface.exp: jnp.exp,
    interface.matmul: jnp.matmul,
}

class_evaluation_funs = {
    interface.Constant: lambda cond_dist: cond_dist.value,
    interface.Sum: lambda cond_dist, a: jnp.sum(a, axis=cond_dist.axis),
    interface.Index: eval_index,  # implemented above
    interface.VMapDist: eval_vmap,
}
