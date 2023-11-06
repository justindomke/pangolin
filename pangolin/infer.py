import numpyro.distributions.transforms

import interface
import dag
import numpy as np
import jax.tree_util
import util
import functools
from jax import numpy as jnp

from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS


################################################################################
# Functions to call MCMC
################################################################################

def variables_to_sample(requested_vars, given_vars):
    # 1. upstream of given variables (so they can be influenced)
    # 2. upstream of requested variables (so they actually matter)
    # 3. not actually included as given variables
    # 4. random

    assert isinstance(requested_vars, list)
    assert isinstance(given_vars, list)
    upstream_of_given = set(dag.upstream_nodes(given_vars))  # all relevent vars
    upstream_of_requested = set(dag.upstream_nodes(requested_vars))
    upstream_vars = upstream_of_given & upstream_of_requested - set(given_vars)
    return list(node for node in upstream_vars if node.cond_dist.is_random)  # run MCMC on randoms only


def cast_to_arrays(vars, vals):
    """
    Given a pytree of random variables and a corresponding pytree of values, cast all the values to arrays.
    (You can't do this for values alone because it's ambiguous if [1,2,3] is a list of three values or an array.)
    """
    return jax.tree_util.tree_map(
        lambda var, val: None if val is None else jnp.array(val),
        vars,
        vals
    )


def sample(vars, given_vars=None, given_vals=None, niter=1000):
    given_vals = cast_to_arrays(given_vars, given_vals)
    vars_flat, treedef = jax.tree_util.tree_flatten(vars)
    given_vars_flat, given_treedef_flat = jax.tree_util.tree_flatten(given_vars)
    given_vals_flat, given_treedef_flat2 = jax.tree_util.tree_flatten(given_vals)

    samps_flat = sample_flat(vars_flat, given_vars_flat, given_vals_flat, niter=niter)
    samps = jax.tree_util.tree_unflatten(treedef, samps_flat)
    return samps


def sample_flat(requested_vars, given_vars=[], given_vals=[], niter=1000):
    assert isinstance(requested_vars, list)
    assert isinstance(given_vars, list)
    assert isinstance(given_vals, list)
    assert len(given_vars) == len(given_vals)

    print(f"{requested_vars=}")
    print(f"{given_vars=}")
    print(f"{given_vals=}")

    random_vars = variables_to_sample(requested_vars, given_vars)

    eval_fn = cond_log_prob_flat(random_vars, given_vars)
    potential_fn = lambda random_vals: -eval_fn(random_vals, given_vals)
    all_zeros = list(map(lambda node: .01 + np.zeros(node.shape), random_vars))

    test_logp = potential_fn(all_zeros)  # test call
    if np.isnan(test_logp):
        raise Exception("got nan for intial parameters")

    nuts_kernel = NUTS(potential_fn=potential_fn)
    rng_key = jax.random.PRNGKey(0)
    mcmc = MCMC(nuts_kernel, num_warmup=niter // 2, num_samples=niter, progress_bar=False)
    mcmc.run(rng_key, init_params=all_zeros)

    # expand to all related variables
    random_samps = mcmc.get_samples()  # str -> samps

    print(f"{random_samps=}")

    upstream_vars = list(dag.upstream_nodes(requested_vars))

    sample_fn = ancestor_sample_flat(upstream_vars)

    indices = [random_vars.index(n) if n in random_vars else None for n in upstream_vars]
    initial_upstream_samps = [random_samps[i] if i is not None else None for i in indices]

    rng_key = jax.random.split(rng_key, niter)
    upstream_samps = jax.vmap(sample_fn)(rng_key, initial_upstream_samps)

    indices = [upstream_vars.index(n) for n in requested_vars]
    requested_samps = [upstream_samps[i] for i in indices]

    return requested_samps


def check_shapes(vars, vals):
    # must be lists
    assert isinstance(vars, list)
    assert isinstance(vals, list)
    # must be same length
    assert len(vars) == len(vals)
    for var, val in zip(vars, vals):
        # var must be RV
        assert isinstance(var, interface.RV)
        # val must be None or same shape as var
        if val is None:
            continue
        assert hasattr(val, 'shape')
        assert var.shape == val.shape


################################################################################
# Functions to compute conditional log probabilities on sets of nodes
################################################################################

def cond_log_prob(vars, given_vars=[]):
    "compute log probabilies—this function isn't used internally but user might like it"
    flat_vars, vars_treedef = jax.tree_util.tree_flatten(vars)
    flat_given_vars, given_vars_treedef = jax.tree_util.tree_flatten(given_vars)

    flat_fn = cond_log_prob_flat(flat_vars, flat_given_vars)

    def fn(vals, given_vals=[]):
        # cast vals to arrays using structure of vars
        vals = jax.tree_util.tree_map(lambda var, val: jnp.array(val), vars, vals)
        given_vals = jax.tree_util.tree_map(lambda var, val: jnp.array(val), given_vars, given_vals)

        flat_vals, vals_treedef = jax.tree_util.tree_flatten(vals, is_leaf=util.is_leaf_with_none)
        assert vals_treedef == vars_treedef, "treedef must match"
        flat_given_vals, given_vals_treedef = jax.tree_util.tree_flatten(given_vals, is_leaf=util.is_leaf_with_none)
        assert given_vals_treedef == given_vars_treedef, "treedef must match"
        return flat_fn(flat_vals, flat_given_vals)

    return fn


def convert_flat_vals(flat_vals):
    "Make sure it's a list. Also, convert each component to an array if it isn't already"
    assert isinstance(flat_vals, list)
    return list(map(lambda x: None if x is None else jnp.array(x), flat_vals))


def cond_log_prob_flat(flat_vars, flat_given_vars=[]):
    """
    both compute the log probability and propagate values for deterministic nodes)
    """

    # transform a collection of random variables into an evaluable potential function
    assert isinstance(flat_vars, list)
    assert isinstance(flat_given_vars, list)

    upstream_vars = dag.upstream_nodes(flat_vars, block_condition=lambda node: node in flat_given_vars)

    def fn(flat_vals, given_vals=[]):
        flat_vals = convert_flat_vals(flat_vals)
        given_vals = convert_flat_vals(given_vals)
        check_shapes(flat_vars, flat_vals)
        check_shapes(flat_given_vars, given_vals)

        input_vals = dict(zip(flat_vars, flat_vals))
        # del flat_vals  # never touched again
        all_vals = util.WriteOnceDict(zip(flat_given_vars, given_vals))

        log_prob = 0.0
        for node in upstream_vars:
            print(f"{node=} {log_prob=}")
            parent_vals = [all_vals[p] for p in node.parents]
            if node.cond_dist.is_random:
                # observed_val = input_vals.get(node)  # None if not present
                observed_val = input_vals[node]
                logp = log_prob_dist(node.cond_dist, observed_val, *parent_vals)
                print(f"{logp=} {observed_val=} {parent_vals=}")
                log_prob += logp
                all_vals[node] = observed_val
            else:
                val = eval_dist(node.cond_dist, *parent_vals)
                all_vals[node] = val

        # flat_outval = [all_vals[node] for node in flat_vars]
        print(f"{log_prob=}")
        return log_prob

    return fn


def ancestor_sample_flat(flat_vars, flat_given_vars=[]):
    """
    both compute the log probability and propagate values for deterministic nodes)
    """

    # transform a collection of random variables into an evaluable potential function
    assert isinstance(flat_vars, list)
    assert isinstance(flat_given_vars, list)

    upstream_vars = dag.upstream_nodes(flat_vars, block_condition=lambda node: node in flat_given_vars)

    def fn(key, flat_vals, given_vals=[]):
        flat_vals = convert_flat_vals(flat_vals)
        given_vals = convert_flat_vals(given_vals)
        check_shapes(flat_vars, flat_vals)
        check_shapes(flat_given_vars, given_vals)

        input_vals = dict(zip(flat_vars, flat_vals))
        # del flat_vals  # never touched again
        all_vals = util.WriteOnceDict(zip(flat_given_vars, given_vals))

        for node in upstream_vars:
            observed_val = input_vals.get(node)  # None if not present
            parent_vals = [all_vals[p] for p in node.parents]
            if node.cond_dist.is_random:
                key, subkey = jax.random.split(key)
                val = sample_dist(node.cond_dist, subkey, *parent_vals)
                all_vals[node] = val
            else:
                val = eval_dist(node.cond_dist, *parent_vals)
                all_vals[node] = val

        flat_outval = [all_vals[node] for node in flat_vars]
        return flat_outval

    return fn


################################################################################
# Function to evaluate individual nodes
################################################################################

def log_prob_dist(cond_dist, observed_val, *parent_vals):
    assert cond_dist.is_random
    if cond_dist in log_prob_funs:
        fn = log_prob_funs[cond_dist]
    else:
        fn = log_prob_funs[type(cond_dist)]
    return fn(cond_dist, observed_val, *parent_vals)


def eval_dist(cond_dist, *parent_vals):
    assert not cond_dist.is_random
    if cond_dist in eval_funs:
        fn = eval_funs[cond_dist]
    else:
        fn = eval_funs[type(cond_dist)]
    return fn(cond_dist, *parent_vals)


def sample_dist(cond_dist, rng_key, *parent_vals):
    assert cond_dist.is_random
    if cond_dist in sample_funs:
        fn = sample_funs[cond_dist]
    else:
        fn = sample_funs[type(cond_dist)]
    return fn(cond_dist, rng_key, *parent_vals)


################################################################################
# Functions to compute conditional log probabilities on individual nodes
################################################################################

# Could have two dicts, one for cond_dist and one for classes
log_prob_funs = {}
eval_funs = {}
sample_funs = {}


def eval_constant(cond_dist, *parent_vals):
    assert len(parent_vals) == 0, "constants should not have parents"
    return cond_dist.value


eval_funs[interface.Constant] = eval_constant


def get_numpyro_dist_log_prob_fun(NumPyroDist, num_parents):
    def fun(cond_dist, observed_val, *parent_vals):
        assert len(parent_vals) == num_parents, "incorrect number of arguments"
        return NumPyroDist(*parent_vals).log_prob(observed_val)

    return fun


def get_numpyro_dist_sample_fun(NumPyroDist, num_parents):
    def fun(cond_dist, rng_key, *parent_vals):
        assert len(parent_vals) == num_parents, "incorrect number of arguments"
        return NumPyroDist(*parent_vals).sample(rng_key)

    return fun


log_prob_funs[interface.normal_scale] = get_numpyro_dist_log_prob_fun(dist.Normal, 2)
log_prob_funs[interface.beta] = get_numpyro_dist_log_prob_fun(dist.Beta, 2)
log_prob_funs[interface.exponential] = get_numpyro_dist_log_prob_fun(dist.Exponential, 1)

sample_funs[interface.normal_scale] = get_numpyro_dist_sample_fun(dist.Normal, 2)
sample_funs[interface.beta] = get_numpyro_dist_sample_fun(dist.Beta, 2)
sample_funs[interface.exponential] = get_numpyro_dist_sample_fun(dist.Exponential, 1)


def deterministic_evaluator(fn):
    def eval_fun(cond_dist, *parent_vals):
        return fn(*parent_vals)

    return eval_fun


eval_funs[interface.add] = deterministic_evaluator(lambda a, b: a + b)
eval_funs[interface.sub] = deterministic_evaluator(lambda a, b: a - b)
eval_funs[interface.mul] = deterministic_evaluator(lambda a, b: a * b)
eval_funs[interface.div] = deterministic_evaluator(lambda a, b: a / b)
eval_funs[interface.pow] = deterministic_evaluator(lambda a, b: a ** b)
eval_funs[interface.abs] = deterministic_evaluator(jnp.abs)


def eval_vmap_deterministic(cond_dist, *parent_vals):
    my_eval = functools.partial(eval_dist, cond_dist.base_cond_dist)
    vals = jax.vmap(my_eval, in_axes=cond_dist.in_axes, axis_size=cond_dist.axis_size)(*parent_vals)
    return vals


eval_funs[interface.VMapDist] = eval_vmap_deterministic


def log_prob_vmap_dist(cond_dist, observed_val, *parent_vals):
    my_log_prob = functools.partial(log_prob_dist, cond_dist.base_cond_dist)
    logps = jax.vmap(my_log_prob, in_axes=[0] + cond_dist.in_axes, axis_size=cond_dist.axis_size)(observed_val,
                                                                                                  *parent_vals)
    return np.sum(logps)


log_prob_funs[interface.VMapDist] = log_prob_vmap_dist


def sample_vmap_dist(cond_dist, rng_key, *parent_vals):
    my_sample = functools.partial(sample_dist, cond_dist.base_cond_dist)
    rng_keys = jax.random.split(rng_key, cond_dist.axis_size)
    samps = jax.vmap(my_sample, in_axes=[0] + cond_dist.in_axes, axis_size=cond_dist.axis_size)(rng_keys,
                                                                                                *parent_vals)
    print(f"{samps.shape=}")
    return samps


sample_funs[interface.VMapDist] = sample_vmap_dist

################################################################################
# Functions to compute conditional log probabilities on individual nodes
################################################################################

# def ancestor_sample_flat(flat_vars, flat_given_vars=[]):
#     """
#     do ancestor sampling
#     """
#
#     # transform a collection of random variables into an evaluable potential function
#     assert isinstance(flat_vars, list)
#     assert isinstance(flat_given_vars, list)
#
#     upstream_vars = dag.upstream_nodes(flat_vars, block_condition=lambda node: node in flat_given_vars)
#
#     def fn(key, flat_vals, given_vals=[]):
#         check_shapes(flat_vars, flat_vals)
#         check_shapes(flat_given_vars, given_vals)
#
#         input_vals = dict(zip(flat_vars, flat_vals))
#         # del flat_vals  # never touched again
#         all_vals = util.WriteOnceDict(zip(flat_given_vars, given_vals))
#
#         log_prob = 0.0
#         for node in upstream_vars:
#             observed_val = input_vals.get(node)  # None if not present
#             parent_vals = [all_vals[p] for p in node.parents]
#             all_vals[node], logp = eval_dist(node.cond_dist, observed_val, *parent_vals)
#             log_prob += logp
#
#         flat_outval = [all_vals[node] for node in flat_vars]
#         return log_prob, flat_outval
#
#     return fn


# def eval_transformed_dist(cond_dist, observed_val, *parent_vals):
#     tform = cond_dist.transform
#     transformed_val, ljd = tform_funs[tform](observed_val)
#     _, base_logp = eval_dist(cond_dist.base_cond_dist, transformed_val, *parent_vals)
#     return observed_val, base_logp + ljd
#
#
# eval_funs[interface.TransformedCondDist] = eval_transformed_dist

# tform_funs = {}
#
#
# # for each, should evaluate x = f^{-1}(y) and log(∇f^{-1}(y)) = -log(∇f(x))
#
# def eval_inverse_softplus(y):
#     # print(f"{y=}")
#     f_inv = numpyro.distributions.transforms.SoftplusTransform()
#     x = f_inv(y)
#     ljd = f_inv.log_abs_det_jacobian(x, y)
#     return x, ljd
#
#
# tform_funs[interface.inverse_softplus] = eval_inverse_softplus
#
#
# def eval_softplus(y):
#     # print(f"{y=}")
#     f = numpyro.distributions.transforms.SoftplusTransform()
#     x = f.inv(y)
#     ljd = -f.log_abs_det_jacobian(x, y)
#     return x, ljd
#
#
# tform_funs[interface.softplus] = eval_softplus
