from . import interface, dag, util
import jax.tree_util
import functools
from jax import numpy as jnp
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
import random

################################################################################
# Cast observed variables to arrays (and check that they have corresponding shapes)
################################################################################

def assimilate_vals(vars, vals):
    "convert vals to a pytree of arrays with the same shape as vars"
    new_vals = jax.tree_map(lambda var, val: jnp.array(val), vars, vals)
    flat_vars, vars_treedef = jax.tree_util.tree_flatten(vars)
    flat_vals, vals_treedef = jax.tree_util.tree_flatten(new_vals)
    assert vars_treedef == vals_treedef, "vars and vals must have same structure (after conversion to arrays)"
    for var, val in zip(flat_vars, flat_vals):
        assert var.shape == val.shape, "vars and vals must have matching shape (after conversion to arrays)"
    return new_vals


################################################################################
# First major primitive — evaluate conditional log probs on groups of nodes
################################################################################

def ancestor_log_prob_flat(vars, vals, given_vars, given_vals):
    # convert things to arrays and make sure shapes match
    vals = assimilate_vals(vars, vals)
    given_vals = assimilate_vals(given_vars, given_vals)

    for var in vars:
        assert var.cond_dist.is_random, "all vars must be random"
    for var in given_vars:
        assert var.cond_dist.is_random, "all given vars must be random"

    upstream_vars = dag.upstream_nodes(vars, block_condition=lambda node: (node in given_vars) and (node not in vars))

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
            logp = log_prob_dist(node.cond_dist, val, *parent_vals)
            l += logp
        else:
            assert node not in vars, "user error: non-random node included"
            val = eval_dist(node.cond_dist, *parent_vals)
        if node not in given_vars:
            computed_vals[node] = val
    return l


################################################################################
# Second major primitive — sample groups of nodes
################################################################################

def ancestor_sample_flat(key, vars, given_vars, given_vals):
    given_vals = assimilate_vals(given_vars, given_vals)

    # do NOT insist that vars or given_vars be random

    upstream_vars = dag.upstream_nodes(vars, block_condition=lambda node: node in given_vars)

    computed_vals = util.WriteOnceDict(zip(given_vars, given_vals))
    for node in upstream_vars:
        for p in node.parents:
            assert p in computed_vals, "bug: all parents should already be computed"
        parent_vals = [computed_vals[p] for p in node.parents]

        if node.cond_dist.is_random:
            key, subkey = jax.random.split(key)
            val = sample_dist(node.cond_dist, subkey, *parent_vals)
        else:
            val = eval_dist(node.cond_dist, *parent_vals)
        computed_vals[node] = val
    return [computed_vals[var] for var in vars]


################################################################################
# Inference workhorse— do MCMC
################################################################################

def variables_to_sample(requested_vars, given_vars):
    has_obs_descendent = dag.upstream_nodes(given_vars)
    children = dag.get_children(requested_vars + given_vars)

    vars = []
    processed_vars = []
    queue = requested_vars.copy()

    def unseen(node):
        return node not in queue and node not in processed_vars

    while queue:
        # print(f"{queue=}")
        node = queue.pop()
        if node.cond_dist.is_random and node in has_obs_descendent:
            vars.append(node)
        processed_vars.append(node)

        for p in node.parents:
            if unseen(p) and p not in given_vars:
                queue.append(p)
        for c in children[node]:
            if unseen(c) and c in has_obs_descendent:
                queue.append(c)
    return vars


def sample_flat(requested_vars, given_vars=None, given_vals=None, niter=1000):
    "do MCMC"

    if given_vars is None:
        given_vars = []
    if given_vals is None:
        given_vals = []

    given_vals = assimilate_vals(given_vars, given_vals)
    assert isinstance(requested_vars, list)
    for node in requested_vars:
        assert isinstance(node, interface.RV)

    # this variable splitting business is a bit of a mess (although seemingly correct)
    random_vars = variables_to_sample(requested_vars, given_vars)
    latent_vars = [node for node in random_vars if node not in given_vars]
    leaf_vars = [node for node in random_vars if node in given_vars]

    if latent_vars == []:
        print("skipping MCMC...")
        latent_samps = []
    else:
        # print(f"{random_vars=}")

        potential_fn = lambda latent_vals: -ancestor_log_prob_flat(latent_vars + given_vars, latent_vals + given_vals,
                                                                   given_vars, given_vals)

        all_zeros = list(map(lambda node: .01 + jnp.zeros(node.shape), latent_vars))

        test_logp = potential_fn(all_zeros)  # test call
        if jnp.isnan(test_logp):
            raise Exception("got nan for intial parameters")

        nuts_kernel = NUTS(potential_fn=potential_fn)
        #rng_key = jax.random.PRNGKey(0)
        key = jax.random.PRNGKey(random.randint(0,10**9))
        mcmc = MCMC(nuts_kernel, num_warmup=niter // 2, num_samples=niter, progress_bar=False)
        mcmc.run(key, init_params=all_zeros)
        latent_samps = mcmc.get_samples()


    #key = jax.random.PRNGKey(0)
    key = jax.random.PRNGKey(random.randint(0,10**9))
    sample_fn = lambda key, latent_vals: ancestor_sample_flat(key, requested_vars, latent_vars + given_vars,
                                                              latent_vals + given_vals)

    rng_key = jax.random.split(key, niter)
    requested_samps = jax.vmap(sample_fn)(rng_key, latent_samps)

    return requested_samps


def sample(vars, given_vars=None, given_vals=None, niter=1000):
    given_vals = assimilate_vals(given_vars, given_vals)

    flat_vars, vars_treedef = jax.tree_util.tree_flatten(vars)
    flat_given_vars, given_vars_treedef = jax.tree_util.tree_flatten(given_vars)
    flat_given_vals, given_vals_treedef = jax.tree_util.tree_flatten(given_vals)
    assert given_vars_treedef == given_vals_treedef

    flat_samps = sample_flat(flat_vars, flat_given_vars, flat_given_vals, niter=niter)
    return jax.tree_util.tree_unflatten(vars_treedef, flat_samps)


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
    elif type(cond_dist) in sample_funs:
        fn = sample_funs[type(cond_dist)]
    else:
        raise Exception(f"sample_funs[{cond_dist}] and sample_funs[{type(cond_dist)}] unknown")
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


# normal_scale = AllScalarCondDist(2, "normal_scale", True)
# normal_prec = AllScalarCondDist(2, "normal_prec", True)
# bernoulli = AllScalarCondDist(1, "bernoulli", True)
# binomial = AllScalarCondDist(2, "binomial", True)
# beta = AllScalarCondDist(2, "beta", True)
# exponential = AllScalarCondDist(2, "exponential", True)
# beta_binomial = AllScalarCondDist(3, "beta_binomial", True)

# TODO: holy god what a mess

log_prob_funs[interface.uniform] = get_numpyro_dist_log_prob_fun(dist.Uniform, 2)
log_prob_funs[interface.normal_scale] = get_numpyro_dist_log_prob_fun(dist.Normal, 2)
log_prob_funs[interface.bernoulli] = get_numpyro_dist_log_prob_fun(dist.Bernoulli, 1)
log_prob_funs[interface.categorical] = get_numpyro_dist_log_prob_fun(dist.Categorical, 1)
log_prob_funs[interface.dirichlet] = get_numpyro_dist_log_prob_fun(dist.Dirichlet, 1)
log_prob_funs[interface.binomial] = get_numpyro_dist_log_prob_fun(dist.Binomial, 2)
log_prob_funs[interface.beta] = get_numpyro_dist_log_prob_fun(dist.Beta, 2)
log_prob_funs[interface.exponential] = get_numpyro_dist_log_prob_fun(dist.Exponential, 1)
log_prob_funs[interface.beta_binomial] = get_numpyro_dist_log_prob_fun(dist.BetaBinomial, 3)
log_prob_funs[interface.multinomial] = get_numpyro_dist_log_prob_fun(dist.Multinomial, 2)

sample_funs[interface.uniform] = get_numpyro_dist_sample_fun(dist.Uniform, 2)
sample_funs[interface.normal_scale] = get_numpyro_dist_sample_fun(dist.Normal, 2)
sample_funs[interface.categorical] = get_numpyro_dist_sample_fun(dist.Categorical, 1)
sample_funs[interface.dirichlet] = get_numpyro_dist_sample_fun(dist.Dirichlet, 1)
sample_funs[interface.beta] = get_numpyro_dist_sample_fun(dist.Beta, 2)
sample_funs[interface.exponential] = get_numpyro_dist_sample_fun(dist.Exponential, 1)
sample_funs[interface.multinomial] = get_numpyro_dist_sample_fun(dist.Multinomial, 2)


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
eval_funs[interface.exp] = deterministic_evaluator(jnp.exp)
eval_funs[interface.matmul] = deterministic_evaluator(jnp.matmul)

def eval_sum(cond_dist, parent_val):
    return jnp.sum(parent_val, cond_dist.axis)


eval_funs[interface.Sum] = eval_sum

def eval_index(cond_dist, val, *indices):
    # TODO: go through slices, extract where appropriate, etc
    #return node[indices]
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

eval_funs[interface.Index] = eval_index


def eval_vmap_deterministic(cond_dist, *parent_vals):
    my_eval = functools.partial(eval_dist, cond_dist.base_cond_dist)
    vals = jax.vmap(my_eval, in_axes=cond_dist.in_axes, axis_size=cond_dist.axis_size)(*parent_vals)
    return vals


eval_funs[interface.VMapDist] = eval_vmap_deterministic


def log_prob_vmap_dist(cond_dist, observed_val, *parent_vals):
    my_log_prob = functools.partial(log_prob_dist, cond_dist.base_cond_dist)
    logps = jax.vmap(my_log_prob, in_axes=(0,) + cond_dist.in_axes, axis_size=cond_dist.axis_size)(observed_val,
                                                                                                   *parent_vals)
    return jnp.sum(logps)


log_prob_funs[interface.VMapDist] = log_prob_vmap_dist


def sample_vmap_dist(cond_dist, rng_key, *parent_vals):
    my_sample = functools.partial(sample_dist, cond_dist.base_cond_dist)
    rng_keys = jax.random.split(rng_key, cond_dist.axis_size)
    samps = jax.vmap(my_sample, in_axes=(0,) + cond_dist.in_axes, axis_size=cond_dist.axis_size)(rng_keys,
                                                                                                 *parent_vals)
    # print(f"{samps.shape=}")
    return samps


sample_funs[interface.VMapDist] = sample_vmap_dist


def log_prob_mixture(cond_dist, observed_val, *parent_vals):
    assert isinstance(cond_dist, interface.Mixture)

    my_log_prob = functools.partial(log_prob_dist, cond_dist.component_cond_dist)

    weight, *parent_vals = parent_vals

    #in_axes = (None,) + (0,) * len(parent_vals)
    in_axes = (None,) + cond_dist.in_axes

    assert len(weight.shape) == 1
    axis_size = weight.shape[0]

    logps = jax.vmap(my_log_prob, in_axes=in_axes, axis_size=axis_size)(observed_val, *parent_vals)

    return jnp.log(jnp.sum(jnp.exp(logps + jnp.log(weight))))

log_prob_funs[interface.Mixture] = log_prob_mixture


def sample_mixture(cond_dist, key, *parent_vals):
    assert isinstance(cond_dist, interface.Mixture)

    my_sample = functools.partial(sample_dist, cond_dist.component_cond_dist)

    weight, *parent_vals = parent_vals

    in_axes = (0,) + cond_dist.in_axes
    assert len(weight.shape) == 1
    axis_size = weight.shape[0]

    key, subkey = jax.random.split(key)
    subkeys = jax.random.split(subkey,axis_size)
    vals = jax.vmap(my_sample, in_axes=in_axes, axis_size=axis_size)(subkeys, *parent_vals)

    key, subkey = jax.random.split(key)
    r = jax.random.uniform(subkey) # uniform random sample
    i = jnp.sum(jnp.cumsum(weight) < r) # integer class
    I = jax.nn.one_hot(i, num_classes=weight.shape[0]) # one-hot vector
    return jnp.tensordot(I,vals,axes=[0,0]) # inner-product with samples

sample_funs[interface.Mixture] = sample_mixture

def eval_cond_prob(cond_dist, *parent_vals):
    assert isinstance(cond_dist, interface.CondProb)
    observed_val, *parent_vals = parent_vals
    return jnp.exp(log_prob_dist(cond_dist.base_cond_dist, observed_val, *parent_vals))

eval_funs[interface.CondProb] = eval_cond_prob

# ################################################################################
# # Functions to transform a DAG
# ################################################################################
#
# class InapplicableTransformationError(Exception):
#     pass
#
#
# def has_observed_descendant(node, observed_vars, blockers=()):
#     return node in dag.upstream_nodes(observed_vars, block_condition=lambda node: node in blockers)
#
#
# # a transformation rule takes a node and its parents as input
# # it outputs a new node and new parents
# # it does not modify any of the original arguments
# # however, the new node and/or new parents should point back to the ORIGINAL nodes not the new ones
# # so when we later modify RV cond_dists and parents in-place the graph will be correct
# # the apply method cannot look at parents or inspect cond_dists (so it can run on abstract RVs)
#
# class TransformationRule:
#     @classmethod
#     def check(cls, node, observed_vars):
#         pass
#
#     @classmethod
#     def extract(cls, node):
#         pass
#
#     @classmethod
#     def apply(cls, node, *extracted):
#         "return new node and new parent"
#         pass
#
#
# class BetaBinomialTransformationRule(TransformationRule):
#     """
#     beta -> binomial ==> beta_binomial -> beta
#     """
#
#     @classmethod
#     def check(cls, node, observed_vars):
#         parents = node.parents
#         if len(parents) == 2:
#             n, p = parents
#         else:
#             raise InapplicableTransformationError("incorrect # of args")
#
#         if node.cond_dist != interface.binomial:
#             raise InapplicableTransformationError("not binomial")
#
#         if p.cond_dist != interface.beta:
#             raise InapplicableTransformationError("p not beta")
#
#         if not has_observed_descendant(node, observed_vars):
#             raise InapplicableTransformationError("node doesn't have observed descendant")
#
#         if has_observed_descendant(p, observed_vars, [node]):
#             raise InapplicableTransformationError("p has other observed descendant")
#
#         if dag.has_second_path(node, 1):
#             raise InapplicableTransformationError("second path")
#
#     @classmethod
#     def extract(cls, node):
#         n, p = node.parents
#         a, b = p.parents
#         return n, p, a, b
#
#     @classmethod
#     def apply(cls, node, n, p, a, b):
#         new_node = interface.beta_binomial(n, a, b)
#         reversed_p = interface.beta(a + node, b + n - node)  # use OLD node so graph structure is preserved
#         return new_node, n, reversed_p
#
#
# class NonCenteredNormalTransformationRule(TransformationRule):
#     @classmethod
#     def check(cls, node, *parents, observed_vars):
#         if node.cond_dist != interface.normal_scale:
#             raise InapplicableTransformationError("not normal_scale")
#         loc, scale = node.parents
#         if (isinstance(loc.cond_dist, interface.Constant) and
#                 isinstance(scale.cond_dist, interface.Constant) and
#                 loc.cond_dist.value == 0 and
#                 scale.cond_dist.value == 1):
#             raise InapplicableTransformationError("already standardized")
#
#     @classmethod
#     def extract(cls, node):
#         return node.parents
#
#     @classmethod
#     def apply(cls, node, *parents):
#         #print(f"NONCENTERED APPLY {node=} {parents=}")
#         loc, scale = parents
#         dummy = interface.normal_scale(0, 1)
#         new_node = loc + scale * dummy
#         new_parents = parents
#         #print(f"{[p1==p2 for p1,p2 in zip(parents, new_parents)]=}")
#         return new_node, *new_parents
#
#
# class NormalNormalTransformationRule(TransformationRule):
#     @classmethod
#     def check(cls, node, observed_vars):
#         parents = node.parents
#         if len(parents) == 2:
#             p, c = parents
#         else:
#             raise InapplicableTransformationError("incorrect # of args")
#
#         if node.cond_dist != interface.normal_scale:
#             raise InapplicableTransformationError("node not normal_scale")
#
#         if p.cond_dist != interface.normal_scale:
#             raise InapplicableTransformationError("p not normal_scale")
#
#         if not has_observed_descendant(node, observed_vars):
#             raise InapplicableTransformationError("node doesn't have observed descendant")
#
#         if has_observed_descendant(p, observed_vars, [node]):
#             raise InapplicableTransformationError("p has other observed descendant")
#
#         if dag.has_second_path(node, 1):
#             raise InapplicableTransformationError("second path")
#
#     @classmethod
#     def extract(cls, node):
#         p, c = node.parents
#         a, b = p.parents
#         return p, a, b, c
#
#     @classmethod
#     def apply(cls, node, p, a, b, c):
#         new_node = interface.normal_scale(a, (b ** 2 + c ** 2) ** 0.5)
#         adj = (1 + c ** 2 / b ** 2)
#         new_mean = a + (node - a) / adj  # use OLD node
#         new_std = b * (1 - 1 / adj) ** 0.5
#         new_par = interface.normal_scale(new_mean, new_std)
#         return new_node, new_par, c
#
#
# class ConstantOpTransformationRule(TransformationRule):
#     @classmethod
#     def check(cls, node, observed_vars):
#         #print(f"{node=}")
#         if node in observed_vars or any(p in observed_vars for p in node.parents):
#             raise InapplicableTransformationError("AA node is observed")
#         if any(p in observed_vars for p in node.parents):
#             raise InapplicableTransformationError("AA parent is observed")
#         if node.cond_dist.is_random:
#             raise InapplicableTransformationError("AA random")
#         if isinstance(node.cond_dist, interface.Constant):
#             raise InapplicableTransformationError("AA constant")
#         if not all(isinstance(p.cond_dist, interface.Constant) for p in node.parents):
#             raise InapplicableTransformationError("AA all parents not constant")
#
#     @classmethod
#     def extract(cls, node):
#         return node.parents
#
#     @classmethod
#     def apply(cls, node, *parents):
#         parent_values = [p.cond_dist.value for p in parents]
#         new_value = eval_dist(node.cond_dist, *parent_values)
#         new_node = interface.RV(interface.Constant(new_value))
#         return new_node, *parents
#
#
# class VMapTransformationRule(TransformationRule):
#     def __init__(self, base_rule):
#         self.base_rule = base_rule
#
#     def check(self, node, *parents, observed_vars):
#         if not isinstance(node.cond_dist, interface.VMapDist):
#             raise InapplicableTransformationError("not a vmapdist")
#         # TODO: need to call base rule check (somehow)
#         # need to create entire dummy graph? (could look at parents of parents..)
#         # should we have an "undo vmap" operator?
#         # this would take a VMapper node and return a dummy tree of all ancestors
#         # where all dimensions corresponding to the vmap are traced out
#         # TODO: ALSO: need to make sure we don't change any sizes of parents...
#
#     def extract(self, node, *parents):
#         return self.base_rule.extract(node, *parents)
#
#     def apply(self, node, *parents):
#         # base_apply = lambda node_i,*parents_i: self.base_rule.apply(node_i,*parents_i)
#         base_apply = self.base_rule.apply
#
#         print(f"{node.cond_dist.in_axes=}")
#         print(f"{node.parents=}")
#
#         # return interface.vmap(base_apply, (0,) + node.cond_dist.in_axes)(node, *parents)
#         new_node, *new_parents = interface.vmap(base_apply, (0,) + node.cond_dist.in_axes)(node, *parents)
#         return new_node, *new_parents
#
#
# # possible tforms
# # normal(0,1)**2 -> chi2(1))
# # sum(plate(N=n)(lambda:chi2(1)))) -> chi2(n)
# # but what about linear transformations in between?
# # normal(0,1) / sqrt(chi2(r)) -> studentt(r)
# # but what if there are extra affine terms?
#
# # normal(0,1)
#
# # TODO:
# # should we enforce no second path in the form rules?
#
# def apply_transformation_rules(vars, rules, observed_vars=(), max_iter=-1):
#     assert isinstance(vars, list)
#     for var in vars:
#         assert isinstance(var, interface.RV)
#
#     old_vars = dag.upstream_nodes(vars)
#
#     # 1: Get upstream nodes
#     # 2: Make a full copy of the RVs in the graph
#     # 3: Operate in-place on the new nodes, modifying cond_dist and parents, but not RV identities
#
#     old_to_new = {}
#     # new_observed_vars = []
#     for old_var in old_vars:
#         new_var = interface.RV(old_var.cond_dist, *[old_to_new[p] for p in old_var.parents])
#         old_to_new[old_var] = new_var
#         # if old_var in observed_vars:
#         #    new_observed_vars.append(new_var)
#
#     def copy_node(node, new_node):
#         node.cond_dist = new_node.cond_dist
#         node.parents = new_node.parents
#
#     def run_loop():
#         new_nodes = list(old_to_new.values())
#         up = dag.upstream_nodes(new_nodes)
#         new_observed_vars = [old_to_new[var] for var in observed_vars]
#         print(f"{new_nodes=}")
#         print(f"{new_observed_vars=}")
#
#         for node in up:
#             for rule in rules:
#                 try:
#                     # new_node, *new_pars = rule.apply(node, *node.parents, observed_vars = new_observed_vars)
#                     rule.check(node, observed_vars=new_observed_vars)
#                     print(f"applying {rule}")
#                     extracted = rule.extract(node)
#                     new_node, *new_pars = rule.apply(node, *extracted)
#
#                     # # update observed nodes
#                     # if node in new_observed_vars:
#                     #     new_observed_vars[new_observed_vars.index(node)] = new_node
#                     # for (par, new_par) in zip(node.parents, new_pars):
#                     #     if par in new_observed_vars:
#                     #         new_observed_vars[new_observed_vars.index(par)] = new_par
#
#                     assert len(new_pars) == len(node.parents)
#                     for old_p, new_p in zip(node.parents, new_pars):
#                         copy_node(old_p, new_p)
#                     copy_node(node, new_node)
#                     return True
#                 except InapplicableTransformationError as e:
#                     print(f"{e=}")
#                     continue
#         return False
#
#     i = 0
#     while run_loop():
#         i += 1
#         if i == max_iter:
#             break
#
#     return [old_to_new[var] for var in vars]
