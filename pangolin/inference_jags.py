"""
how should this work?
- need a routine to generate code for a single distribution
- that routine should NOT take a RV as input, otherwise things will become a
nightmare when we get to vmap
- that routine must be told what number to use for each dist
- that routine must be told about loop indices, so it can work inside of vmap

Maybe we should have a Reference object.

Rules:
- If it's a normal dist, just return the code, easy.
- If it's a VMapDist, create a loop, insert the loop variable name into the references
for each RV and recurse
- If it's an Index, create as many nested loops as the output should have, loop over
everything

"""


from . import interface, dag, util, inference
from . import ezjags
import random
from pangolin.inference_jags_stan_shared import Helper, Reference


# get shared JAGS / Stan function customized for JAGS syntax
helper = Helper("JAGS", "<-", "")
indent = helper.indent
gencode_infix_factory = helper.gencode_infix_factory
gencode_deterministic_factory = helper.gencode_deterministic_factory
slice_to_str = helper.slice_to_str
gencode_index = helper.gencode_index
gencode_dist_factory = helper.gencode_dist_factory
# gencode_dist_factory_swapargs = helper.gencode_dist_factory_swapargs
gencode_sum = helper.gencode_sum
gencode_categorical_factory = helper.gencode_categorical_factory
gencode_vmapdist_factory = helper.gencode_vmapdist_factory


def gencode(cond_dist, loopdepth, id, *parent_ids):
    if cond_dist in gencode_fns:
        gencode_fn = gencode_fns[cond_dist]
    elif type(cond_dist) in class_gencode_fns:
        gencode_fn = class_gencode_fns[type(cond_dist)]
    else:
        raise NotImplementedError(f"cond dist {cond_dist} not implemented")
    return gencode_fn(cond_dist, loopdepth, id, *parent_ids)


def gencode_bernoulli_logit(cond_dist, loopdepth, ref, *parent_refs):
    return f"{ref} ~ dbern(ilogit({parent_refs[0]}));\n"


def gencode_normal_scale(cond_dist, loopdepth, ref, *parent_refs):
    assert cond_dist == interface.normal_scale
    return f"{ref} ~ dnorm({parent_refs[0]},1/({parent_refs[1]})^2);\n"


# # not easy to support softmax because JAGS doesn't allow exp to be vectorized
# def gencode_softmax(cond_dist, loopdepth, ref, *parent_refs):
#     p_ref = parent_refs[0]
#     N = p_ref.shape[0]
#     return f"{ref} <- exp({p_ref}[1:{N}])/sum(exp({p_ref}[1:{N}]));\n"


# def gencode_categorical(cond_dist, loopdepth, ref, *parent_refs):
#     """
#     special code needed since JAGS is 1-indexed
#     """
#     assert cond_dist == interface.categorical
#     assert len(parent_refs) == 1
#     code1 = f"tmp_{ref} ~ dcat({parent_refs[0]})\n"
#     code2 = f"{ref} <- tmp_{ref}-1\n"
#     return code1 + code2


# def gencode_vmapdist(cond_dist, loopdepth, ref, *parent_refs):
#     # print(f"{cond_dist=}")
#
#     loop_index = f"i{loopdepth}"
#
#     new_ref = ref.index(0, loop_index)
#     new_parent_refs = [
#         p_ref.index(axis, loop_index)
#         for p_ref, axis in zip(parent_refs, cond_dist.in_axes)
#     ]
#
#     loop_code = f"for ({loop_index} in 1:" + str(cond_dist.axis_size) + "){\n"
#     middle_code = gencode(
#         cond_dist.base_cond_dist, loopdepth + 1, new_ref, *new_parent_refs
#     )
#     end_code = "}\n"
#
#     middle_code = indent(middle_code, 1)
#
#     code = loop_code + middle_code + end_code
#
#     return code


# def gencode_dist_factory(name):
#     def gencode_dist(cond_dist, loopdepth, ref, *parent_refs):
#         return f"{ref} ~ {name}" + util.comma_separated(parent_refs, str) + "\n"
#
#     return gencode_dist
#
#
# def gencode_dist_factory_swapargs(name):
#     def gencode_dist(cond_dist, loopdepth, ref, *parent_refs):
#         assert len(parent_refs) == 2
#         new_parent_refs = (parent_refs[1], parent_refs[0])
#         return f"{ref} ~ {name}" + util.comma_separated(new_parent_refs, str) + "\n"
#
#     return gencode_dist
#
#
# def gencode_sum(cond_dist, loopdepth, ref, parent_ref):
#     assert isinstance(cond_dist, interface.Sum)
#     axis = cond_dist.axis
#     loop_code = ""
#     end_code = ""
#     for n in range(ref.ndim):
#         if n == axis:
#             parent_ref = parent_ref.index(0, "")  # empty string - get all
#         else:
#             loop_index = f"l{loopdepth}"
#             open_axis = ref.nth_open_axis(0)
#             loop_code += f"for ({loop_index} in 1:{ref.shape[open_axis]})" + "{" "\n"
#             end_code = "}\n" + end_code
#             loopdepth += 1
#             ref = ref.index(0, loop_index)
#             parent_ref = parent_ref.index(0, loop_index)
#     middle_code = f"{ref} <- sum({parent_ref});\n"
#     code = loop_code + middle_code + end_code
#     return code


def gencode_unsupported():
    def gencode_dist(cond_dist, loopdepth, ref, *parent_refs):
        raise NotImplementedError(f"JAGS does not support distribution {cond_dist}")

    return gencode_dist


gencode_fns = {
    interface.normal_scale: gencode_normal_scale,
    interface.normal_prec: gencode_dist_factory("dnorm"),
    interface.bernoulli: gencode_dist_factory("dbern"),
    interface.bernoulli_logit: gencode_bernoulli_logit,
    # interface.binomial: gencode_dist_factory("dbin"),
    interface.binomial: gencode_dist_factory("dbin", [1, 0]),
    interface.uniform: gencode_dist_factory("dunif"),
    interface.beta: gencode_dist_factory("dbeta"),
    interface.exponential: gencode_dist_factory("dexp"),
    interface.dirichlet: gencode_dist_factory("ddirch"),
    # interface.categorical: gencode_dist_factory("dcat"),
    interface.categorical: gencode_categorical_factory("dcat"),
    interface.multinomial: gencode_dist_factory("dmulti", [1, 0]),
    interface.multi_normal_cov: gencode_dist_factory("mnorm.vcov"),
    interface.beta_binomial: gencode_unsupported(),
    interface.mul: gencode_infix_factory("*"),
    interface.add: gencode_infix_factory("+"),
    interface.sub: gencode_infix_factory("-"),
    interface.div: gencode_infix_factory("/"),
    interface.pow: gencode_infix_factory("^"),
    interface.matmul: gencode_infix_factory("%*%"),  # TODO: only support 1 or 2d args
    interface.inv: gencode_deterministic_factory("inverse"),
    interface.abs: gencode_deterministic_factory("abs"),
    interface.arccos: gencode_deterministic_factory("arccos"),
    interface.arccosh: gencode_deterministic_factory("arccosh"),
    interface.arcsin: gencode_deterministic_factory("arcsin"),
    interface.arcsinh: gencode_deterministic_factory("arcsinh"),
    interface.arctan: gencode_deterministic_factory("arctan"),
    interface.arctanh: gencode_deterministic_factory("arctanh"),
    interface.cos: gencode_deterministic_factory("cos"),
    interface.cosh: gencode_deterministic_factory("cosh"),
    interface.exp: gencode_deterministic_factory("exp"),
    interface.inv_logit: gencode_deterministic_factory("ilogit"),
    interface.log: gencode_deterministic_factory("log"),
    interface.loggamma: gencode_deterministic_factory("loggam"),
    interface.logit: gencode_deterministic_factory("logit"),
    interface.sin: gencode_deterministic_factory("sin"),
    interface.sinh: gencode_deterministic_factory("sinh"),
    interface.step: gencode_deterministic_factory("step"),
    interface.tan: gencode_deterministic_factory("tan"),
    interface.tanh: gencode_deterministic_factory("tanh"),
    interface.softmax: gencode_unsupported(),
}


class_gencode_fns = {
    interface.VMapDist: gencode_vmapdist_factory(gencode),
    interface.Index: gencode_index,
    # interface.Sum: gencode_unsupported(),
    interface.Sum: gencode_sum,
    # interface.CondProb: gencode_unsupported(),
    # interface.Mixture: gencode_unsupported(),
}


for cond_dist in interface.all_cond_dists:
    assert cond_dist in gencode_fns, f"{cond_dist} gencode not found"


for cond_dist_class in interface.all_cond_dist_classes:
    assert cond_dist_class in class_gencode_fns, f"{cond_dist_class} gencode not found"


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

    assert len(given_vars) == len(given_vals)
    for var, val in zip(given_vars, given_vals):
        assert var.shape == val.shape

    # this variable splitting business is a bit of a mess (although seemingly correct)
    random_vars = inference.upstream_with_descendent(requested_vars, given_vars)
    latent_vars = [node for node in random_vars if node not in given_vars]

    included_vars = dag.upstream_nodes(requested_vars + given_vars)

    evidence = util.WriteOnceDict()

    n = 0
    ids = {}
    for var in dag.upstream_nodes(requested_vars + given_vars):
        ids[var] = f"v{n}"
        n += 1

    # conditioning variables
    for var, val in zip(given_vars, given_vals):
        evidence[ids[var]] = val

    # declare all variable with shapes
    code = "var "
    for var in included_vars:
        code += ids[var]
        if var.shape != ():
            code += "[" + util.comma_separated(var.shape, str, parens=False) + "]"
        if var != included_vars[-1]:
            code += ", "
    code += ";\n"

    code += "model{\n"
    for var in included_vars:
        if isinstance(var.cond_dist, interface.Constant):
            evidence[ids[var]] = var.cond_dist.value  # constant RVs
        else:
            ref = Reference(ids[var], var.shape)
            parent_refs = [Reference(ids[p], p.shape) for p in var.parents]
            cond_dist = var.cond_dist
            code += gencode(cond_dist, 0, ref, *parent_refs)  # others

    code += "}\n"

    monitor_vars = [ids[var] for var in requested_vars]

    results = ezjags.jags(code, monitor_vars, niter=niter, nchains=1, **evidence)

    return results


def jags_code(vars):
    import jax.tree_util

    flat_vars, vars_treedef = jax.tree_util.tree_flatten(vars)
    n = 0
    ids = {}
    for var in dag.upstream_nodes(flat_vars):
        ids[var] = f"v{n}"
        n += 1

    # declare all variable with shapes
    code = "var "
    for var in flat_vars:
        code += ids[var]
        if var.shape != ():
            code += "[" + util.comma_separated(var.shape, str, parens=False) + "]"
        if var != flat_vars[-1]:
            code += ", "
    code += ";\n"

    code += "model{\n"
    for var in flat_vars:
        if isinstance(var.cond_dist, interface.Constant):
            pass
        else:
            ref = Reference(ids[var], var.shape)
            parent_refs = [Reference(ids[p], p.shape) for p in var.parents]
            cond_dist = var.cond_dist
            code += gencode(cond_dist, 0, ref, *parent_refs)  # others

    code += "}\n"
    return code
