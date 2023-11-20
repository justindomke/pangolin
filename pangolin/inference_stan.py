import pangolin
from . import interface, dag, util, inference
from . import ezstan
import textwrap
import numpy as np
from pangolin import inference_jags_stan_shared

"""
Programmatically generate and run Stan code.
"""

# Improvements to be done someday
# - Find branches without observed descendents, place in generated quantities block
# - Put stuff in transformed data block when possible

# from pangolin.inference_jags_stan_shared import indent, gencode_infix_factory


from pangolin.inference_jags_stan_shared import Helper, Reference

# get shared JAGS / Stan function customized for JAGS syntax
helper = Helper("Stan", "=", ":")
indent = helper.indent
gencode_infix_factory = helper.gencode_infix_factory
gencode_deterministic_factory = helper.gencode_deterministic_factory
slice_to_str = helper.slice_to_str
gencode_index = helper.gencode_index
gencode_dist_factory = helper.gencode_dist_factory
gencode_sum = helper.gencode_sum
gencode_categorical_factory = helper.gencode_categorical_factory
gencode_vmapdist_factory = helper.gencode_vmapdist_factory

# def gencode_vmapdist(cond_dist, loopdepth, ref, *parent_refs):
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


def gencode(cond_dist, loopdepth, id, *parent_ids):
    if cond_dist in gencode_fns:
        gencode_fn = gencode_fns[cond_dist]
    elif type(cond_dist) in class_gencode_fns:
        gencode_fn = class_gencode_fns[type(cond_dist)]
    else:
        raise NotImplementedError(f"cond dist {cond_dist} not implemented")
    return gencode_fn(cond_dist, loopdepth, id, *parent_ids)


def gencode_unsupported():
    def gencode_dist(cond_dist, loopdepth, ref, *parent_refs):
        raise NotImplementedError(f"Stan does not support distribution {cond_dist}")

    return gencode_dist


gencode_fns = {
    interface.normal_scale: gencode_dist_factory("normal"),
    interface.normal_prec: gencode_unsupported(),
    interface.uniform: gencode_dist_factory("uniform"),
    interface.bernoulli: gencode_dist_factory("bernoulli"),
    interface.binomial: gencode_dist_factory("binomial"),
    # interface.binomial: gencode_dist_factory_swapargs("dbin"),
    interface.beta: gencode_dist_factory("beta"),
    # interface.exponential: gencode_dist_factory("dexp"),
    interface.dirichlet: gencode_dist_factory("dirichlet"),
    # # interface.categorical: gencode_dist_factory("dcat"),
    # interface.categorical: gencode_categorical,
    # interface.multinomial: gencode_dist_factory_swapargs("dmulti"),
    interface.multinomial: gencode_dist_factory("multinomial", [1]),  # drop num trials
    # interface.multi_normal_cov: gencode_dist_factory("mnorm.vcov"),
    # interface.beta_binomial: gencode_unsupported(),
    interface.mul: gencode_infix_factory("*"),
    interface.add: gencode_infix_factory("+"),
    interface.sub: gencode_infix_factory("-"),
    interface.div: gencode_infix_factory("/"),
    interface.pow: gencode_infix_factory("^"),
    interface.matmul: gencode_unsupported(),  # stan needs specific matrix/vector types
    interface.abs: gencode_deterministic_factory("abs"),
    interface.exp: gencode_deterministic_factory("exp"),
}


class_gencode_fns = {
    interface.VMapDist: gencode_vmapdist_factory(gencode),
    interface.Index: gencode_index,
    interface.Sum: gencode_sum,
    interface.CondProb: gencode_unsupported(),
    interface.Mixture: gencode_unsupported(),
}


# for cond_dist in interface.all_cond_dists:
#     assert cond_dist in gencode_fns, f"{cond_dist} gencode not found"
#
#
# for cond_dist_class in interface.all_cond_dist_classes:
#     assert cond_dist_class in class_gencode_fns, f"{cond_dist_class} gencode not found"


class StanType:
    def __init__(self, base_type, event_shape=None, batch_shape=None):
        self.base_type = base_type
        self.event_shape = event_shape
        self.batch_shape = batch_shape

    def declare(self, varname):
        if self.batch_shape is not None and self.batch_shape != ():
            s = "array[" + util.comma_separated(self.batch_shape, str, False) + "] "
        else:
            s = ""
        s += self.base_type
        if self.event_shape is not None and self.event_shape != ():
            s += "[" + util.comma_separated(self.event_shape, str, False) + "]"
        return s + " " + varname + ";"


# One theory for how to do Stan declarations:
# 1. variables have these types
# - real
# - int
# - simplex
# 2. they can also have upper or lower bounds
# 3. when getting final type outputs, reals are converted to as much vector/matrix as
# possible

# now, the major way things could go wrong is this:
# - some distribution expects `vector` inputs
# - the vector inputs are given by vmapping over the non-last dims


def stan_type(var, *parent_types):
    if var.cond_dist in (interface.normal_scale, interface.uniform):
        return StanType("real")
    elif var.cond_dist in (interface.beta,):
        return StanType("real<lower=0,upper=1>")
    elif var.cond_dist in (interface.bernoulli,):
        return StanType("int<lower=0,upper=1>")
    elif isinstance(var.cond_dist, interface.Constant):
        if np.issubdtype(var.cond_dist.value.dtype, np.floating):
            # if you can, declare as a vector
            # should this be done in the final stage?
            # should we declare a matrix if we can?
            if var.ndim == 0:
                return StanType("real", var.shape, None)
            elif var.ndim == 1:
                return StanType("vector", var.shape, None)
            elif var.ndim == 2:
                return StanType("matrix", var.shape, None)
            else:
                return StanType("matrix", var.shape[-2:], var.shape[:-2])
        elif np.issubdtype(var.cond_dist.value.dtype, np.integer):
            return StanType("int", None, var.shape)
        else:
            raise NotImplementedError("Array neither float nor integer type")
    elif var.cond_dist in (interface.dirichlet,):
        return StanType("simplex", var.shape)
    elif var.cond_dist in (interface.multinomial,):
        return StanType("int", None, var.shape)
    elif isinstance(var.cond_dist, interface.Index):
        return StanType(parent_types[0])

    else:
        raise NotImplementedError(f"type string not implemented for {var}")


# def stan_type_string(cond_dist, *parent_type_strings):
#     if isinstance(cond_dist, interface.VMapDist):
#         # return stan_type_string(cond_dist.base_cond_dist)
#         # tmp_var = interface.RV(cond_dist.base_cond_dist, *var.parents)
#         # return stan_type_string(tmp_var)
#         return stan_type_string(cond_dist.base_cond_dist, *parent_type_strings)
#     elif isinstance(cond_dist, interface.Constant):
#         if np.issubdtype(cond_dist.value.dtype, np.floating):
#             return "real"
#         elif np.issubdtype(cond_dist.value.dtype, np.integer):
#             return "int"
#             # return "real"  # TODO FIX
#         else:
#             raise NotImplementedError("Array neither float nor integer type")
#     elif isinstance(cond_dist, (interface.Index, interface.Sum)):
#         return parent_type_strings[0]
#     elif cond_dist in [interface.normal_scale, interface.uniform, interface.dirichlet]:
#         return "real"
#     elif cond_dist in [interface.bernoulli]:
#         return "int<lower=0,upper=1>"
#     elif cond_dist in [interface.beta]:
#         return "real<lower=0,upper=1>"
#     elif cond_dist in [interface.binomial, interface.multinomial]:
#         return "int"
#     elif cond_dist in [
#         interface.add,
#         interface.mul,
#         interface.add,
#         interface.sub,
#         interface.mul,
#         interface.div,
#         interface.pow,
#         interface.abs,
#         interface.exp,
#         interface.matmul,
#     ]:
#         # assert len(parent_type_strings) == 2
#         if all(type_string == "int" for type_string in parent_type_strings):
#             return "int"
#         else:
#             return "real"
#     else:
#         raise NotImplementedError(f"type string not implemented for {cond_dist}")


def stan_code_flat(requested_vars, given_vars, given_vals):
    included_vars = dag.upstream_nodes(requested_vars + given_vars)

    evidence = util.WriteOnceDict()

    n = 0
    ids = {}
    for var in dag.upstream_nodes(requested_vars + given_vars):
        ids[var] = f"v{n}"
        n += 1

    parameters_code = "parameters{\n"
    data_code = "data{\n"
    # TODO: add transformed_data_code for extra efficiency
    model_code = "model{\n"
    transformed_parameters_code = "transformed parameters{\n"

    # conditioning variables
    for var, val in zip(given_vars, given_vals):
        evidence[ids[var]] = val

    # type_strings = {}
    types = {}
    for var in included_vars:
        # mycode = stan_type_string(var)
        # parent_type_strings = [type_strings[p] for p in var.parents]
        # type_strings[var] = stan_type_string(var.cond_dist, *parent_type_strings)
        parent_types = [types[p] for p in var.parents]
        types[var] = stan_type(var, *parent_types)

        mycode = types[var].declare(ids[var]) + "\n"

        # mycode = type_strings[var]
        # mycode += " " + ids[var]
        # if var.shape != ():
        #     mycode += "[" + util.comma_separated(var.shape, str, parens=False) + "]"
        # mycode += ";\n"

        # if var.ndim == 0:
        #     mycode = type_strings[var] + " " + ids[var] + ";\n"
        # elif type_strings[var] == "real" and var.ndim == 1:
        #     mycode = "vector[" + str(var.shape[0]) + "] " + ids[var] + ";\n"
        # else:
        #     mycode = type_strings[var]
        #     mycode += " " + ids[var]
        #     if var.shape != ():
        #         mycode += "[" + util.comma_separated(var.shape, str, parens=False) + "]"
        #     mycode += ";\n"

        print(f"{var=}")
        if var in given_vars or isinstance(var.cond_dist, pangolin.Constant):
            data_code += mycode
        elif var.cond_dist.is_random:
            parameters_code += mycode
        else:
            transformed_parameters_code += mycode

    for var in included_vars:
        if isinstance(var.cond_dist, interface.Constant):  # constant RVs
            evidence[ids[var]] = var.cond_dist.value
        else:  # others
            ref = Reference(ids[var], var.shape)
            parent_refs = [Reference(ids[p], p.shape) for p in var.parents]
            cond_dist = var.cond_dist
            mycode = gencode(cond_dist, 0, ref, *parent_refs)
            if cond_dist.is_random:
                model_code += mycode
            else:
                transformed_parameters_code += mycode

    parameters_code += "}\n"
    data_code += "}\n"
    model_code += "}\n"
    transformed_parameters_code += "}\n"

    code = data_code + parameters_code + transformed_parameters_code + model_code

    monitor_vars = [ids[var] for var in requested_vars]

    return code, monitor_vars, evidence


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

    code, monitor_vars, evidence = stan_code_flat(
        requested_vars, given_vars, given_vals
    )

    print("CODE")
    print(code)

    print("EVIDENCE")
    print(evidence)

    results = ezstan.stan(code, monitor_vars, niter=niter, nchains=1, **evidence)

    return results
