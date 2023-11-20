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


def gencode_matmul(cond_dist, loopdepth, ref, *parent_refs):
    assert len(parent_refs) == 2
    a = parent_refs[0]
    b = parent_refs[1]
    assert a.num_empty >= 1
    assert a.num_empty <= 2
    assert b.num_empty >= 1
    assert b.num_empty <= 2

    if a.num_empty == 1 and b.num_empty == 1:
        return f"{ref} = (to_row_vector({a}) * to_vector({b}));"
    elif a.num_empty == 1 and b.num_empty == 2:
        return f"{ref} = to_vector(to_row_vector({a}) * to_matrix({b}));"
    elif a.num_empty == 2 and b.num_empty == 1:
        return f"{ref} = (to_matrix({a}) * to_vector({b}));"
    elif a.num_empty == 2 and b.num_empty == 2:
        return f"{ref} = (to_matrix({a}) * to_matrix({b}));"
    else:
        raise NotImplementedError("should be impossible...")


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
    interface.matmul: gencode_matmul,
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
    def __init__(self, base_type, shape=None, lower=None, upper=None):
        self.base_type = base_type
        self.shape = shape
        self.lower = lower
        self.upper = upper

    def declare(self, varname):
        if self.base_type == "int":
            if self.shape != ():
                s = "array[" + util.comma_separated(self.shape, str, False) + "] "
            else:
                s = ""

            s += "int"

            if self.lower and self.upper:
                s += f"<lower={self.lower},upper={self.upper}>"
            elif self.lower:
                s += f"<lower={self.lower}>"
            elif self.upper:
                s += f"<upper={self.upper}>"

            return s + " " + varname + ";"

        elif self.base_type == "real":
            batch_shape = self.shape[:-2]
            event_shape = self.shape[-2:]

            if batch_shape != ():
                s = "array[" + util.comma_separated(batch_shape, str, False) + "] "
            else:
                s = ""

            if event_shape == ():
                s += "real"
            elif len(event_shape) == 1:
                s += "vector"
            elif len(event_shape) == 2:
                s += "matrix"
            else:
                assert False, "should be impossible"

            if self.lower and self.upper:
                s += f"<lower={self.lower},upper={self.upper}>"
            elif self.lower:
                s += f"<lower={self.lower}>"
            elif self.upper:
                s += f"<upper={self.upper}>"

            if event_shape != ():
                s += "[" + util.comma_separated(event_shape, str, False) + "]"

            return s + " " + varname + ";"

        elif self.base_type == "simplex":
            batch_shape = self.shape[:-1]
            event_shape = self.shape[-1:]

            if batch_shape != ():
                s = "array[" + util.comma_separated(batch_shape, str, False) + "] "
            else:
                s = ""

            assert len(event_shape) == 1

            s += f"simplex[{event_shape[0]}]"

            assert self.lower is None, "simplex should not have bounds"
            assert self.upper is None, "simplex should not have bounds"

            return s + " " + varname + ";"

        else:
            raise NotImplementedError(f"type {self.base_type} not implemented")


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


def base_type(cond_dist, *parent_types):
    """
    get the type of a cond_dist without worrying about shape
    """
    if cond_dist in [
        interface.add,
        interface.mul,
        interface.add,
        interface.sub,
        interface.mul,
        interface.pow,
        interface.abs,
        interface.matmul,
    ]:
        if all(t.base_type == "int" for t in parent_types):
            return StanType("int")
        else:
            return StanType("real")
    elif cond_dist in [interface.div, interface.exp]:
        return StanType("real")
    elif cond_dist in (interface.normal_scale, interface.uniform):
        return StanType("real")
    elif cond_dist in (interface.beta,):
        return StanType("real", lower=0, upper=1)
    elif cond_dist in (interface.bernoulli,):
        return StanType("int", lower=0, upper=1)
    elif isinstance(cond_dist, interface.Constant):
        if np.issubdtype(cond_dist.value.dtype, np.floating):
            return StanType("real")
        elif np.issubdtype(cond_dist.value.dtype, np.integer):
            return StanType("int")
        else:
            raise NotImplementedError("Array neither float nor integer type")
    elif cond_dist in (interface.multinomial, interface.binomial):
        return StanType("int")
    elif cond_dist in (interface.dirichlet,):
        return StanType("simplex")
    elif isinstance(cond_dist, interface.Index):
        # TODO: need to check if last dimension is being indexed
        t = parent_types[0]
        assert t.base_type != "simplex", "don't handle this case yet"
        return StanType(t.base_type, lower=t.lower, upper=t.upper)
    elif isinstance(cond_dist, interface.Sum):
        return StanType("real")
    elif isinstance(cond_dist, interface.VMapDist):
        return base_type(cond_dist.base_cond_dist)
    else:
        raise NotImplementedError(f"type string not implemented for {cond_dist}")


def stan_type(var, *parent_types):
    t = base_type(var.cond_dist, *parent_types)
    return StanType(t.base_type, var.shape, t.lower, t.upper)


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

        print(f"{var=}")
        if var in given_vars or isinstance(var.cond_dist, pangolin.Constant):
            # transformed DATA can be int
            mycode = types[var].declare(ids[var]) + "\n"
            data_code += mycode
        elif var.cond_dist.is_random:
            mycode = types[var].declare(ids[var]) + "\n"
            parameters_code += mycode
        else:
            # transformed PARAMS can't be
            if types[var].base_type == "int":
                types[var].base_type = "real"
            mycode = types[var].declare(ids[var]) + "\n"
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
