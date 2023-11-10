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
import pangolin
from . import interface, dag, util, inference
from . import ezstan
import textwrap
import numpy as np

import random


def indent(code, n):
    return textwrap.indent(code, "    " * n)


# def gencode_normal_scale(cond_dist, loopdepth, ref, *parent_refs):
#     assert cond_dist == interface.normal_scale
#     return f"{ref} ~ dnorm({parent_refs[0]},1/({parent_refs[1]})^2)\n"


def gencode_categorical(cond_dist, loopdepth, ref, *parent_refs):
    """
    special code needed since JAGS is 1-indexed
    """
    assert cond_dist == interface.categorical
    assert len(parent_refs) == 1
    code1 = f"tmp_{ref} ~ dcat({parent_refs[0]})\n"
    code2 = f"{ref} <- tmp_{ref}-1\n"
    return code1 + code2


# def gencode_mul(cond_dist, loopdepth, ref, *parent_refs):
#     assert cond_dist == interface.mul
#     return f"{ref} <- ({parent_refs[0]}) * ({parent_refs[1]})\n"


def gencode_infix_factory(infix_str):
    def gencode_infix(cond_dist, loopdepth, ref, *parent_refs):
        return f"{ref} <- ({parent_refs[0]}) {infix_str} ({parent_refs[1]})\n"

    return gencode_infix


def gencode_deterministic_factory(fun_str):
    def gencode_deterministic(cond_dist, loopdepth, ref, *parent_refs):
        return f"{ref} <- {fun_str}{util.comma_separated(parent_refs)}\n"

    return gencode_deterministic


def gencode_vmapdist(cond_dist, loopdepth, ref, *parent_refs):
    # print(f"{cond_dist=}")

    loop_index = f"i{loopdepth}"

    new_ref = ref.index(0, loop_index)
    new_parent_refs = [
        p_ref.index(axis, loop_index)
        for p_ref, axis in zip(parent_refs, cond_dist.in_axes)
    ]

    loop_code = f"for ({loop_index} in 1:" + str(cond_dist.axis_size) + "){\n"
    middle_code = gencode(
        cond_dist.base_cond_dist, loopdepth + 1, new_ref, *new_parent_refs
    )
    end_code = "}\n"

    middle_code = indent(middle_code, 1)

    code = loop_code + middle_code + end_code

    return code


def slice_to_str(my_slice, ref):
    start = my_slice.start
    stop = my_slice.stop
    step = my_slice.step
    if start is None:
        start = 0
    if stop is None:
        # use next unused axis
        axis = ref.nth_open_axis(0)
        stop = ref.shape[axis]
    if step:
        raise NotImplementedError("JAGS doesn't support step in slices :(")

    # JAGS 1-indexed and inclusive so start increase but not stop
    loop_index_str = f"{start + 1}:{stop}"
    loop_shape_str = f"1:{stop - start}"
    return loop_index_str, loop_shape_str


def num_index_dims(index_refs):
    """
    get the number of dimensions in the index refs, or None if empty
    """
    # assert all dimensions same
    for index_ref1 in index_refs:
        for index_ref2 in index_refs:
            assert (
                index_ref1.shape == index_ref2.shape
            ), "all indices must have same dimensions"
    if index_refs:
        return index_refs[0].ndim
    else:
        return None


def check_gencode_index_inputs(cond_dist, ref, parent_ref, *index_refs):
    """
    1. that the number of empty slots in `parent_ref` is the number of slots in
    `cond_dist` (each can be either a slice of a scalar or an array)
    2. that the number of empty slots in `ref` is equal to the number of *slice*s in
    `cond_dist` plus the number of dims on index (if it exists)
    """
    expected_parent_empty_slots = len(cond_dist.slices)
    assert parent_ref.num_empty == expected_parent_empty_slots

    num_actual_slices = sum(1 for my_slice in cond_dist.slices if my_slice)
    if index_refs:
        index_dims = index_refs[0].ndim
        assert ref.num_empty == num_actual_slices + index_dims


def gencode_index(cond_dist, loopdepth, ref, parent_ref, *index_refs):
    check_gencode_index_inputs(cond_dist, ref, parent_ref, *index_refs)

    # currently can only index with 1d or 2d arrays

    loop_code = ""
    end_code = ""

    index_ndim = num_index_dims(index_refs)

    ref_loop_index_needed = True
    idx_loop_indices = []
    if index_ndim is not None:
        first_shape = index_refs[0].shape

        # idx_loop_indices = []
        for n in range(index_ndim):
            idx_loop_index = f"l{loopdepth}"
            loop_code += f"for ({idx_loop_index} in 1:{first_shape[n]})" + "{" "\n"
            end_code = "}\n" + end_code
            idx_loop_indices.append(idx_loop_index)
            loopdepth += 1

        # add loop indices for LHS if should go at start
        if cond_dist.advanced_at_start:
            ref = ref.index_mult(idx_loop_indices)
            ref_loop_index_needed = False

    index_refs_iter = iter(index_refs)
    # go through all slots in cond_dist (Index object)
    for my_slice in cond_dist.slices:
        if my_slice:
            loop_index_str, loop_shape_str = slice_to_str(my_slice, parent_ref)
            parent_ref = parent_ref.index(0, loop_index_str)
            ref = ref.index(0, loop_shape_str)
        else:
            # grab next index parent and add index loop indices
            my_index_ref = next(index_refs_iter).index_mult(idx_loop_indices)

            # add loop indices for LHS if should go here
            if ref_loop_index_needed:
                ref = ref.index_mult(idx_loop_indices)
                ref_loop_index_needed = False

            parent_loop_index = "1+" + str(my_index_ref)  # JAGS 1 indexed
            parent_ref = parent_ref.index(0, parent_loop_index)

    middle_code = str(ref) + " <- " + str(parent_ref) + "\n"
    middle_code = middle_code
    code = loop_code + middle_code + end_code
    return code


def gencode_dist_factory(name):
    def gencode_dist(cond_dist, loopdepth, ref, *parent_refs):
        return f"{ref} ~ {name}" + util.comma_separated(parent_refs, str) + ";\n"

    return gencode_dist


def gencode_dist_factory_swapargs(name):
    def gencode_dist(cond_dist, loopdepth, ref, *parent_refs):
        assert len(parent_refs) == 2
        new_parent_refs = (parent_refs[1], parent_refs[0])
        return f"{ref} ~ {name}" + util.comma_separated(new_parent_refs, str) + "\n"

    return gencode_dist


def gencode_unsupported():
    def gencode_dist(cond_dist, loopdepth, ref, *parent_refs):
        raise NotImplementedError(f"Stan does not support distribution {cond_dist}")

    return gencode_dist


gencode_fns = {
    interface.normal_scale: gencode_dist_factory("normal"),
    interface.normal_prec: gencode_unsupported(),
    interface.uniform: gencode_dist_factory("uniform"),
    interface.bernoulli: gencode_dist_factory("bernoulli"),
    # # interface.binomial: gencode_dist_factory("dbin"),
    # interface.binomial: gencode_dist_factory_swapargs("dbin"),
    # interface.beta: gencode_dist_factory("dbeta"),
    # interface.exponential: gencode_dist_factory("dexp"),
    # interface.dirichlet: gencode_dist_factory("ddirch"),
    # # interface.categorical: gencode_dist_factory("dcat"),
    # interface.categorical: gencode_categorical,
    # interface.multinomial: gencode_dist_factory_swapargs("dmulti"),
    # interface.multi_normal_cov: gencode_dist_factory("mnorm.vcov"),
    # interface.beta_binomial: gencode_unsupported(),
    # interface.mul: gencode_infix_factory("*"),
    # interface.add: gencode_infix_factory("+"),
    # interface.sub: gencode_infix_factory("-"),
    # interface.div: gencode_infix_factory("/"),
    # interface.pow: gencode_infix_factory("^"),
    # interface.matmul: gencode_infix_factory("%*%"),
    # interface.abs: gencode_deterministic_factory("abs"),
    # interface.exp: gencode_deterministic_factory("exp"),
}


class_gencode_fns = {
    interface.VMapDist: gencode_vmapdist,
    # interface.Index: gencode_index,
    # interface.Sum: gencode_unsupported(),
    # interface.CondProb: gencode_unsupported(),
    # interface.Mixture: gencode_unsupported(),
}


def gencode(cond_dist, loopdepth, id, *parent_ids):
    if cond_dist in gencode_fns:
        gencode_fn = gencode_fns[cond_dist]
    elif type(cond_dist) in class_gencode_fns:
        gencode_fn = class_gencode_fns[type(cond_dist)]
    else:
        raise NotImplementedError(f"cond dist {cond_dist} not implemented")
    return gencode_fn(cond_dist, loopdepth, id, *parent_ids)


# for cond_dist in interface.all_cond_dists:
#     assert cond_dist in gencode_fns, f"{cond_dist} gencode not found"
#
#
# for cond_dist_class in interface.all_cond_dist_classes:
#     assert cond_dist_class in class_gencode_fns, f"{cond_dist_class} gencode not found"


class Reference:
    def __init__(self, id, shape, loop_indices=None):
        if loop_indices is None:
            loop_indices = [None] * len(shape)
        self.id = id
        self.shape = shape
        self.loop_indices = loop_indices

    def __str__(self):
        if all(i is None for i in self.loop_indices):
            return self.id
        ret = self.id + "["
        for n, i in enumerate(self.loop_indices):
            if i:
                ret += i
            if n < len(self.loop_indices) - 1:
                ret += ","
        ret += "]"
        return ret

    def nth_open_axis(self, axis):
        return util.nth_index(self.loop_indices, None, axis)

    def index(self, axis, i):
        if axis is None:
            return self

        # i should go in the axis-th empty slot
        where = self.nth_open_axis(axis)

        new_loop_indices = self.loop_indices.copy()
        new_loop_indices[where] = i
        return Reference(self.id, self.shape, new_loop_indices)

    def index_mult(self, indices):
        rez = self
        for i in indices:
            rez = rez.index(0, i)
        return rez

    # def index_shifted(self, axis, i, where_to_place):
    #     tmp_indices = self.index(axis, i).loop_indices
    #     where = self.nth_open_axis(axis)
    #     new_loop_indices = util.swapped_list(tmp_indices, where, where_to_place)
    #     return Reference(self.id, self.shape, new_loop_indices)

    @property
    def num_empty(self):
        return sum(x is None for x in self.loop_indices)

    @property
    def ndim(self):
        return len(self.shape)


def stan_type_string(cond_dist):
    if isinstance(cond_dist, interface.VMapDist):
        return stan_type_string(cond_dist.base_cond_dist)
    elif isinstance(cond_dist, interface.Constant):
        if np.issubdtype(cond_dist.value.dtype, np.floating):
            return "real"
        elif np.issubdtype(cond_dist.value.dtype, np.integer):
            # return "int"
            return "real"
        else:
            raise NotImplementedError("Array neither float nor integer type")
    elif cond_dist in [interface.normal_scale, interface.uniform]:
        return "real"
    elif cond_dist in [interface.bernoulli]:
        return "int<lower=0,upper=1>"
    else:
        raise NotImplementedError()


def stan_code_flat(requested_vars, given_vars, given_vals):
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

    parameters_code = "parameters{\n"
    data_code = "data{\n"
    for var in included_vars:
        # mycode = "real " + ids[var]
        mycode = stan_type_string(var.cond_dist)
        mycode += " " + ids[var]
        if var.shape != ():
            mycode += "[" + util.comma_separated(var.shape, str, parens=False) + "]"
        mycode += ";\n"
        print(f"{var=}")
        if var in given_vars or isinstance(var.cond_dist, pangolin.Constant):
            data_code += mycode
        else:
            parameters_code += mycode
    parameters_code += "}\n"
    data_code += "}\n"
    code = data_code + parameters_code

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
