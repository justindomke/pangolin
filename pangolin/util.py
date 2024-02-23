from jax import numpy as jnp


def comma_separated(stuff, fun=None, parens=True):
    "convenience function for printing and such"
    ret = ""
    if parens:
        ret += "("
    for i, thing in enumerate(stuff):
        if fun is None:
            ret += thing  # should this be str(thing)?
        else:
            ret += fun(thing)
        if i < len(stuff) - 1:
            ret += ","
    if parens:
        ret += ")"
    return ret


class VarNames:
    def __init__(self):
        self.num = 0
        self.node_to_name = {}

    def __getitem__(self, node):
        if node in self.node_to_name:
            varname = self.node_to_name[node]
        else:
            varname = f"v{self.num}v"
            self.num += 1
            self.node_to_name[node] = varname
        return varname


def one_not_none(*args):
    num = 0
    for arg in args:
        if arg is not None:
            num += 1
    return num


def all_unique(lst):
    return len(lst) == len(set(lst))


class WriteOnceDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(f"Cannot overwrite existing key {key} in WriteOnceDict")
        super().__setitem__(key, value)


def is_leaf_with_none(xi):
    return (xi is None) or not isinstance(xi, (list, tuple, dict))


def tree_flatten_with_none_as_leaf(x):
    import jax.tree_util

    # return jax.tree_util.tree_flatten(x, lambda xi: (xi is None) or not isinstance(x,(list,tuple,dict)) )
    return jax.tree_util.tree_flatten(x, is_leaf_with_none)


# import jax.tree_util
# jax.tree_util.tree_flatten({'a':1,'b':2,'c':None})
#
# import util
# a, treedef = util.tree_flatten_with_none_as_leaf({'a':1,'b':2,'c':None})
# print(a)
# print(treedef)
#
# jax.tree_util.tree_unflatten(treedef,a)


def num2str(id):
    if id < 26:
        return chr(ord("`") + 1 + id)
    else:
        last = id % 26
        rest = (id - last) // 26
        return num2str(rest) + num2str(last)


def is_shape_tuple(a):
    if not isinstance(a, tuple):
        return False
    for ai in a:
        if not isinstance(ai, int):
            return False
    return True


import jax.tree_util


def tree_map_recurse_at_leaf(f, tree, *remaining_trees, is_leaf=None):
    def mini_eval(leaf, *remaining_subtrees):
        return jax.tree_map(lambda *leaves: f(leaf, *leaves), *remaining_subtrees)

    return jax.tree_util.tree_map(mini_eval, tree, *remaining_trees, is_leaf=is_leaf)


def flatten_fun(f, *args):
    "get a new function that takes a single input (which is a list)"

    out = f(*args)
    args_treedef = jax.tree_util.tree_structure(args)
    out_treedef = jax.tree_util.tree_structure(out)

    # print(f"{out=}")
    # print(f"{args_treedef=}")
    # print(f"{out_treedef=}")

    def flat_f(*flat_args):
        args = jax.tree_util.tree_unflatten(args_treedef, flat_args)
        out = f(*args)
        flat_out, out_treedef2 = jax.tree_util.tree_flatten(out)
        assert out_treedef2 == out_treedef
        return flat_out

    def flatten_input(*args):
        flat_args, treedef = jax.tree_util.tree_flatten(args)
        # print(f"{args_treedef=}")
        # print(f"{treedef=}")
        assert treedef == args_treedef, "args don't match original"
        return flat_args

    def unflatten_output(flat_out):
        return jax.tree_util.tree_unflatten(out_treedef, flat_out)

    return flat_f, flatten_input, unflatten_output


# def replace_item(t,old2new):
#     assert isinstance(t,tuple), "must be called on tuple"
#     return (ti if ti not in old2new else old2new[ti] for ti in t)


################################################################################
# Cast observed variables to arrays (and check that they have corresponding shapes)
################################################################################


def assimilate_vals(vars, vals):
    """
    convert `vals` to a pytree of arrays with the same shape as `vars`
    The purpose of this is when users might provide lists / tuples
    that should be auto-casted to a pytree of arrays. (Without `vars`
    it would be impossible to tell a list of arrays of the same length
    from a big array with one more dimension.)
    """
    new_vals = jax.tree_map(lambda var, val: jnp.array(val), vars, vals)
    flat_vars, vars_treedef = jax.tree_util.tree_flatten(vars)
    flat_vals, vals_treedef = jax.tree_util.tree_flatten(new_vals)
    assert (
        vars_treedef == vals_treedef
    ), "vars and vals must have same structure (after conversion to arrays)"
    for var, val in zip(flat_vars, flat_vals):
        assert (
            var.shape == val.shape
        ), "vars and vals must have matching shape (after conversion to arrays)"
    return new_vals


################################################################################
# flatten pytree argument triplets
################################################################################


def flatten_args(vars, given_vars=None, given_vals=None):
    given_vals = assimilate_vals(given_vars, given_vals)

    flat_vars, vars_treedef = jax.tree_util.tree_flatten(vars)
    flat_given_vars, given_vars_treedef = jax.tree_util.tree_flatten(given_vars)
    flat_given_vals, given_vals_treedef = jax.tree_util.tree_flatten(given_vals)
    assert given_vars_treedef == given_vals_treedef

    def unflatten_vars(flat_samps):
        return jax.tree_util.tree_unflatten(vars_treedef, flat_samps)

    return flat_vars, flat_given_vars, flat_given_vals, unflatten_vars


def nth_index(lst, item, n):
    # https://stackoverflow.com/questions/22267241/how-to-find-the-index-of-the-nth-time-an-item-appears-in-a-list
    return [i for i, n in enumerate(lst) if n == item][n]


def first_index(lst, condition):
    for n, x in enumerate(lst):
        if condition(x):
            return n
    raise Exception("none found")


def swapped_list(lst, i, j):
    new_lst = lst.copy()
    new_lst[i], new_lst[j] = new_lst[j], new_lst[i]
    return new_lst


def first(lst, cond, default=None):
    """
    get first element of `lst` satisfying `cond` or if none then `default`
    """
    return next((x for x in lst if cond(x)), default)


def replace_in_sequence(seq, i, new):
    assert i >= 0
    assert i < len(seq)
    s = type(seq)
    return seq[:i] + s([new]) + seq[i + 1 :]
