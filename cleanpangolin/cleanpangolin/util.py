from jax import numpy as jnp
import jax.tree_util
import numpy as np
from typing import Sequence
import collections


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


class WriteOnceDefaultDict(dict):
    def __init__(self, default_factory):
        self.default_factory = default_factory
        super().__init__()

    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(f"Cannot overwrite existing key {key} in WriteOnceDefaultDict")
        super().__setitem__(key, value)

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        else:
            return self.default_factory(key)


def is_leaf_with_none(xi):
    return (xi is None) or not isinstance(xi, (list, tuple, dict))


def tree_map_with_none_as_leaf(f, tree, *rest):
    """
    Call jax.tree_util.tree_map using a special is_leaf function that preserves None
    """
    return jax.tree_util.tree_map(f, tree, *rest, is_leaf=is_leaf_with_none)


def tree_map_preserve_none(f, tree, *rest):
    def new_f(*args):
        if all(arg is None for arg in args):
            return None
        elif not any(arg is None for arg in args):
            return f(*args)
        else:
            assert False, "only handle all None or zero None args"

    return tree_map_with_none_as_leaf(new_f, tree, *rest)


def tree_flatten_with_none_as_leaf(x):
    import jax.tree_util

    # return jax.tree_util.tree_flatten(x, lambda xi: (xi is None) or not isinstance(x,(list,tuple,dict)) )
    return jax.tree_util.tree_flatten(x, is_leaf_with_none)


def same(x, y):
    """
    assert that x are either:
    * both None
    * equal floats
    * equal arrays (with equal shapes)
    """
    if x is None:
        return y is None
    elif hasattr(x, "shape"):
        return hasattr(y, "shape") and np.array_equal(x, y)
    else:
        return x == y


def same_tree(x, y, is_leaf=None):
    """
    Check that x and y have same tree structure (including None) and that all leaves
    are equal. Arrays are equal regardless of if they come from regular numpy or jax
    numpy and ignore types. (e.g. numpy.array([2,3])) is considered equal to
    jax.numpy.array([2.0,3.0]).)
    """
    treedef_x = jax.tree_util.tree_structure(x, is_leaf=is_leaf)
    treedef_y = jax.tree_util.tree_structure(y, is_leaf=is_leaf)
    if treedef_x != treedef_y:
        return False
    leaves_x = jax.tree_util.tree_leaves(x, is_leaf=is_leaf)
    leaves_y = jax.tree_util.tree_leaves(y, is_leaf=is_leaf)
    return all(same(xi, yi) for xi, yi in zip(leaves_x, leaves_y))


def map_inside_tree(f, tree):
    """
    Map a function over the leading axis for all leaf nodes inside a tree. If a None
    value is encountered at any input, it is presented to each function unchanged. If a
    None appears as an output, it is presented as an output unchanged.

    Examples
    -------
    >>> def f(t):
    ...     a, (b, c) = t
    ...     return (a + b, a * b), c
    >>> tree = np.array([1, 2]), (np.array([3, 4]), np.array([5, 6]))
    >>> map_inside_tree(f, tree)
    ((array([4, 6]), array([3, 8])), array([5, 6]))

    >>> def f(t):
    ...     a, (b, c) = t
    ...     return (a + b, c), None
    ... tree = np.array([1, 2]), (np.array([3, 4]), None)
    ... map_inside_tree(f, tree)
    (np.array([4, 6]), None), None
    """
    axis_size = tree_map_preserve_none(
        lambda n: n.shape[0],
        tree,
    )
    axis_sizes = jax.tree_util.tree_leaves(axis_size)
    if len(axis_sizes) == 0:
        # there are no inputs, so input to base observer is same
        rez = f(tree)
        if jax.tree_util.tree_leaves(rez):
            assert False, "size unclear for new value with no observations"
        return rez
    else:
        axis_size = axis_sizes[0]
        assert all(size == axis_size for size in axis_sizes)
        rez = []
        for i in range(axis_size):
            my_tree = tree_map_preserve_none(
                lambda n: n[i],
                tree,
            )
            my_rez = f(my_tree)
            rez.append(my_rez)
        rez = tree_map_preserve_none(
            lambda *r: np.stack(r),
            *rez,
        )
        return rez


def assert_all_leaves_instance_of(tree, type, is_leaf=None):
    for node in jax.tree_util.tree_flatten(tree, is_leaf)[0]:
        assert isinstance(node, type)


def assert_all_leaves_instance_of_with_none(tree, type):
    assert_all_leaves_instance_of(tree, type, is_leaf_with_none)


def assert_is_sequence_of(seq, type):
    assert isinstance(seq, Sequence)
    for x in seq:
        assert isinstance(x, type)


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


def tree_map_recurse_at_leaf(f, tree, *remaining_trees, is_leaf=None):
    def mini_eval(leaf, *remaining_subtrees):
        return jax.tree_map(lambda *leaves: f(leaf, *leaves), *remaining_subtrees)

    return jax.tree_util.tree_map(mini_eval, tree, *remaining_trees, is_leaf=is_leaf)


def flatten_fun(f, *args, is_leaf=None):
    "get a new function that takes a single input (which is a list)"

    out = f(*args)
    args_treedef = jax.tree_util.tree_structure(args, is_leaf=is_leaf)
    out_treedef = jax.tree_util.tree_structure(out, is_leaf=is_leaf)

    # print(f"{out=}")
    # print(f"{args_treedef=}")
    # print(f"{out_treedef=}")

    def flat_f(*flat_args):
        args = jax.tree_util.tree_unflatten(args_treedef, flat_args)
        out = f(*args)
        flat_out, out_treedef2 = jax.tree_util.tree_flatten(out, is_leaf=is_leaf)
        assert out_treedef2 == out_treedef
        return flat_out

    def flatten_input(*args):
        flat_args, treedef = jax.tree_util.tree_flatten(args, is_leaf=is_leaf)
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

    def unflatten_given(flat_given_vars):
        return jax.tree_util.tree_unflatten(given_vars_treedef, flat_given_vars)

    return flat_vars, flat_given_vars, flat_given_vals, unflatten_vars, unflatten_given


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


def camel_case_to_snake_case(name):
    # from https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    import re

    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def most_specific_class(*args, base_classes=()):
    classes = base_classes + tuple(type(a) for a in args)
    print(f"{classes=}")
    for c in classes:
        if all(issubclass(c, d) for d in classes):
            return c
    raise ValueError("no single most-specific argument type")


def is_numeric_numpy_array(x):
    return np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)


def unzip(source: Sequence[tuple], strict=False):
    """
    Reverses zip
    """
    return zip(*source, strict=strict)

