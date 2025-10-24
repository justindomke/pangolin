from jax import numpy as jnp
import jax.tree_util
import numpy as np
from typing import Sequence, Any, Callable, Iterable


def comma_separated(stuff, fun=None, parens=True, spaces=False):
    """convenience function for printing and such

    Examples
    --------
    >>> comma_separated(['a', 'b', 'c'])
    '(a,b,c)'
    >>> comma_separated(['a', 'b', 'c'], lambda s: s + "0")
    '(a0,b0,c0)'
    >>> comma_separated(['a', 'b', 'c'], parens=False)
    'a,b,c'
    >>> comma_separated(['a', 'b', 'c'], lambda s: s + "0", parens=False)
    'a0,b0,c0'
    """
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
            if spaces:
                ret += " "
    if parens:
        ret += ")"
    return ret


class VarNames:
    """Convenience class to automatically give unique string names to objects.

    Examples
    --------
    >>> var_names = VarNames()
    >>> var_names['bob']
    'v0v'
    >>> var_names['alice']
    'v1v'
    >>> var_names['bob']
    'v0v'
    >>> var_names['carlos']
    'v2v'
    >>> var_names['alice']
    'v1v'
    >>> var_names['bob']
    'v0v'
    """

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
    """Is exactly one argument not None?

    Examples
    --------
    >>> one_not_none(1)
    1
    >>> one_not_none(1,2)
    2
    >>> one_not_none(1,None)
    1
    >>> one_not_none(None,1,None)
    1
    """
    # TODO: should be num_not_none

    num = 0
    for arg in args:
        if arg is not None:
            num += 1
    return num


def all_unique(lst):
    return len(lst) == len(set(lst))


def intersects(A: Iterable, B: Iterable) -> bool:
    """
    Check if two collections have any common elements.

    Parameters
    ----------
    A : iterable
        First collection of elements.
    B : iterable
        Second collection of elements.

    Returns
    -------
    bool
        True if A and B share at least one element, False otherwise.

    Examples
    --------
    >>> intersects([1, 2, 3], [3, 4, 5])
    True

    >>> intersects(['apple', 'banana'], ['orange', 'grape'])
    False
    """
    return bool(set(A) & set(B))


class WriteOnceDict(dict):
    """A dict where you can't overwrite entries

    Examples
    --------
    >>> d = WriteOnceDict()
    >>> d['a'] = 2
    >>> d['a']
    2
    >>> d['b'] = 3
    >>> d['b']
    3
    >>> d['a'] = 1
    Traceback (most recent call last):
    ValueError: Cannot overwrite existing key a in WriteOnceDict
    """

    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(f"Cannot overwrite existing key {key} in WriteOnceDict")
        super().__setitem__(key, value)


class WriteOnceDefaultDict(dict):
    """A dict where you can't overwrite entries, with a default value

    Examples
    --------
    >>> d = WriteOnceDefaultDict(lambda s : len(s))
    >>> d['bob']
    3
    >>> d['bob'] = 5
    >>> d['bob']
    5
    >>> d['bob'] = 3
    Traceback (most recent call last):
    ValueError: Cannot overwrite existing key bob in WriteOnceDefaultDict
    """

    # TODO: if an entry is accessed, should we freeze it so it can't be overwritten after that?

    def __init__(self, default_factory):
        self.default_factory = default_factory
        super().__init__()

    def __setitem__(self, key, value):
        if key in self:
            raise ValueError(
                f"Cannot overwrite existing key {key} in WriteOnceDefaultDict"
            )
        super().__setitem__(key, value)

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        else:
            return self.default_factory(key)


def is_leaf_with_none(xi):
    # return (xi is None) or not isinstance(xi, (list, tuple, dict))
    return xi is None


def tree_map_with_none_as_leaf(f, tree, *rest):
    """
    Call jax.tree_util.tree_map using a special is_leaf function that preserves None

    Examples
    --------
    >>> pytree = [0, (1, None)]
    >>> tree_map_with_none_as_leaf(lambda x: x, pytree)
    [0, (1, None)]

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


def tree_structure_with_none_as_lead(pytree):
    import jax.tree_util

    return jax.tree_util.tree_structure(pytree, is_leaf=is_leaf_with_none)


def tree_flatten_with_none_as_leaf(x):
    import jax.tree_util

    # return jax.tree_util.tree_flatten(x, lambda xi: (xi is None) or not isinstance(x,(list,tuple,dict)) )
    return jax.tree_util.tree_flatten(x, is_leaf=is_leaf_with_none)


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
    >>> tree = np.array([1, 2]), (np.array([3, 4]), None)
    >>> map_inside_tree(f, tree)
    ((array([4, 6]), None), None)
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


################################################################################
# Pytree stuff
################################################################################


PyTree = Any


def tree_map_recurse_at_leaf(
    f: Callable[..., Any],
    tree: PyTree,
    *remaining_trees: PyTree,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
    """
    Applies a function `f` to corresponding leaves of `tree` and `*remaining_trees`.

    This function implements a "recursive broadcast" behavior. If `tree` has a leaf
    at a path where any of `remaining_trees` has a subtree, that `tree` leaf is
    "broadcast" to all leaves of the corresponding subtree in `remaining_trees`.

    Parameters
    ----------
    f : Callable[..., Any]
        The function to apply to the leaves. Its first argument will be a leaf
        from `tree`, and subsequent arguments will be corresponding leaves from
        `remaining_trees`. If broadcasting occurs, the first argument (`leaf_from_tree`)
        will be fixed for all leaves within the broadcasted subtree.
    tree : PyTree
        The primary PyTree. Its leaves will trigger the broadcasting behavior.
        It is expected to be a "prefix" or "smaller" structure compared to
        `remaining_trees` at corresponding paths.
    *remaining_trees : PyTree
        One or more additional PyTrees to map over. These are expected to be
        "superset" structures relative to `tree` at corresponding paths.
    is_leaf : Callable[[Any], bool], optional
        An optional callable that takes a single argument (a node in a PyTree)
        and returns `True` if that node should be considered a leaf (i.e.,
        `tree_map` should not recurse into it), and `False` otherwise.
        This function is applied to both the outer and inner `jax.tree_map` calls.
        If `None`, JAX's default leaf detection is used.

    Returns
    -------
    PyTree
        A new PyTree with the results of `f` application. Its structure will
        match that of the first PyTree in `remaining_trees` (or `tree` if
        `remaining_trees` is empty).

    Notes
    -----
    - If `tree` has a subtree at a path where one of `remaining_trees` has a leaf,
      `jax.tree_map` will raise a `ValueError` due to a structural mismatch.
    - This function leverages nested `jax.tree_map` calls for conciseness.


    Examples
    --------
    >>> # No broadcasting (standard tree_map behavior)
    >>> tree1 = {'c': 5, 'd': 6}
    >>> tree2 = {'c': 2, 'd': 3}
    >>> tree_map_recurse_at_leaf(lambda l1, l2: l1 * l2, tree1, tree2)
    {'c': 10, 'd': 18}

    >>> # Simple broadcasting (multiply first leaf by second)
    >>> tree1 = {'a': 10, 'b': 20}
    >>> tree2 = {'a': {'x': 1, 'y': 2}, 'b': 3}
    >>> tree_map_recurse_at_leaf(lambda l1, l2: l1 * l2, tree1, tree2)
    {'a': {'x': 10, 'y': 20}, 'b': 60}

    >>> # Custom is_leaf (treating None as a leaf)
    >>> tree1 = {'data': 100, 'config': None}
    >>> tree2 = {'data': {'val': 1, 'factor': 2}, 'config': 'default'}
    >>> tree_map_recurse_at_leaf(
    ...     lambda l1, l2: f"{l1}_{l2}" if l1 is None else l1 * l2,
    ...     tree1, tree2, is_leaf=lambda x: x is None
    ... )
    {'config': 'None_default', 'data': {'factor': 200, 'val': 100}}
    """

    def mini_eval(leaf: Any, *remaining_subtrees: PyTree) -> PyTree:
        # `leaf` here is a leaf from the `tree` (first argument) as determined
        # by the outer `jax.tree_map`'s `is_leaf` rule.
        # `remaining_subtrees` are the corresponding subtrees from `*remaining_trees`.
        # The inner `jax.tree_map` then broadcasts `leaf` across the leaves
        # of `remaining_subtrees`.
        return jax.tree_map(
            lambda *leaves: f(leaf, *leaves),
            *remaining_subtrees,
            is_leaf=is_leaf,  # Propagate the custom is_leaf to the inner map
        )

    # The outer `jax.tree_map` traverses `tree` and `*remaining_trees` in parallel.
    # When it encounters a leaf in `tree`, it calls `mini_eval` with that leaf
    # and the corresponding subtrees from `*remaining_trees`.
    return jax.tree_util.tree_map(
        mini_eval,
        tree,
        *remaining_trees,
        is_leaf=is_leaf,  # Pass the custom is_leaf to the outer map
    )


def tree_map_recurse_at_leaf_with_none_as_leaf(f, tree, *remaining_trees):
    """
    Examples
    --------
    >>> pytree1 = (0,[1,2])
    >>> pytree2 = ("dog", ["cat", None])
    >>> tree_map_recurse_at_leaf_with_none_as_leaf(lambda a,b: a, pytree1, pytree2)
    (0, [1, 2])

    >>> pytree1 = 3
    >>> pytree2 = {"cat": 0, "dog": 2}
    >>> tree_map_recurse_at_leaf_with_none_as_leaf(lambda a,b: a, pytree1, pytree2)
    {'cat': 3, 'dog': 3}
    """
    return tree_map_recurse_at_leaf(
        f, tree, *remaining_trees, is_leaf=is_leaf_with_none
    )


def flatten_fun(f, *args, is_leaf=None):
    "get a new function that takes a single input (which is a list)"

    out = f(*args)
    args_treedef = jax.tree_util.tree_structure(args, is_leaf=is_leaf)
    out_treedef = jax.tree_util.tree_structure(out, is_leaf=is_leaf)

    def flat_f(*flat_args):
        args = jax.tree_util.tree_unflatten(args_treedef, flat_args)
        out = f(*args)
        flat_out, out_treedef2 = jax.tree_util.tree_flatten(out, is_leaf=is_leaf)
        assert out_treedef2 == out_treedef
        return flat_out

    def flatten_input(*args):
        flat_args, treedef = jax.tree_util.tree_flatten(args, is_leaf=is_leaf)
        assert treedef == args_treedef, "args don't match original"
        return flat_args

    def unflatten_output(flat_out):
        return jax.tree_util.tree_unflatten(out_treedef, flat_out)

    return flat_f, flatten_input, unflatten_output


def _check_tree_consistency(*args):
    trees = [jax.tree_util.tree_structure(args, is_leaf=is_leaf_with_none)]
    # trees = [tree_structure_with_none(args)]
    for t in trees:
        assert t == trees[0]


def dual_flatten(pytree1, pytree2):
    """
    Examples
    --------
    >>> pytree1 = (0,[1,2])
    >>> pytree2 = ("dog",["cat", "owl"])
    >>> dual_flatten(pytree1, pytree2)
    ([0, 1, 2], ['dog', 'cat', 'owl'])

    >>> pytree1 = (None,[1,2])
    >>> pytree2 = ("dog", ["cat", None])
    >>> dual_flatten(pytree1, pytree2)
    ([None, 1, 2], ['dog', 'cat', None])

    >>> pytree1 = (None, 3)
    >>> pytree2 = ("dog", ["cat", None])
    >>> dual_flatten(pytree1, pytree2)
    ([None, 3, 3], ['dog', 'cat', None])
    """
    new_tree1 = tree_map_recurse_at_leaf_with_none_as_leaf(
        lambda a, b: a, pytree1, pytree2
    )
    _check_tree_consistency(new_tree1, pytree2)

    flat_tree1, tree1_treedef = tree_flatten_with_none_as_leaf(new_tree1)
    flat_tree2, tree2_treedef = tree_flatten_with_none_as_leaf(pytree2)
    return flat_tree1, flat_tree2


################################################################################
# Cast observed variables to arrays (and check that they have corresponding shapes)
################################################################################


def short_pytree_string(treedef):
    "Get a string for a JAX PyTreeDef without printing PyTreeDef and scaring the noobs"
    s = str(treedef)
    assert s[:10] == "PyTreeDef(", "fdjhdj"
    assert s[-1] == ")"
    return s[10:-1]


def assimilate_vals(vars, vals):
    """
    convert `vals` to a pytree of arrays with the same shape as `vars`
    The purpose of this is when users might provide lists / tuples
    that should be auto-casted to a pytree of arrays. (Without `vars`
    it would be impossible to tell a list of arrays of the same length
    from a big array with one more dimension.)
    """
    try:
        new_vals = jax.tree_map(lambda var, val: jnp.array(val), vars, vals)
    except ValueError:
        vars_treedef = jax.tree_util.tree_structure(vars)
        vals_treedef = jax.tree_util.tree_structure(vals)
        raise ValueError(
            f"Not able to find common pytree structure for given vars and vals.\n"
            f"For given vars got: {short_pytree_string(vars_treedef)}.\n"
            f"For given vals got: {short_pytree_string(vals_treedef)}."
        )

    flat_vars, vars_treedef = jax.tree_util.tree_flatten(vars)
    flat_vals, vals_treedef = jax.tree_util.tree_flatten(new_vals)
    assert vars_treedef == vals_treedef, (
        f"vars and vals must have same structure (after conversion to arrays). ({vars_treedef} vs "
        f"{vals_treedef}"
    )
    for var, val in zip(flat_vars, flat_vals):
        if var.shape != val.shape:
            raise ValueError(
                f"given var {var} with shape {var.shape} does not match given val"
                f" {val} with shape {val.shape}."
            )
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


def reverse_dict(d: dict):
    return {value: key for key, value in d.items()}


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


################################################################################
# call np.allclose on pytrees instead of arrays
################################################################################


def tree_allclose(a: PyTree, b: PyTree, **kwargs: Any) -> bool:
    """Checks if two PyTrees are structurally identical and all leaves are close.

    This function first verifies that the two PyTrees have the exact same
    structure. If they do, it then compares each corresponding leaf node (array)
    using `np.allclose` and returns `True` if all leaves are close, `False`
    otherwise.

    Parameters
    ----------
    a : PyTree
        The first PyTree to compare.
    b : PyTree
        The second PyTree to compare.
    **kwargs : dict, optional
        Additional keyword arguments to be passed directly to `np.allclose`.
        Common arguments include `rtol` (relative tolerance) and `atol`
        (absolute tolerance).

    Returns
    -------
    bool
        `True` if the structures match and all corresponding leaves are close.
        `False` if the structures match but any leaves are not close.

    Raises
    ------
    ValueError
        If the PyTree structures of `a` and `b` are not identical.

    Examples
    --------
    >>> tree1 = {'a': jnp.array([1.0, 2.0]), 'b': (jnp.array(3.0),)}
    >>> tree2 = {'a': jnp.array([1.000001, 2.0]), 'b': (jnp.array(3.0),)}
    >>> tree_allclose(tree1, tree2, atol=1e-5)
    True

    >>> tree3 = {'a': jnp.array([1.0, 2.5]), 'b': (jnp.array(3.0),)}
    >>> tree_allclose(tree1, tree3)
    False

    >>> tree4 = {'a': jnp.array([1.0, 2.0])} # Different structure
    >>> try:
    ...     tree_allclose(tree1, tree4)
    ... except ValueError as e:
    ...     print(e)
    PyTree structures do not match.
     a structure: PyTreeDef({'a': *, 'b': (*,)})
     b structure: PyTreeDef({'a': *})

    """
    # 1. Check if the tree structures are the same.
    struct_a = jax.tree_util.tree_structure(a)
    struct_b = jax.tree_util.tree_structure(b)
    if struct_a != struct_b:
        raise ValueError(
            "PyTree structures do not match.\n"
            f" a structure: {struct_a}\n"
            f" b structure: {struct_b}"
        )

    # 2. Map np.allclose over the leaves to get a PyTree of booleans.
    leaves_are_close = jax.tree_util.tree_map(
        lambda x, y: np.allclose(x, y, **kwargs), a, b
    )

    # 3. Flatten the boolean PyTree and check if all values are True.
    return all(jax.tree_util.tree_leaves(leaves_are_close))


################################################################################
# Tricks for dealing with named arguments
################################################################################


import inspect


def get_positional_args(target_func, *args, **kwargs):
    """
    Transforms a mix of positional and keyword arguments into a list of
    purely positional arguments for a target function.

    This function inspects the signature of `target_func` to correctly
    map and order the provided arguments. It ensures that the resulting
    list of positional arguments, when passed to `target_func`, will
    produce the identical outcome as the original call with mixed arguments.

    Raises a `ValueError` if `target_func` contains any keyword-only
    arguments, as these cannot be represented in a purely positional call.
    Raises a `TypeError` if the provided arguments (`*args`, `**kwargs`)
    are invalid for `target_func` (e.g., missing required arguments,
    unexpected arguments).

    Parameters
    ----------
    target_func : callable
        The function whose signature will be used to transform and order
        the arguments.
    *args : tuple
        Positional arguments to be transformed.
    **kwargs : dict
        Keyword arguments to be transformed.

    Returns
    -------
    list
        A list of arguments in the correct positional order for `target_func`.

    Raises
    ------
    ValueError
        If `target_func` has keyword-only arguments.
    TypeError
        If the provided `*args` and `**kwargs` do not match the signature
        of `target_func`.

    See Also
    --------
    inspect.signature : For detailed information on function signature inspection.

    Examples
    --------
    >>> def add_five_numbers(a, b, c, d, e):
    ...     return a + b + c + d + e

    >>> # Case 1: All arguments provided positionally
    >>> get_positional_args(add_five_numbers, 1, 2, 3, 4, 5)
    [1, 2, 3, 4, 5]

    >>> # Case 2: Mixed positional and keyword arguments
    >>> get_positional_args(add_five_numbers, 1, 2, 3, e=5, d=4)
    [1, 2, 3, 4, 5]

    >>> # Case 3: All arguments provided as keywords
    >>> get_positional_args(add_five_numbers, a=10, b=20, c=30, d=40, e=50)
    [10, 20, 30, 40, 50]

    >>> # Case 4: Function with default arguments
    >>> def greet(name, greeting="Hello"):
    ...     return f"{greeting}, {name}!"
    >>> get_positional_args(greet, "Alice")
    ['Alice', 'Hello']
    >>> get_positional_args(greet, "Bob", greeting="Hi")
    ['Bob', 'Hi']
    >>> get_positional_args(greet, name="Charlie", greeting="Greetings")
    ['Charlie', 'Greetings']

    >>> # Case 5: Verification that the transformed arguments yield the same result
    >>> def multiply(x, y, z):
    ...     return x * y * z
    >>> original_args = (2,)
    >>> original_kwargs = {'z': 5, 'y': 3}
    >>> transformed = get_positional_args(multiply, *original_args, **original_kwargs)
    >>> multiply(*original_args, **original_kwargs) == multiply(*transformed)
    True

    >>> # Case 6: Attempting to transform a function with keyword-only arguments (raises ValueError)
    >>> def func_with_kw_only(a, b, *, kw_only_arg, default_kw_only=10):
    ...     pass
    >>> try:
    ...     get_positional_args(func_with_kw_only, 1, 2, kw_only_arg=3)
    ... except ValueError as e:
    ...     print(e)
    Function 'func_with_kw_only' has keyword-only arguments (kw_only_arg), which is not allowed by this transformer.

    >>> # Case 7: Missing a required argument (raises TypeError)
    >>> try:
    ...     get_positional_args(add_five_numbers, 1, 2, 3)
    ... except TypeError as e:
    ...     # Use '...' to match any varying parts of the error message
    ...     # and check for a key phrase.
    ...     assert "missing a required argument" in str(e)
    ...     print(e.__class__.__name__ + ": " + str(e))
    TypeError: ...

    >>> # Case 8: Providing an unexpected argument (raises TypeError)
    >>> try:
    ...     get_positional_args(add_five_numbers, 1, 2, 3, 4, 5, extra_arg=6)
    ... except TypeError as e:
    ...     print(e.__class__.__name__ + ": " + str(e))
    TypeError: ... an unexpected keyword argument 'extra_arg'
    """
    sig = inspect.signature(target_func)

    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            raise ValueError(
                f"Function '{target_func.__name__}' has keyword-only arguments "
                f"({param.name}), which is not allowed by this transformer."
            )

    # Bind the arguments. This will raise TypeError if arguments are invalid
    # (e.g., missing required args, unexpected args).
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    positional_args_for_target = []
    for param_name in sig.parameters:
        positional_args_for_target.append(bound_args.arguments[param_name])

    return positional_args_for_target
