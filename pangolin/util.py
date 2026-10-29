"""General-purpose utilities used throughout Pangolin.

This foundational module does not import or use other parts of Pangolin.
"""

from __future__ import annotations  # so it's possible to preserve type aliases

import inspect
import re
from collections.abc import Hashable
from typing import Sequence, Mapping, Any, Callable, Iterable, Literal, overload

import jax.tree_util
from jax import numpy as jnp
import numpy as np

__all__ = [
    "PyTree",
    "comma_separated",
    "num_not_none",
    "all_unique",
    "intersects",
    "nth_index",
    "first_index",
    "first",
    "swapped_list",
    "replace_in_sequence",
    "reverse_dict",
    "node_names",
    "unzip",
    "starts_with",
    "num2str",
    "camel_case_to_snake_case",
    "is_shape_tuple",
    "is_numeric_numpy_array",
    "most_specific_class",
    "VarNames",
    "WriteOnceDict",
    "WriteOnceDefaultDict",
    "is_leaf_with_none",
    "tree_map_with_none_as_leaf",
    "tree_flatten_with_none_as_leaf",
    "tree_flatten_with_path_with_none_as_leaf",
    "tree_map_recurse_at_leaf",
    "tree_map_recurse_at_leaf_with_none_as_leaf",
    "flatten_fun",
    "dual_flatten",
    "same_tree",
    "tree_allclose",
    "map_inside_tree",
    "assert_all_leaves_instance_of",
    "assert_all_leaves_instance_of_with_none",
    "assert_is_sequence_of",
    "assimilate_vals",
    "flatten_args",
    "fixed_arity",
    "get_positional_args",
    "dag_lines",
    "render_dag",
]


# General collections, sequences, and naming


def node_names[N: Hashable](
    nodes: Iterable[N], namespace: Mapping[str, object]
) -> dict[N, list[str]]:
    """Find all names of selected objects in a namespace and its nested containers.

    Parameters
    ----------
    nodes
        Objects to name, matched by identity, without traversing their parents. Objects must
        be hashable and distinct objects must not compare equal, as for DAG nodes and RVs.
    namespace
        Mapping from names to values, such as ``locals()``. Dictionary values, lists, and
        tuples are searched recursively in iteration order. Other values are leaves; dictionary
        keys are used in paths but are not searched. Containers must be acyclic and must not
        change during traversal.

    Returns
    -------
    dict
        Each matched object maps to all its names in depth-first namespace traversal order.
        Names use list/tuple indices and the repr of dictionary keys, for example
        ``states[0]`` or ``params['scale']``. Missing objects are omitted. Shared containers
        are traversed once; their relative paths are reused for every occurrence. Selected
        objects are treated as leaves, even if they are themselves containers.
        Repeated names for the same object are retained, but a name shared by distinct
        selected objects is rejected.

    Raises
    ------
    ValueError
        If a cycle is encountered in the searched containers, or a name refers to two
        distinct selected objects.

    Examples
    --------
    >>> x, y = object(), object()
    >>> names = node_names([x, y], {"states": [x, x, y], "x": x})
    >>> names[x]
    ['states[0]', 'states[1]', 'x']
    >>> names[y]
    ['states[2]']
    >>> shared = {"scale": (y,)}
    >>> node_names([y], {"a": shared, "b": shared})[y]
    ["a['scale'][0]", "b['scale'][0]"]
    >>> node_names([x], {"count": 3})
    {}
    """
    selected = {id(node): node for node in nodes}
    cached: dict[int, list[tuple[N, str]]] = {}
    active: set[int] = set()

    def relative_names(value: object) -> list[tuple[N, str]]:
        identity = id(value)
        if identity in selected:
            return [(selected[identity], "")]
        if not isinstance(value, (dict, list, tuple)):
            return []
        if identity in active:
            raise ValueError("node_names requires acyclic containers")
        if identity in cached:
            return cached[identity]

        active.add(identity)
        paths: list[tuple[N, str]] = []
        entries = value.items() if isinstance(value, dict) else enumerate(value)
        for key, child in entries:
            prefix = f"[{key!r}]"
            paths.extend((node, prefix + path) for node, path in relative_names(child))
        active.remove(identity)
        cached[identity] = paths
        return paths

    names: dict[N, list[str]] = {}
    named_nodes: dict[str, N] = {}
    for name, value in namespace.items():
        for node, path in relative_names(value):
            full_name = name + path
            if full_name in named_nodes and named_nodes[full_name] is not node:
                raise ValueError(f"Name {full_name!r} refers to distinct objects")
            named_nodes[full_name] = node
            names.setdefault(node, []).append(full_name)
    return names


def comma_separated(
    stuff: Sequence[Any],
    fun: Callable[[Any], str] | None = None,
    parens: bool = True,
    spaces: bool = False,
) -> str:
    """Join a sequence into a comma-separated string.

    Parameters
    ----------
    stuff
        Strings to join.
    fun
        Optional conversion applied to each item before joining.
    parens
        Whether to surround the result with parentheses.
    spaces
        Whether to add a space after each comma.

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
    >>> comma_separated(['a', 'b', 'c'], spaces=True)
    '(a, b, c)'
    >>> comma_separated(['a', 'b', 'c'], lambda s: s + "0", spaces=True)
    '(a0, b0, c0)'
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


def num_not_none(*args: Any) -> int:
    """Count the arguments that are not ``None``.

    Examples
    --------
    >>> num_not_none(1)
    1
    >>> num_not_none(1,2)
    2
    >>> num_not_none(1,None)
    1
    >>> num_not_none(None,1,None)
    1
    >>> num_not_none(None,"cat","dog",77,None)
    3
    """

    num = 0
    for arg in args:
        if arg is not None:
            num += 1
    return num


def all_unique(a: Sequence[Any]) -> bool:
    """Return whether every item in a sequence is unique.

    The items must be hashable.

    Examples
    --------
    >>> all_unique([1, 2, 3])
    True
    >>> all_unique([1, 2, 1])
    False
    """
    return len(a) == len(set(a))


def intersects(A: Iterable[Any], B: Iterable[Any]) -> bool:
    """Return whether two iterables contain a common item.

    The items must be hashable.

    Examples
    --------
    >>> intersects([1, 2, 3], [3, 4, 5])
    True
    >>> intersects(['apple', 'banana'], ['orange', 'grape'])
    False
    """
    return bool(set(A) & set(B))


def nth_index(lst: Sequence[Any], item: Any, n: int) -> int:
    """Return the index of the ``n``-th occurrence of an item.

    Occurrences are numbered from zero. An ``IndexError`` is raised if the
    requested occurrence does not exist.

    Examples
    --------
    >>> nth_index(['a', 'b', 'a'], 'a', 1)
    2
    """
    # https://stackoverflow.com/questions/22267241/how-to-find-the-index-of-the-nth-time-an-item-appears-in-a-list
    return [i for i, n in enumerate(lst) if n == item][n]


def first_index(lst: Sequence[Any], condition: Callable[[Any], bool]) -> int:
    """Return the index of the first item satisfying a condition.

    An ``Exception`` is raised if no item satisfies the condition.

    Examples
    --------
    >>> first_index([1, 4, 7], lambda x: x > 3)
    1
    """
    for n, x in enumerate(lst):
        if condition(x):
            return n
    raise Exception("none found")


def first(lst: Iterable[Any], cond: Callable[[Any], bool], default: Any = None) -> Any:
    """Return the first item satisfying a condition, or a default value.

    Examples
    --------
    >>> first([1, 4, 7], lambda x: x > 3)
    4
    >>> first([1, 4, 7], lambda x: x > 10, default='missing')
    'missing'
    """
    return next((x for x in lst if cond(x)), default)


def swapped_list(lst: list[Any], i: int, j: int) -> list[Any]:
    """Return a shallow copy of a list with two items exchanged.

    Examples
    --------
    >>> swapped_list(['a', 'b', 'c'], 0, 2)
    ['c', 'b', 'a']
    """
    new_lst = lst.copy()
    new_lst[i], new_lst[j] = new_lst[j], new_lst[i]
    return new_lst


@overload
def replace_in_sequence(seq: list[Any], i: int, new: Any) -> list[Any]: ...


@overload
def replace_in_sequence(seq: tuple[Any, ...], i: int, new: Any) -> tuple[Any, ...]: ...


def replace_in_sequence(
    seq: list[Any] | tuple[Any, ...], i: int, new: Any
) -> list[Any] | tuple[Any, ...]:
    """Return a list or tuple with one item replaced, preserving its type.

    Examples
    --------
    >>> replace_in_sequence(('a', 'b', 'c'), 1, 'x')
    ('a', 'x', 'c')
    """
    assert i >= 0
    assert i < len(seq)
    if isinstance(seq, tuple):
        return seq[:i] + (new,) + seq[i + 1 :]
    return seq[:i] + [new] + seq[i + 1 :]


def reverse_dict(d: dict[Any, Any]) -> dict[Any, Any]:
    """Exchange the keys and values of a dictionary.

    Later keys win when the input contains duplicate values.

    Examples
    --------
    >>> reverse_dict({'a': 1, 'b': 2})
    {1: 'a', 2: 'b'}
    """
    return {value: key for key, value in d.items()}


def unzip(source: Sequence[tuple[Any, ...]], strict: bool = False) -> zip:
    """Transpose a sequence of tuples, reversing ``zip``.

    Examples
    --------
    >>> tuple(unzip([(1, 'a'), (2, 'b')]))
    ((1, 2), ('a', 'b'))
    """
    return zip(*source, strict=strict)


def starts_with(main: Sequence[Any], prefix: Sequence[Any]) -> bool:
    """Return whether a sequence begins with another sequence.

    Examples
    --------
    >>> starts_with([1, 2, 3], [1, 2])
    True
    """
    return main[: len(prefix)] == prefix


def num2str(id: int) -> str:
    """Encode a nonnegative integer using lowercase letters as base-26 digits.

    Examples
    --------
    >>> [num2str(i) for i in (0, 25, 26, 676)]
    ['a', 'z', 'ba', 'baa']
    """
    if id < 26:
        return chr(ord("`") + 1 + id)
    else:
        last = id % 26
        rest = (id - last) // 26
        return num2str(rest) + num2str(last)


def camel_case_to_snake_case(name: str) -> str:
    """Convert a CamelCase name to snake_case.

    Examples
    --------
    >>> camel_case_to_snake_case('NormalPrec')
    'normal_prec'
    """
    # from https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def is_shape_tuple(a: Any) -> bool:
    """Return whether an object is a tuple containing only integers.

    Examples
    --------
    >>> is_shape_tuple((2, 3, 4))
    True
    >>> is_shape_tuple([2, 3, 4])
    False
    """
    if not isinstance(a, tuple):
        return False
    for ai in a:
        if not isinstance(ai, int):
            return False
    return True


def is_numeric_numpy_array(x: np.ndarray) -> bool:
    """Return whether a NumPy array has an integer or floating-point dtype.

    Examples
    --------
    >>> is_numeric_numpy_array(np.array([1.0, 2.0]))
    True
    >>> is_numeric_numpy_array(np.array(['a', 'b']))
    False
    """
    return np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)


def most_specific_class(*args: Any, base_classes: tuple[type, ...] = ()) -> type:
    """Find the single most-specific class shared by the given classes.

    The candidates are the runtime classes of ``args`` plus ``base_classes``.
    A ``ValueError`` is raised if no candidate is a subclass of every other
    candidate.

    Examples
    --------
    >>> most_specific_class(1, True)
    <class 'bool'>
    """
    classes = base_classes + tuple(type(a) for a in args)
    for c in classes:
        if all(issubclass(c, d) for d in classes):
            return c
    raise ValueError("no single most-specific argument type")


class VarNames:
    """Assign a stable, unique string name to each object encountered.

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

    def __init__(self) -> None:
        self.num = 0
        self.node_to_name: dict[Any, str] = {}

    def __getitem__(self, node: Any) -> str:
        if node in self.node_to_name:
            varname = self.node_to_name[node]
        else:
            varname = f"v{self.num}v"
            self.num += 1
            self.node_to_name[node] = varname
        return varname


class WriteOnceDict(dict):
    """Dictionary that rejects attempts to overwrite an existing key.

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

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self:
            raise ValueError(f"Cannot overwrite existing key {key} in WriteOnceDict")
        super().__setitem__(key, value)


class WriteOnceDefaultDict(dict):
    """Write-once dictionary that creates missing values on first access.

    Accessing a missing key calls ``default_factory`` with that key and stores
    the result. Once a key has been written or accessed, its value cannot be
    changed.

    Parameters
    ----------
    default_factory
        Function used to create values for missing keys.

    Examples
    --------
    >>> d = WriteOnceDefaultDict(lambda s : len(s))
    >>> d['cat'] = 5
    >>> d['bob']
    3
    >>> d['cat']
    5
    >>> d['bob'] = 5 # raises an error (as expected)
    Traceback (most recent call last):
    ValueError: Cannot overwrite existing key bob in WriteOnceDefaultDict
    """

    def __init__(self, default_factory: Callable[[Any], Any]) -> None:
        self.default_factory = default_factory
        super().__init__()

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self:
            raise ValueError(f"Cannot overwrite existing key {key} in WriteOnceDefaultDict")
        super().__setitem__(key, value)

    def __getitem__(self, key: Any) -> Any:
        if key not in self:
            self.__setitem__(key, self.default_factory(key))

        return super().__getitem__(key)


# PyTree utilities


type PyTree[T] = T | Sequence[PyTree[T]] | Mapping[Any, PyTree[T]]
"""Statically checkable approximation of a JAX PyTree whose leaves have type ``T``.

Unlike ``jaxtyping.PyTree``, which is runtime-only, this alias lets static checkers enforce ``T``.
The recursive interfaces are read-only so concrete container types are preserved. JAX's runtime
registry remains authoritative. ``None`` is accepted only when it is included in ``T``.
"""


def is_leaf_with_none(xi: Any) -> bool:
    """Return whether a PyTree node is ``None``.

    Pass this predicate to JAX PyTree functions when ``None`` must be retained
    as a leaf instead of treated as an empty tree.

    Examples
    --------
    >>> is_leaf_with_none(None)
    True
    >>> is_leaf_with_none(0)
    False
    """
    return xi is None


def tree_map_with_none_as_leaf(f: Callable[..., Any], tree: PyTree, *rest: PyTree) -> PyTree:
    """Map over PyTrees while treating ``None`` as a leaf.

    This behaves like :func:`jax.tree_util.tree_map`, except that ``None``
    nodes are passed to ``f`` instead of treated as empty trees.

    Parameters
    ----------
    f
        Function applied to corresponding leaves.
    tree
        PyTree that supplies the first argument to ``f``.
    rest
        Additional PyTrees that supply subsequent arguments to ``f``.

    Examples
    --------
    >>> f = lambda x: 0 if x is None else len(x)
    >>> tree = ["cat", ("bear", None)]
    >>> jax.tree_util.tree_map(f, tree)
    [3, (4, None)]
    >>> tree_map_with_none_as_leaf(f, tree)
    [3, (4, 0)]

    >>> f2 = lambda leaf, seq: f(leaf) + len(seq)
    >>> tree1 = ["cat", ("bear", None)]
    >>> tree2 = [[1,2,3,4,5], ([], [1,2,3,4,5])]
    >>> tree_map_with_none_as_leaf(f2, tree1, tree2)
    [8, (4, 5)]
    """
    return jax.tree_util.tree_map(f, tree, *rest, is_leaf=is_leaf_with_none)


def tree_flatten_with_none_as_leaf(x: PyTree) -> tuple[list[Any], Any]:
    """Flatten a PyTree while retaining ``None`` as a leaf.

    Examples
    --------
    >>> leaves, treedef = tree_flatten_with_none_as_leaf([1, None])
    >>> leaves
    [1, None]
    >>> jax.tree_util.tree_unflatten(treedef, leaves)
    [1, None]
    """
    return jax.tree_util.tree_flatten(x, is_leaf=is_leaf_with_none)


def tree_flatten_with_path_with_none_as_leaf(
    x: PyTree,
) -> tuple[list[tuple[Any, Any]], Any]:
    """Flatten a PyTree with paths while retaining ``None`` as a leaf.

    Examples
    --------
    >>> path_leaves, _ = tree_flatten_with_path_with_none_as_leaf({"x": [1, None]})
    >>> [(jax.tree_util.keystr(path), leaf) for path, leaf in path_leaves]
    [("['x'][0]", 1), ("['x'][1]", None)]
    """
    return jax.tree_util.tree_flatten_with_path(x, is_leaf=is_leaf_with_none)


def tree_map_recurse_at_leaf(
    f: Callable[..., Any],
    tree: PyTree,
    *remaining_trees: PyTree,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
    """Map over PyTrees, recursing into later trees where the first has a leaf.

    If ``tree`` has a leaf where a remaining tree has a subtree, that leaf is
    broadcast to every corresponding leaf in the remaining subtree. The result
    has the expanded structure of the remaining trees.

    Parameters
    ----------
    f
        Function applied to corresponding leaves.
    tree
        Primary PyTree, whose leaves trigger recursive broadcasting.
    remaining_trees
        PyTrees whose corresponding subtrees determine the expanded structure.
    is_leaf
        Optional predicate used by both the outer and inner PyTree mappings.

    Notes
    -----
    If ``tree`` has a subtree where a remaining tree has a leaf, JAX raises a
    ``ValueError`` for the structural mismatch.

    Examples
    --------
    >>> tree1 = {'c': 5, 'd': 6}
    >>> tree2 = {'c': 2, 'd': 3}
    >>> tree_map_recurse_at_leaf(lambda l1, l2: l1 * l2, tree1, tree2)
    {'c': 10, 'd': 18}

    >>> tree1 = {'a': 10, 'b': 20}
    >>> tree2 = {'a': {'x': 1, 'y': 2}, 'b': 3}
    >>> tree_map_recurse_at_leaf(lambda l1, l2: l1 * l2, tree1, tree2)
    {'a': {'x': 10, 'y': 20}, 'b': 60}

    >>> tree1 = {'data': 100, 'config': None}
    >>> tree2 = {'data': {'val': 1, 'factor': 2}, 'config': 'default'}
    >>> tree_map_recurse_at_leaf(
    ...     lambda l1, l2: f"{l1}_{l2}" if l1 is None else l1 * l2,
    ...     tree1, tree2, is_leaf=lambda x: x is None
    ... )
    {'config': 'None_default', 'data': {'factor': 200, 'val': 100}}
    """

    def mini_eval(leaf: Any, *remaining_subtrees: PyTree) -> PyTree:
        return jax.tree_util.tree_map(
            lambda *leaves: f(leaf, *leaves),
            *remaining_subtrees,
            is_leaf=is_leaf,
        )

    return jax.tree_util.tree_map(
        mini_eval,
        tree,
        *remaining_trees,
        is_leaf=is_leaf,
    )


def tree_map_recurse_at_leaf_with_none_as_leaf(
    f: Callable[..., Any], tree: PyTree, *remaining_trees: PyTree
) -> PyTree:
    """Recursively broadcast a PyTree mapping while retaining ``None`` leaves.

    This is :func:`tree_map_recurse_at_leaf` with
    :func:`is_leaf_with_none` selected as its leaf predicate.

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
    return tree_map_recurse_at_leaf(f, tree, *remaining_trees, is_leaf=is_leaf_with_none)


def flatten_fun(
    f: Callable[..., PyTree],
    *args: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> tuple[Callable, Callable, Callable]:
    """Adapt a PyTree function to accept and return flat lists of leaves.

    ``args`` establish the input structure and evaluate ``f`` once to establish
    its output structure. The returned callables respectively evaluate the flat
    function, flatten new inputs, and reconstruct outputs.

    Parameters
    ----------
    f
        Function to adapt.
    args
        Representative arguments establishing the expected input structure.
    is_leaf
        Optional PyTree leaf predicate.

    Examples
    --------
    >>> def process(state, scale):
    ...     return {"x": state["x"] * scale, "y": state["y"] * scale}
    >>> flat_f, flatten_input, unflatten_output = flatten_fun(
    ...     process, {"x": 1.0, "y": 2.0}, 5.0
    ... )
    >>> flat_input = flatten_input({"x": 10.0, "y": 20.0}, 2.0)
    >>> flat_input
    [10.0, 20.0, 2.0]
    >>> flat_output = flat_f(flat_input)
    >>> flat_output
    [20.0, 40.0]
    >>> unflatten_output(flat_output)
    {'x': 20.0, 'y': 40.0}
    """

    out = f(*args)
    args_treedef = jax.tree_util.tree_structure(args, is_leaf=is_leaf)
    out_treedef = jax.tree_util.tree_structure(out, is_leaf=is_leaf)

    def flat_f(flat_args):
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


def dual_flatten[Leaf1, Leaf2](
    pytree1: PyTree[Leaf1], pytree2: PyTree[Leaf2], *, allow_prefix: bool = True
) -> tuple[list[Leaf1], list[Leaf2]]:
    """Flatten two PyTrees after broadcasting the first to the second.

    By default, ``pytree1`` may have the same structure as ``pytree2`` or a structure that is a
    prefix of it. Set ``allow_prefix=False`` to require identical structures. ``None`` is retained
    as a leaf.

    Parameters
    ----------
    pytree1
        The first PyTree.
    pytree2
        The second PyTree, whose structure determines the flattened result when prefix
        broadcasting is enabled.
    allow_prefix
        Whether leaves in ``pytree1`` may broadcast over corresponding subtrees in ``pytree2``.

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
    if allow_prefix:
        pytree1 = tree_map_recurse_at_leaf_with_none_as_leaf(lambda a, b: a, pytree1, pytree2)

    flat_tree1, tree1_treedef = tree_flatten_with_none_as_leaf(pytree1)
    flat_tree2, tree2_treedef = tree_flatten_with_none_as_leaf(pytree2)
    if tree1_treedef != tree2_treedef:
        raise ValueError("PyTrees must have the same structure")
    return flat_tree1, flat_tree2


def same_tree(
    x: PyTree,
    y: PyTree,
    is_leaf: Callable[[Any], bool] | None = None,
) -> bool:
    """Return whether two PyTrees have equal structures and leaves.

    Array leaves are compared by value regardless of whether they are NumPy or
    JAX arrays, and numeric dtype differences are ignored.

    Parameters
    ----------
    x
        First PyTree.
    y
        Second PyTree.
    is_leaf
        Optional PyTree leaf predicate.

    Examples
    --------
    >>> same_tree({'x': np.array([1, 2])}, {'x': jnp.array([1.0, 2.0])})
    True
    >>> same_tree([1, 2], (1, 2))
    False
    """
    treedef_x = jax.tree_util.tree_structure(x, is_leaf=is_leaf)
    treedef_y = jax.tree_util.tree_structure(y, is_leaf=is_leaf)
    if treedef_x != treedef_y:
        return False
    leaves_x = jax.tree_util.tree_leaves(x, is_leaf=is_leaf)
    leaves_y = jax.tree_util.tree_leaves(y, is_leaf=is_leaf)
    return all(_same(xi, yi) for xi, yi in zip(leaves_x, leaves_y))


def tree_allclose(a: PyTree, b: PyTree, **kwargs: Any) -> bool:
    """Return whether corresponding leaves of identical PyTrees are close.

    Each pair of leaves is compared with :func:`numpy.allclose`. A ``ValueError``
    is raised when the PyTree structures differ.

    Parameters
    ----------
    a
        First PyTree.
    b
        Second PyTree.
    kwargs
        Options forwarded to :func:`numpy.allclose`.

    Examples
    --------
    >>> tree1 = {'a': jnp.array([1.0, 2.0]), 'b': (jnp.array(3.0),)}
    >>> tree2 = {'a': jnp.array([1.000001, 2.0]), 'b': (jnp.array(3.0),)}
    >>> tree_allclose(tree1, tree2, atol=1e-5)
    True

    >>> tree3 = {'a': jnp.array([1.0, 2.5]), 'b': (jnp.array(3.0),)}
    >>> tree_allclose(tree1, tree3)
    False

    """
    struct_a = jax.tree_util.tree_structure(a)
    struct_b = jax.tree_util.tree_structure(b)
    if struct_a != struct_b:
        raise ValueError(
            "PyTree structures do not match.\n"
            f" a structure: {struct_a}\n"
            f" b structure: {struct_b}"
        )

    leaves_are_close = jax.tree_util.tree_map(lambda x, y: np.allclose(x, y, **kwargs), a, b)

    return all(jax.tree_util.tree_leaves(leaves_are_close))


def map_inside_tree(f: Callable[[PyTree], PyTree], tree: PyTree) -> PyTree:
    """Map a function over the leading axis of every array leaf in a PyTree.

    ``None`` leaves are passed unchanged to every call and retained unchanged in
    the result. All non-``None`` input leaves must have the same leading size.

    Parameters
    ----------
    f
        Function applied to each leading-axis slice of the PyTree.
    tree
        PyTree containing arrays and optional ``None`` leaves.

    Examples
    --------
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
    axis_size = _tree_map_preserve_none(
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
            my_tree = _tree_map_preserve_none(
                lambda n: n[i],
                tree,
            )
            my_rez = f(my_tree)
            rez.append(my_rez)
        rez = _tree_map_preserve_none(
            lambda *r: np.stack(r),
            *rez,
        )
        return rez


def assert_all_leaves_instance_of(
    tree: PyTree,
    type: type | tuple[type, ...],
    is_leaf: Callable[[Any], bool] | None = None,
) -> None:
    """Assert that every leaf in a PyTree is an instance of a class.

    Examples
    --------
    >>> assert_all_leaves_instance_of({'x': 1, 'y': [2, 3]}, int)
    """
    for node in jax.tree_util.tree_flatten(tree, is_leaf)[0]:
        assert isinstance(node, type)


def assert_all_leaves_instance_of_with_none(tree: PyTree, type: type | tuple[type, ...]) -> None:
    """Assert that every leaf, including ``None``, is an instance of a class.

    Examples
    --------
    >>> assert_all_leaves_instance_of_with_none([1, None], (int, type(None)))
    """
    assert_all_leaves_instance_of(tree, type, is_leaf_with_none)


def assert_is_sequence_of(seq: Sequence[Any], type: type | tuple[type, ...]) -> None:
    """Assert that an object is a sequence whose items share a class.

    Examples
    --------
    >>> assert_is_sequence_of([1, 2, 3], int)
    """
    assert isinstance(seq, Sequence)
    for x in seq:
        assert isinstance(x, type)


def _tree_map_preserve_none(f: Callable, tree: PyTree, *rest: PyTree):
    def new_f(*args):
        if all(arg is None for arg in args):
            return None
        elif not any(arg is None for arg in args):
            return f(*args)
        else:
            assert False, "only handle all None or zero None args"

    return tree_map_with_none_as_leaf(new_f, tree, *rest)


def _same(x, y):
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


# Array and observed-value utilities


def assimilate_vals(vars: PyTree, vals: PyTree) -> PyTree[jnp.ndarray]:
    """Convert observed values to arrays shaped like corresponding variables.

    The structure of ``vars`` determines whether a Python container in ``vals``
    represents a PyTree branch or one array-valued leaf. A ``ValueError`` is
    raised when the structures or leaf shapes cannot be matched.

    Parameters
    ----------
    vars
        PyTree whose leaves provide the required shapes.
    vals
        Values to convert to JAX arrays.

    Examples
    --------
    >>> variables = {'x': jnp.zeros((2,))}
    >>> values = assimilate_vals(variables, {'x': [1.0, 2.0]})
    >>> np.asarray(values['x']).tolist()
    [1.0, 2.0]
    """
    try:
        new_vals = jax.tree_util.tree_map(lambda var, val: jnp.array(val), vars, vals)
    except ValueError:
        vars_treedef = jax.tree_util.tree_structure(vars)
        vals_treedef = jax.tree_util.tree_structure(vals)
        raise ValueError(
            f"Not able to find common pytree structure for given vars and vals.\n"
            f"For given vars got: {_short_pytree_string(vars_treedef)}.\n"
            f"For given vals got: {_short_pytree_string(vals_treedef)}."
        )

    flat_vars, vars_treedef = jax.tree_util.tree_flatten(vars)
    flat_vals, vals_treedef = jax.tree_util.tree_flatten(new_vals)
    assert vars_treedef == vals_treedef, (
        "vars and vals must have same structure (after conversion to arrays). "
        f"({vars_treedef} vs {vals_treedef})"
    )
    for var, val in zip(flat_vars, flat_vals):
        if var.shape != val.shape:
            raise ValueError(
                f"given var {var} with shape {var.shape} does not match given val"
                f" {val} with shape {val.shape}."
            )
    return new_vals


def flatten_args(
    vars: PyTree,
    given_vars: PyTree = None,
    given_vals: PyTree = None,
) -> tuple[list[Any], list[Any], list[Any], Callable, Callable]:
    """Flatten requested variables and shape-checked observed variables/values.

    The two returned reconstruction functions rebuild the original ``vars`` and
    ``given_vars`` structures from flat sequences.

    Parameters
    ----------
    vars
        Variables whose values will be calculated.
    given_vars
        Variables whose values are observed.
    given_vals
        Observed values corresponding to ``given_vars``.

    Examples
    --------
    >>> variables = {'x': jnp.zeros(())}
    >>> given_variables = (jnp.zeros((2,)),)
    >>> result = flatten_args(variables, given_variables, ([1.0, 2.0],))
    >>> flat_vars, flat_given, flat_vals, unflatten_vars, unflatten_given = result
    >>> len(flat_vars), len(flat_given), np.asarray(flat_vals[0]).tolist()
    (1, 1, [1.0, 2.0])
    >>> list(unflatten_vars(flat_vars))
    ['x']
    """
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


def _short_pytree_string(treedef):
    "Get a string for a JAX PyTreeDef without printing PyTreeDef and scaring the noobs"
    s = str(treedef)
    return s


# Function signature inspection


def fixed_arity(fun: Callable) -> int:
    """Return the number of parameters in a fixed positional signature.

    A fixed positional signature may contain only positional-only and positional-or-keyword
    parameters, and none may have default values.

    Parameters
    ----------
    fun
        Callable whose signature to inspect.

    Returns
    -------
    int
        Number of parameters in ``fun``.

    Raises
    ------
    TypeError
        If ``fun`` has a default value, variadic parameter, or keyword-only parameter.

    Examples
    --------
    >>> def add(x, y):
    ...     return x + y
    >>> fixed_arity(add)
    2

    >>> def optional(x=0):
    ...     return x
    >>> fixed_arity(optional)
    Traceback (most recent call last):
    ...
    TypeError: optional has no fixed positional signature: 'x' has a default

    >>> def variadic(*xs):
    ...     return xs
    >>> fixed_arity(variadic)
    Traceback (most recent call last):
    ...
    TypeError: variadic has no fixed positional signature: 'xs' is not a fixed positional parameter
    """
    name = getattr(fun, "__name__", type(fun).__name__)
    parameters = tuple(inspect.signature(fun).parameters.values())

    for parameter in parameters:
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError(
                f"{name} has no fixed positional signature: "
                f"{parameter.name!r} is not a fixed positional parameter"
            )
        if parameter.default is not inspect.Parameter.empty:
            raise TypeError(
                f"{name} has no fixed positional signature: " f"{parameter.name!r} has a default"
            )

    return len(parameters)


def get_positional_args(target_func: Callable, *args: Any, **kwargs: Any) -> list[Any]:
    """Convert mixed arguments into positional order for a callable.

    Default values are included in the result. The returned list produces the
    same call for functions whose parameters can all be supplied positionally.

    Parameters
    ----------
    target_func
        Callable whose signature determines the argument order.
    args
        Positional arguments to bind.
    kwargs
        Keyword arguments to bind.

    Raises
    ------
    ValueError
        If ``target_func`` declares a keyword-only parameter.
    TypeError
        If the supplied arguments do not match the callable's signature.

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
    target_name = getattr(target_func, "__name__", type(target_func).__name__)

    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            raise ValueError(
                f"Function '{target_name}' has keyword-only arguments "
                f"({param.name}), which is not allowed by this transformer."
            )

    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    positional_args_for_target = []
    for param_name in sig.parameters:
        positional_args_for_target.append(bound_args.arguments[param_name])

    return positional_args_for_target


# Unicode DAG rendering


_CIRCLE = "○"  # a node
_VERT = "│"  # a line passing vertically
_HORIZ = "─"  # a line passing horizontally
_BRANCH_RIGHT = "├"  # a vertical line continues and also branches right
_TURN_RIGHT = "└"  # a vertical line ends by turning right
_BEND_DOWN = "┐"  # a rightward branch bends down toward the child row
_JOIN = "┴"  # a line from above joins a horizontal line
_CROSS = "┼"  # pure crossing: unrelated vertical
_REPEATED = "⁎"  # one dependency edge supplies multiple parent positions
_SUBSCRIPTS = "₀₁₂₃₄₅₆₇₈₉"
_RENDER_MODES = ("off", "on", "auto")
_RenderMode = Literal["off", "on", "auto"]


def dag_lines(
    nodes: Sequence[Any],
    parents: Sequence[Sequence[Any]],
    label_fn: Callable[[Any], str] = str,
    number_parents: _RenderMode = "auto",
) -> list[str]:
    """Return a Unicode DAG drawing as a list of lines.

    Parameters
    ----------
    nodes
        Distinct, hashable nodes in topological order.
    parents
        Parent nodes for each corresponding node.
    label_fn
        Function that produces a single-line label for each node.
    number_parents
        Whether to number no edges, all edges, or only edges whose parent order
        would otherwise be ambiguous. An annotated edge is marked ``⁎`` instead
        of a position when its parent occurs more than once.

    Examples
    --------
    >>> dag_lines(['A', 'B'], [[], ['A']])
    ['○ A', '│', '○ B']
    """
    if number_parents not in _RENDER_MODES:
        raise ValueError(f"number_parents must be one of {_RENDER_MODES}, not {number_parents!r}")

    parent_idx, raw_parent_idx, issues = _validate(nodes, parents)
    children = _children(parent_idx)
    col_of, ncols = _assign_columns(parent_idx, children)
    annotated = _annotated_rows(raw_parent_idx, col_of, number_parents)

    # A node's parent list only has to be numberable if it is numbered.
    for i, issue in enumerate(issues):
        if issue and annotated[i]:
            raise ValueError(issue)

    return _draw(
        nodes,
        parent_idx,
        raw_parent_idx,
        children,
        col_of,
        ncols,
        label_fn,
        annotated,
    )


def render_dag(
    nodes: Sequence[Any],
    parents: Sequence[Sequence[Any]],
    label_fn: Callable[[Any], str] = str,
    number_parents: _RenderMode = "auto",
) -> None:
    """Print a Unicode rendering of a DAG.

    Each node is plotted on its own text row. Consecutive node rows have a
    routing row between them, plus a separate numbering row when parent-order
    digits would otherwise collide with a bend. Each node occupies a text
    column, and the character immediately to its right is that column's routing
    lane.

    **Characters used:** ``○ │ ─ └ ├ ┐ ┴ ┼``, the space, the digits ``0``-``9``
    where edges are numbered, and ``⁎`` where one edge represents repeated
    occurrences of the same parent.

    **Invariants**

    - Lines only travel down and to the right, so no arrowheads are needed.
    - A node is never placed in a column left of any parent. Every edge
      therefore travels only down and right.
    - The leftmost parent turns directly onto the child's row: ``├`` if its
      trunk continues to later children, ``└`` if this is its last child.
    - A terminating interior parent joins the child's horizontal route directly
      with ``┴``.
    - A continuing interior parent branches right and down with ``├┐`` on the
      routing row. Its original trunk crosses the child's horizontal route at
      ``┼``, while the branch joins through ``┴`` in the adjacent routing lane.
    - ``┼`` always means a pure crossing; joins always use ``┴``.
    - A parent already in the child's column enters the ``○`` vertically.
    - Parent annotations appear only on connector rows, labels only on node
      rows.

    Placement
    ---------

    Columns are created on demand and numbered from 0. Nodes are placed one at
    a time in the given order. Place each node N in the lowest-numbered column
    k such that:

    1. k is at or right of the column of every parent of N;
    2. no node already in column k has an un-plotted child other than N.

    A new column at the far right satisfies both conditions, so placement
    never fails.

    Condition 2 also guarantees that no two parents of N share a column: the
    earlier of them would have had N un-plotted when the later was placed.

    Rendering
    ---------

    Column k occupies character position 2k, and position 2k+1 is its routing
    lane. Node rows are separated by one routing row. A numbered node that also
    needs a bend gets an additional straight numbering row before its routing
    row.

    A column is *live* from the row of its most recent node until the row of
    that node's last un-plotted child, inclusive; otherwise it is *dead*.
    Rule 2 guarantees a live column's pending line belongs to its most recent
    node and that only that node's sole remaining child may reuse the column.
    Hence a ``│`` entering a ``○`` from directly above is always an edge, and its
    absence is always the absence of one.

    A straight connector row contains ``│`` at position 2k for every live
    column k and spaces elsewhere. A bend row starts the same way, then replaces
    the ``│`` of each continuing interior parent with ``├`` and places ``┐`` in
    its routing lane.

    Node rows: let N sit at column c, and let m be the lowest column holding a
    parent of N. If N has no parents, or m == c, no horizontal is drawn;
    otherwise fill positions 2m+1 .. 2c-1 with ``─``. Then position 2k of each
    column k is::

        k == c                           ○
        k == m                           ├   line continues below
                                         └   this row is its last child
        m < k < c, terminating parent    ┴
        m < k < c, continuing parent     ┼   its branch joins at 2k+1 with ┴
        m < k < c, live, not a parent    ┼
        m < k < c, dead                  ─
        k < m or k > c                   │ if live, else space

    A continuing interior parent's edge to N does not join at its trunk. On the
    preceding bend row, ``├┐`` copies that edge into position 2k+1; the short
    vertical branch then joins the child's horizontal through ``┴``. The trunk
    itself passes through ``┼`` unchanged. Thus ``┼`` never conceals a merge.

    A line ends only at the row of its node's last child, so any column between
    m and c whose line ends on this row is necessarily a parent of N. That is
    why a terminating interior parent can use ``┴`` directly without ambiguity.

    Parent order
    ------------

    An annotation appears immediately right of a parent's vertical line. A
    subscript digit gives that parent's position in the child's ``parents``
    entry. ``⁎`` means the same parent occupies multiple positions, which a
    single dependency edge cannot distinguish. ``number_parents`` selects
    which nodes are annotated::

        'off'    none
        'on'     every node with at least one distinct parent
        'auto'   only nodes whose parent order or multiplicity would be lost

    The bare drawing loses information when the columns of the parents, read in
    listed order, are not strictly increasing. A decrease means parent order
    differs from left-to-right order; an equality means the same parent occurs
    again. ``auto`` annotates exactly these cases. An unannotated fan-in under
    ``auto`` is therefore the positive assertion that its parents are distinct
    and listed from left to right.

    Digits read left to right in *column* order, which need not be parent
    order: ``│₁│₀`` means the column-0 parent is listed second. Under ``auto``
    that mismatch or a repeated parent is the only reason annotations are
    present. For example, ``│⁎│₁`` means the left parent occurs more than once
    while the right parent occupies position 1. Consult the original parent
    list when the exact positions represented by ``⁎`` matter.

    The annotation is unambiguous:

    - At most one distinct parent of N occupies any column, so at most one
      annotation lands in each slot.
    - A parent's line is live at the row of every one of its children, so a
      digit always sits immediately right of the ``│`` it annotates.
    - If N has no bends, annotations use its sole connector row. If N has
      bends, a separate straight annotation row precedes the bend row, so an
      annotation never displaces ``┐``.
    - Annotations never make the drawing wider than the label column: the
      rightmost possible annotation remains one character left of the label.

    Labels
    ------

    Let W be the largest column index used. Every node row is padded with
    spaces to position 2W + 2 and the label follows, so all labels are
    vertically aligned one space right of the rightmost possible ``○``.
    Connector rows carry no trailing whitespace, node rows none after the
    label, and the output ends with a single newline.

    Parameters
    ----------
    nodes
        The nodes of the DAG in topological order (each node appears after all
        of its parents). Nodes must be distinct and hashable; they are
        compared by equality.
    parents
        One entry per node: the parents of that node, given as nodes.
        ``len(parents)`` must equal ``len(nodes)``, and every parent must appear
        earlier in ``nodes`` than the node it parents. Repeated entries are
        treated as one dependency edge. When that edge is annotated, ``⁎``
        indicates that it represents multiple parent positions.
    label_fn
        Maps a node to its label, which must be a single-line string
        (default: ``str``).
    number_parents
        When to annotate edges with their position in the child's parent list
        (default: ``'auto'``, which annotates only where the drawing would
        otherwise lose order or multiplicity). An annotated node must have at
        most ten parent positions.

    Raises
    ------
    ValueError
        If ``number_parents`` is not one of ``'off'``, ``'on'`` or ``'auto'``; if
        ``len(parents) != len(nodes)``; if ``nodes`` contains duplicates; or if a
        parent is missing from ``nodes`` or does not precede its child.
        Additionally, if a node that is *annotated* has more than ten parent
        positions. This is checked per node and only where annotations are
        actually emitted.

    See Also
    --------
    dag_lines : the same drawing returned as a list of lines.

    Notes
    -----
    Width is ``O(n)`` in the worst case (n mutually unrelated chains need n
    columns), and because labels are globally aligned, one wide drawing pushes
    every label right.

    The routing grammar deliberately avoids mixed-weight box-drawing glyphs.
    Every ``┼`` is a pure crossing, and every connection to a horizontal route
    is shown separately with ``┴``.

    Examples
    --------
    Except where noted, these use the default ``'auto'`` mode.

    >>> nodes = ["A"]
    >>> parents = [[]]
    >>> render_dag(nodes, parents)
    ○ A

    >>> nodes = ["A", "B", "C"]
    >>> parents = [[], ["A"], ["B"]]
    >>> render_dag(nodes, parents)
    ○ A
    │
    ○ B
    │
    ○ C

    >>> nodes = ["A", "B", "C"]
    >>> parents = [[], ["A"], ["A"]]
    >>> render_dag(nodes, parents)
    ○   A
    │
    ├─○ B
    │
    ○   C

    >>> nodes = ["A", "B", "C"]
    >>> parents = [[], [], ["A", "B"]]
    >>> render_dag(nodes, parents)
    ○   A
    │
    │ ○ B
    │ │
    └─○ C

    >>> nodes = ["A", "B", "C"]
    >>> parents = [[], ["A"], ["A", "B"]]
    >>> render_dag(nodes, parents)
    ○   A
    │
    ├─○ B
    │ │
    └─○ C

    The simplest DAG that ``auto`` annotates: D lists B, whose column is 1,
    before C, whose column is 0.

    >>> nodes = ["A", "B", "C", "D"]
    >>> parents = [[], ["A"], ["A"], ["B", "C"]]
    >>> render_dag(nodes, parents)
    ○   A
    │
    ├─○ B
    │ │
    ○ │ C
    │₁│₀
    └─○ D

    An interior parent that continues to a later child bends right before the
    child row. Its trunk then crosses through ``┼`` while its branch joins
    through ``┴``.

    >>> nodes = ["A", "B", "C", "D"]
    >>> parents = [[], ["A"], ["A", "B"], ["A", "B", "C"]]
    >>> render_dag(nodes, parents)
    ○     A
    │
    ├─○   B
    │ ├┐
    ├─┼┴○ C
    │ │ │
    └─┴─○ D

    A pure crossing: B's line passes under C's row without touching it.

    >>> nodes = ["A", "B", "C", "D", "E"]
    >>> parents = [[], [], ["A"], ["A"], ["B"]]
    >>> render_dag(nodes, parents)
    ○     A
    │
    │ ○   B
    │ │
    ├─┼─○ C
    │ │
    ○ │   D
      │
      ○   E

    The contrasting case: B is a parent of C and continues on to D.

    >>> nodes = ["A", "B", "C", "D"]
    >>> parents = [[], [], ["A", "B"], ["B"]]
    >>> render_dag(nodes, parents)
    ○     A
    │
    │ ○   B
    │ ├┐
    └─┼┴○ C
      │
      ○   D

    A fan-in from three roots, where every parent's line ends.

    >>> nodes = ["A", "B", "C", "D"]
    >>> parents = [[], [], [], ["A", "B", "C"]]
    >>> render_dag(nodes, parents)
    ○     A
    │
    │ ○   B
    │ │
    │ │ ○ C
    │ │ │
    └─┴─○ D

    The empty DAG prints nothing.

    >>> render_dag([], [])

    Labels come from ``label_fn``, and alignment follows the widest column.

    >>> render_dag([1, 2, 3], [[], [1], [1]], "node {}".format)
    ○   node 1
    │
    ├─○ node 2
    │
    ○   node 3

    **Parent order**

    Reversing D's parent list from the example above puts them in column
    order, and ``auto`` falls silent:

    >>> nodes = ["A", "B", "C", "D"]
    >>> parents = [[], ["A"], ["A"], ["C", "B"]]
    >>> render_dag(nodes, parents)
    ○   A
    │
    ├─○ B
    │ │
    ○ │ C
    │ │
    └─○ D

    Which is what ``'off'`` produces for *either* ordering, and is exactly the
    ambiguity ``auto`` exists to close:

    >>> render_dag(nodes, [[], ["A"], ["A"], ["B", "C"]], number_parents='off')
    ○   A
    │
    ├─○ B
    │ │
    ○ │ C
    │ │
    └─○ D

    ``'on'`` numbers every edge, including the sole edge of a single-parent
    node:

    >>> render_dag(["A", "B", "C"], [[], ["A"], ["B"]], number_parents='on')
    ○ A
    │₀
    ○ B
    │₀
    ○ C

    >>> nodes = ["A", "B", "C"]
    >>> parents = [[], [], ["A", "B"]]
    >>> render_dag(nodes, parents, number_parents='on')
    ○   A
    │
    │ ○ B
    │₀│₁
    └─○ C

    The widest case: digits fill every separator, still one short of the
    label column.

    >>> nodes = ["A", "B", "C", "D"]
    >>> parents = [[], ["A"], ["A", "B"], ["A", "B", "C"]]
    >>> render_dag(nodes, parents, number_parents='on')
    ○     A
    │₀
    ├─○   B
    │₀│₁
    │ ├┐
    ├─┼┴○ C
    │₀│₁│₂
    └─┴─○ D

    A crossed column gets no digit, since it is not a parent:

    >>> nodes = ["A", "B", "C", "D", "E"]
    >>> parents = [[], [], ["A"], ["A"], ["B"]]
    >>> render_dag(nodes, parents, number_parents='on')
    ○     A
    │
    │ ○   B
    │₀│
    ├─┼─○ C
    │₀│
    ○ │   D
      │₀
      ○   E

    Modes are spelled out; the former boolean is no longer accepted.

    >>> render_dag(["A"], [[]], number_parents=True)
    Traceback (most recent call last):
        ...
    ValueError: number_parents must be one of ('off', 'on', 'auto'), not True

    Eleven parents cannot be numbered with single digits. Under ``'auto'`` this
    DAG draws without complaint, since its parents are already in column
    order and no digits are needed.

    >>> nodes = list(range(12))
    >>> parents = [[] for _ in range(11)] + [list(range(11))]
    >>> render_dag(nodes, parents, number_parents='on')
    Traceback (most recent call last):
        ...
    ValueError: node 11 has 11 parents; numbering allows at most 10

    A repeated parent produces one dependency edge. ``auto`` and ``on`` mark
    that edge with ``⁎`` to show that it supplies multiple parent positions;
    ``off`` shows the bare dependency DAG.

    >>> render_dag(["A", "B"], [[], ["A", "A"]])
    ○ A
    │⁎
    ○ B

    >>> render_dag(["A", "B"], [[], ["A", "A"]], number_parents='off')
    ○ A
    │
    ○ B
    """
    for line in dag_lines(nodes, parents, label_fn, number_parents):
        print(line)


def _validate(nodes, parents):
    """Check the inputs.

    Returns both deduplicated and original parent-index lists, alongside one
    entry per node: `None`, or the reason that node's edges cannot be numbered.
    The reason is raised by the caller only if the node turns out to be
    annotated, so a drawing is never rejected over an annotation it does not
    make.
    """
    if len(parents) != len(nodes):
        raise ValueError(f"len(parents) == {len(parents)} != {len(nodes)} == len(nodes)")

    index = {}
    for i, node in enumerate(nodes):
        if node in index:
            raise ValueError(f"duplicate node: {node!r}")
        index[node] = i

    parent_idx = []
    raw_parent_idx = []
    issues = []
    for i, raw in enumerate(parents):
        issue = None
        if len(raw) > 10:  # every slot must fit in one digit
            issue = f"node {nodes[i]!r} has {len(raw)} parents; " f"numbering allows at most 10"

        seen = []
        raw_indices = []
        for p in raw:
            if p not in index:
                raise ValueError(f"unknown parent {p!r} of node {nodes[i]!r}")
            j = index[p]
            if j >= i:
                raise ValueError(f"parent {p!r} does not precede its child {nodes[i]!r}")
            raw_indices.append(j)
            if j in seen:
                continue  # repeated entries are a single edge
            seen.append(j)

        parent_idx.append(seen)
        raw_parent_idx.append(raw_indices)
        issues.append(issue)
    return parent_idx, raw_parent_idx, issues


def _children(parent_idx):
    """Invert the parent lists; each child list is in increasing index order."""
    children = [[] for _ in parent_idx]
    for i, ps in enumerate(parent_idx):
        for j in ps:
            children[j].append(i)
    return children


def _assign_columns(parent_idx, children):
    """Place every node, returning its column and the number of columns used.

    `pending[k]` holds the un-plotted children of the nodes in column k, so
    the two placement conditions read directly off it.
    """
    pending = []
    col_of = [0] * len(parent_idx)

    for i in range(len(parent_idx)):
        for waiting in pending:  # node i is being plotted now
            waiting.discard(i)

        # 1. at or right of every parent
        first = max((col_of[j] for j in parent_idx[i]), default=0)
        # 2. nothing else in the column is still waiting for a child
        k = next(
            (k for k in range(first, len(pending)) if not pending[k]),
            len(pending),
        )
        if k == len(pending):
            pending.append(set())

        col_of[i] = k
        pending[k].update(children[i])

    return col_of, len(pending)


def _annotated_rows(parent_idx, col_of, mode):
    """Decide whether each node's incoming edges need annotations."""
    if mode == "off":
        return [False] * len(parent_idx)

    annotated = []
    for ps in parent_idx:
        if not ps:
            flag = False  # no edges to annotate
        elif mode == "on":
            flag = True
        else:
            # Equal adjacent columns are repeated parents; decreasing columns
            # are reordered parents. Either loses information in a bare DAG.
            cols = [col_of[j] for j in ps]
            flag = any(b <= a for a, b in zip(cols, cols[1:]))
        annotated.append(flag)
    return annotated


def _parent_annotations(parent_idx, col_of):
    """Map parent columns to position digits or the repeated-parent marker."""
    annotations = {}
    for slot, parent in enumerate(parent_idx):
        column = col_of[parent]
        if column in annotations:
            annotations[column] = _REPEATED
        else:
            annotations[column] = _SUBSCRIPTS[slot]
    return annotations


def _connector_row(reach, node_index, label_at, parent_annotations=None):
    """Draw live trunks, optionally with parent-position annotations."""
    row = [" "] * label_at
    for column, last_row in enumerate(reach):
        if last_row >= node_index:
            row[2 * column] = _VERT

    if parent_annotations is not None:
        for column, annotation in parent_annotations.items():
            row[2 * column + 1] = annotation

    return "".join(row).rstrip()


def _bend_row(reach, node_index, label_at, bent_columns):
    """Draw the row where continuing interior parents branch right and down."""
    row = list(_connector_row(reach, node_index, label_at).ljust(label_at))
    for column in bent_columns:
        assert reach[column] > node_index
        row[2 * column] = _BRANCH_RIGHT
        row[2 * column + 1] = _BEND_DOWN
    return "".join(row).rstrip()


def _node_row(node, node_index, column, parent_slots, bent_columns, reach, ncols, label_fn):
    """Draw one node and every route that reaches or crosses its row."""
    leftmost_parent = min(parent_slots, default=column)
    width = 2 * ncols - 1
    label_at = 2 * ncols
    row = [" "] * width

    for position in range(2 * leftmost_parent + 1, 2 * column):
        row[position] = _HORIZ

    for current_column in range(ncols):
        if current_column == column:
            glyph = _CIRCLE
        elif leftmost_parent <= current_column < column:
            if current_column == leftmost_parent and current_column in parent_slots:
                glyph = _BRANCH_RIGHT if reach[current_column] > node_index else _TURN_RIGHT
            elif current_column in parent_slots and reach[current_column] == node_index:
                glyph = _JOIN
            elif reach[current_column] >= node_index:
                glyph = _CROSS
            else:
                glyph = _HORIZ
        else:
            glyph = _VERT if reach[current_column] >= node_index else " "
        row[2 * current_column] = glyph

    for bent_column in bent_columns:
        row[2 * bent_column + 1] = _JOIN

    label = label_fn(node)
    return ("".join(row).ljust(label_at) + label).rstrip()


def _draw(nodes, parent_idx, raw_parent_idx, children, col_of, ncols, label_fn, annotated):
    """Render placed nodes with fixed trunk columns and interleaved bend lanes."""
    if not nodes:
        return []

    last_child = [node_children[-1] if node_children else -1 for node_children in children]
    parent_columns = [{col_of[parent] for parent in parents} for parents in parent_idx]
    parent_annotations = [
        _parent_annotations(parents, col_of) if annotated[node] else None
        for node, parents in enumerate(raw_parent_idx)
    ]
    # reach[column] is the last node row reached by the live trunk in that column.
    reach = [-1] * ncols
    label_at = 2 * ncols

    lines = []
    for node_index, node in enumerate(nodes):
        column = col_of[node_index]
        columns = parent_columns[node_index]
        leftmost_parent = min(columns, default=column)
        bent_columns = {
            parent_column
            for parent_column in columns
            if leftmost_parent < parent_column < column and reach[parent_column] > node_index
        }

        if node_index > 0:
            if annotated[node_index] and bent_columns:
                lines.append(
                    _connector_row(
                        reach,
                        node_index,
                        label_at,
                        parent_annotations[node_index],
                    )
                )
                lines.append(_bend_row(reach, node_index, label_at, bent_columns))
            elif bent_columns:
                lines.append(_bend_row(reach, node_index, label_at, bent_columns))
            else:
                lines.append(
                    _connector_row(
                        reach,
                        node_index,
                        label_at,
                        parent_annotations[node_index],
                    )
                )

        lines.append(
            _node_row(
                node,
                node_index,
                column,
                columns,
                bent_columns,
                reach,
                ncols,
                label_fn,
            )
        )
        reach[column] = max(reach[column], last_child[node_index])

    return lines


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)
