from pangolin import util, dag
import numpy as np
from jax import numpy as jnp
import pytest


def test_public_api():
    assert util.__all__ == [
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


def test_collection_helpers():
    assert util.comma_separated([1, 2], fun=str, parens=False, spaces=True) == "1, 2"
    assert util.num_not_none(None, 0, False, None) == 2
    assert util.all_unique(["a", "b"])
    assert not util.all_unique(["a", "a"])
    assert util.intersects((1, 2), (2, 3))
    assert not util.intersects((1, 2), (3, 4))


def test_sequence_search_helpers():
    values = ["a", "b", "a", "c"]
    assert util.nth_index(values, "a", 0) == 0
    assert util.nth_index(values, "a", 1) == 2
    with pytest.raises(IndexError):
        util.nth_index(values, "a", 2)

    assert util.first_index(values, lambda value: value > "b") == 3
    with pytest.raises(Exception, match="none found"):
        util.first_index(values, lambda value: value > "z")

    assert util.first(iter(values), lambda value: value > "a") == "b"
    assert util.first(values, lambda value: value > "z", default="missing") == "missing"


def test_sequence_transformation_helpers():
    original = [1, 2, 3]
    assert util.swapped_list(original, 0, 2) == [3, 2, 1]
    assert original == [1, 2, 3]

    assert util.replace_in_sequence(original, 1, 9) == [1, 9, 3]
    assert util.replace_in_sequence((1, 2, 3), 1, 9) == (1, 9, 3)
    assert util.starts_with(original, [1, 2])
    assert util.starts_with(original, [])
    assert not util.starts_with(original, [2])


def test_dictionary_and_zip_helpers():
    assert util.reverse_dict({"a": 1, "b": 2}) == {1: "a", 2: "b"}
    assert util.reverse_dict({"a": 1, "b": 1}) == {1: "b"}
    assert tuple(util.unzip([(1, "a"), (2, "b")])) == ((1, 2), ("a", "b"))
    with pytest.raises(ValueError):
        tuple(util.unzip([(1, "a"), (2,)], strict=True))


def test_naming_and_type_helpers():
    assert [util.num2str(i) for i in (0, 25, 26, 676)] == ["a", "z", "ba", "baa"]
    assert util.camel_case_to_snake_case("NormalPrec") == "normal_prec"
    assert util.is_shape_tuple(())
    assert util.is_shape_tuple((2, 3))
    assert not util.is_shape_tuple([2, 3])
    assert not util.is_shape_tuple((2, 3.0))
    assert util.is_numeric_numpy_array(np.array([1, 2]))
    assert util.is_numeric_numpy_array(np.array([1.0, 2.0]))
    assert not util.is_numeric_numpy_array(np.array([True, False]))
    assert not util.is_numeric_numpy_array(np.array(["a", "b"]))


def test_varnames2():
    x = dag.Node()
    y = dag.Node()
    var_names = util.VarNames()
    a = var_names[x]
    assert a == "v0v"
    b = var_names[y]
    assert b == "v1v"
    c = var_names[x]
    assert c == "v0v"


def test_tree_map_recurse_at_leaf():
    f = lambda a, b: a + b
    tree1 = [1, 2]
    tree2 = [3, [4, 5]]
    output = util.tree_map_recurse_at_leaf(f, tree1, tree2)
    expected = [4, [6, 7]]
    assert output == expected


def test_none_aware_pytree_helpers():
    tree = {"x": [1, None], "y": 2}
    mapped = util.tree_map_with_none_as_leaf(
        lambda value: "missing" if value is None else value * 2, tree
    )
    assert mapped == {"x": [2, "missing"], "y": 4}

    leaves, treedef = util.tree_flatten_with_none_as_leaf(tree)
    assert leaves == [1, None, 2]
    assert treedef.unflatten(leaves) == tree

    path_leaves, _ = util.tree_flatten_with_path_with_none_as_leaf(tree)
    assert [("".join(str(key) for key in path), leaf) for path, leaf in path_leaves] == [
        ("['x'][0]", 1),
        ("['x'][1]", None),
        ("['y']", 2),
    ]


def test_dual_flatten_broadcasts_prefix_tree():
    flat_prefix, flat_tree = util.dual_flatten((None, 3), ("dog", ["cat", None]))
    assert flat_prefix == [None, 3, 3]
    assert flat_tree == ["dog", "cat", None]


def test_dual_flatten_can_require_identical_structures():
    with pytest.raises(ValueError, match="same structure"):
        util.dual_flatten((None, 3), ("dog", ["cat", None]), allow_prefix=False)


def test_flatten_fun():
    def f(stuff):
        x = stuff["x"]
        y = stuff["y"]
        return (x, (y,), {"x": x, "x2": x, "x+y": x + y})

    stuff = {"x": 2, "y": 3}
    # print(f"{stuff=}")
    flat_f, flatten_input, unflatten_output = util.flatten_fun(f, stuff)
    flat_stuff = flatten_input(stuff)
    # print(f"{flat_stuff=}")
    flat_out = flat_f(flat_stuff)
    # print(f"{flat_out=}")
    out = unflatten_output(flat_out)

    expected = f(stuff)

    # print(f"{flat_f(flat_stuff)=}")
    # print(f"{flat_out=}")
    # print(f"{out=}")
    # print(f"{expected=}")

    assert out == expected

    with pytest.raises(AssertionError, match="args don't match original"):
        flatten_input({"x": 2})


def test_same_tree1():
    tree1 = (1, 2, 3)
    tree2 = (1, 2, 3)
    tree3 = (1, 2, 4)
    assert util.same_tree(tree1, tree2)
    assert not util.same_tree(tree1, tree3)


def test_same_tree2():
    tree1 = (1, (2, 3))
    tree2 = ((1, 2), 3)
    tree3 = (1, 2, 3)
    assert not util.same_tree(tree1, tree2)
    assert not util.same_tree(tree1, tree3)
    assert not util.same_tree(tree2, tree3)


def test_same_tree3():
    tree1 = (1, 2, np.array([4, 5]))
    tree2 = (1, 2, np.array([4, 5]))
    tree3 = (1, 2, [4, 5])
    assert util.same_tree(tree1, tree2)
    assert not util.same_tree(tree1, tree3)


def test_same_tree4():
    tree1 = (1, 2, np.array([4, 5]))
    tree2 = (1, 2, None)
    tree3 = (1, 2)
    assert not util.same_tree(tree1, tree2)
    assert not util.same_tree(tree1, tree3)
    assert not util.same_tree(tree2, tree3)


def test_same_tree5():
    tree1 = (1, 2, None)
    tree2 = (1, 2, None)
    tree3 = (1, 2)
    assert util.same_tree(tree1, tree2)
    assert not util.same_tree(tree1, tree3)
    assert not util.same_tree(tree2, tree3)


def test_same_tree6():
    tree1 = (1, 2, None)
    tree2 = (1, None, 2)
    tree3 = (1, 2)
    assert not util.same_tree(tree1, tree2)
    assert not util.same_tree(tree1, tree3)
    assert not util.same_tree(tree2, tree3)


def test_same_tree7():
    tree1 = (1, 2, None)
    tree2 = (1, None, 2)
    tree3 = (1, 2)
    assert not util.same_tree(tree1, tree2)
    assert not util.same_tree(tree1, tree3)
    assert not util.same_tree(tree2, tree3)


def test_same_tree8():
    tree1 = (1, 2, [3, 4])
    tree2 = (1, 2, np.array([3, 4]))
    tree3 = (1, 2, jnp.array([3, 4]))
    tree4 = (1, 2, jnp.array([3.0, 4.0]))
    tree5 = (1, 2, np.array([3.0, 4.0]))
    assert not util.same_tree(tree1, tree2)
    assert not util.same_tree(tree1, tree3)
    assert util.same_tree(tree2, tree3)
    assert util.same_tree(tree2, tree4)
    assert util.same_tree(tree2, tree5)
    assert util.same_tree(tree3, tree4)
    assert util.same_tree(tree3, tree5)
    assert util.same_tree(tree4, tree5)


def test_map_inside_tree1():
    def f(t):
        a, (b, c) = t
        return (a + b, a * b), c

    tree = np.array([1, 2]), (np.array([3, 4]), np.array([5, 6]))
    rez = util.map_inside_tree(f, tree)
    expected = (np.array([4, 6]), np.array([3, 8])), np.array([5, 6])
    assert util.same_tree(rez, expected)

    tree = np.array([1, 2]), (np.array([3, 4]), None)
    rez = util.map_inside_tree(f, tree)
    expected = (np.array([4, 6]), np.array([3, 8])), None
    assert util.same_tree(rez, expected)


def test_map_inside_tree2():
    def f(t):
        a, (b, c) = t
        return (a + b, c), None

    tree = np.array([1, 2]), (np.array([3, 4]), np.array([5, 6]))
    rez = util.map_inside_tree(f, tree)
    expected = (np.array([4, 6]), np.array([5, 6])), None
    assert util.same_tree(rez, expected)

    tree = np.array([1, 2]), (np.array([3, 4]), None)
    rez = util.map_inside_tree(f, tree)
    expected = (np.array([4, 6]), None), None
    assert util.same_tree(rez, expected)


def test_tree_allclose():
    first = {"x": np.array([1.0, 2.0]), "y": (3.0,)}
    close = {"x": jnp.array([1.0, 2.00001]), "y": (3.0,)}
    far = {"x": jnp.array([1.0, 2.1]), "y": (3.0,)}
    assert util.tree_allclose(first, close, atol=1e-4)
    assert not util.tree_allclose(first, far, atol=1e-4)
    with pytest.raises(ValueError, match="PyTree structures do not match"):
        util.tree_allclose(first, {"x": close["x"]})


def test_pytree_assertion_helpers():
    util.assert_all_leaves_instance_of({"x": [1, 2]}, int)
    with pytest.raises(AssertionError):
        util.assert_all_leaves_instance_of({"x": [1, "two"]}, int)

    util.assert_all_leaves_instance_of_with_none([1, None], (int, type(None)))
    with pytest.raises(AssertionError):
        util.assert_all_leaves_instance_of_with_none([1, None], int)

    util.assert_is_sequence_of((1, 2), int)
    with pytest.raises(AssertionError):
        util.assert_is_sequence_of((1, "two"), int)


def test_assimilate_vals():
    variables = {"x": jnp.zeros((2,)), "y": jnp.zeros(())}
    values = util.assimilate_vals(variables, {"x": [1, 2], "y": 3})
    assert np.array_equal(values["x"], np.array([1, 2]))
    assert np.array_equal(values["y"], np.array(3))

    with pytest.raises(ValueError, match="common pytree structure"):
        util.assimilate_vals(variables, [1, 2])
    with pytest.raises(ValueError, match="does not match given val"):
        util.assimilate_vals(variables, {"x": [1], "y": 3})


def test_flatten_args():
    variables = {"x": jnp.zeros(()), "y": jnp.ones(())}
    given_variables = (jnp.zeros((2,)),)
    result = util.flatten_args(variables, given_variables, ([1.0, 2.0],))
    flat_vars, flat_given, flat_vals, unflatten_vars, unflatten_given = result

    assert len(flat_vars) == 2
    assert len(flat_given) == 1
    assert np.array_equal(flat_vals[0], np.array([1.0, 2.0]))
    assert util.same_tree(unflatten_vars(flat_vars), variables)
    assert util.same_tree(unflatten_given(flat_given), given_variables)


def test_get_positional_args():
    def example(a, b=2, c=3):
        return a + b + c

    positional = util.get_positional_args(example, 1, c=4)
    assert positional == [1, 2, 4]
    assert example(1, c=4) == example(*positional)

    with pytest.raises(TypeError):
        util.get_positional_args(example)


def test_get_positional_args_accepts_callable_objects():
    class CallableWithKeywordOnly:
        def __call__(self, value, *, scale=1):
            return value * scale

    with pytest.raises(ValueError, match="Function 'CallableWithKeywordOnly'"):
        util.get_positional_args(CallableWithKeywordOnly(), 2)


def test_more_specific_class1():
    assert util.most_specific_class(1, 2) == int
    assert util.most_specific_class(True, 2) == bool
    assert util.most_specific_class(3, True) == bool
    assert util.most_specific_class(3, True, 12) == bool
    assert util.most_specific_class(12) == int
    assert util.most_specific_class(True) == bool

    with pytest.raises(ValueError, match="no single most-specific argument type"):
        util.most_specific_class(1, "two")


def test_write_once_dict():
    d = util.WriteOnceDict()
    d["key"] = "value"
    assert d["key"] == "value"
    with pytest.raises(ValueError, match="Cannot overwrite existing key key"):
        d["key"] = "replacement"


def test_write_once_default_dict():
    calls = []

    def make_value(key):
        calls.append(key)
        return 2 * key

    d = util.WriteOnceDefaultDict(make_value)
    d[2] = 5
    assert d[2] == 5
    assert 2 in d
    assert 5 not in d
    assert d[5] == 10
    assert d[5] == 10
    assert calls == [5]
    with pytest.raises(ValueError):
        d[2] = 6
