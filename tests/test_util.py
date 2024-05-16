from pangolin import dag, util
import numpy as np  # type: ignore
from jax import numpy as jnp  # type: ignore

def test_twoset1():
    s = util.TwoSet()
    s.add('a','b')
    s.add('a','c')
    s.add('b','c')
    assert set(s.items) == {('a','b'),('a','c'),('b','c')}
    s.clear_keys('a')
    assert set(s.items) == {('b','c')}

def test_twoset2():
    s = util.TwoSet()
    s.add('a','b')
    s.add('a','c')
    s.add('b','c')
    assert set(s.items) == {('a','b'),('a','c'),('b','c')}
    s.clear_keys('c')
    assert set(s.items) == {('a','b')}

def test_twoset3():
    s = util.TwoSet()
    s.add('a','b')
    s.add('a','c')
    s.add('b','c')
    assert ('a','b') in s
    assert ('b','a') in s
    assert ('c','b') in s
    s.clear_keys('c')
    assert set(s.items) == {('a','b')}
    assert ('a','b') in s
    assert ('b','a') in s
    assert ('b','c') not in s
    assert ('c','b') not in s


def test_twodict1():
    s = util.TwoDict()
    s['a','b'] = 1
    s['a','c'] = 2
    s['b','c'] = 3

    assert ('a','b') in s
    assert ('b','a') in s
    assert ('c','b') in s
    assert ('a','d') not in s
    assert ('d','a') not in s

    assert s['a', 'b'] == 1
    assert s['b', 'a'] == 1
    assert s['a', 'c'] == 2
    assert s['c', 'a'] == 2
    try:
        s['b','a'] = 4
        assert False
    except ValueError:
        pass

    try:
        tmp = s['a','d']
        assert False
    except KeyError:
        pass

    assert set(s.keys()) == {('a','b'),('a','c'),('b','c')}
    s.clear_keys('a')
    assert set(s.keys()) == {('b','c')}
    assert s['b','c'] == 3
    assert s['c', 'b'] == 3
    try:
        tmp = s['a','b']
        assert False
    except KeyError:
        pass

# def test_twoset2():
#     s = util.TwoSet()
#     s.add('a','b')
#     s.add('a','c')
#     s.add('b','c')
#     assert set(s.items) == {('a','b'),('a','c'),('b','c')}
#     s.clear_keys('c')
#     assert set(s.items) == {('a','b')}
#
# def test_twoset3():
#     s = util.TwoSet()
#     s.add('a','b')
#     s.add('a','c')
#     s.add('b','c')
#     assert ('a','b') in s
#     assert ('b','a') in s
#     assert ('c','b') in s
#     s.clear_keys('c')
#     assert set(s.items) == {('a','b')}
#     assert ('a','b') in s
#     assert ('b','a') in s
#     assert ('b','c') not in s
#     assert ('c','b') not in s


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


def test_flatten_fun():
    def f(stuff):
        x = stuff["x"]
        y = stuff["y"]
        return (x, (y,), {"x": x, "x2": x, "x+y": x + y})

    stuff = {"x": 2, "y": 3}
    print(f"{stuff=}")
    flat_f, flatten_input, unflatten_output = util.flatten_fun(f, stuff)
    flat_stuff = flatten_input(stuff)
    print(f"{flat_stuff=}")
    flat_out = flat_f(*flat_stuff)
    print(f"{flat_out=}")
    out = unflatten_output(flat_out)

    expected = f(stuff)

    # print(f"{flat_f(flat_stuff)=}")
    print(f"{flat_out=}")
    print(f"{out=}")
    print(f"{expected=}")

    assert out == expected


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


def test_more_specific_class1():
    assert util.most_specific_class(1, 2) == int
    assert util.most_specific_class(True, 2) == bool
    assert util.most_specific_class(3, True) == bool
    assert util.most_specific_class(3, True, 12) == bool
    assert util.most_specific_class(12) == int
    assert util.most_specific_class(True) == bool
