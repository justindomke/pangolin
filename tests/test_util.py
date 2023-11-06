from pangolin import dag, util
import jax


def test_varnames2():
    x = dag.Node()
    y = dag.Node()
    var_names = util.VarNames()
    a = var_names[x]
    assert a == 'v0v'
    b = var_names[y]
    assert b == 'v1v'
    c = var_names[x]
    assert c == 'v0v'


def test_tree_map_recurse_at_leaf():
    f = lambda a, b: a + b
    tree1 = [1, 2]
    tree2 = [3, [4, 5]]
    output = util.tree_map_recurse_at_leaf(f, tree1, tree2)
    expected = [4, [6, 7]]
    assert output == expected


def test_flatten_fun():
    def f(stuff):
        x = stuff['x']
        y = stuff['y']
        return (x, (y,), {'x': x, 'x2': x, "x+y": x + y})

    stuff = {'x': 2, 'y': 3}
    print(f"{stuff=}")
    flat_f, flatten_input, unflatten_output = util.flatten_fun(f, stuff)
    flat_stuff = flatten_input(stuff)
    print(f"{flat_stuff=}")
    flat_out = flat_f(*flat_stuff)
    print(f"{flat_out=}")
    out = unflatten_output(flat_out)

    expected = f(stuff)

    #print(f"{flat_f(flat_stuff)=}")
    print(f"{flat_out=}")
    print(f"{out=}")
    print(f"{expected=}")

    assert out == expected
