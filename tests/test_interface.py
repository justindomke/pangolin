from pangolin.interface import *

from pangolin import inference_numpyro, calculate
from scipy import stats

calc = calculate.Calculate("numpyro")


def test_constant1():
    x = Constant(0)
    assert x.get_shape() == ()


def test_constant2():
    x = Constant([0, 1, 2])
    assert x.get_shape() == (3,)


def test_normal1():
    x = normal(0, 1)
    assert x.cond_dist == normal_scale
    assert x.shape == ()
    assert x.ndim == 0


def test_normal2():
    x = normal(0, scale=1)
    assert x.cond_dist == normal_scale
    assert x.shape == ()
    assert x.ndim == 0


def test_normal3():
    x = normal(0, prec=1)
    assert x.cond_dist == normal_prec
    assert x.shape == ()
    assert x.ndim == 0


def test_normal4():
    try:
        # should fail because can't provide both scale and prec
        x = normal(0, scale=1, prec=1)
        assert False
    except Exception as e:
        assert True


def test_tform1():
    x = normal(0, 1)
    y = x * x + x
    assert y.shape == ()
    assert y.ndim == 0


def test_tform2():
    x = normal(0, 1)
    y = normal(0, 1)
    z = x * y + y ** (y ** 2)
    assert z.shape == ()
    assert z.ndim == 0


def test_vmap_dummy_args1():
    a = makerv(np.ones((5, 3)))
    b = makerv(np.ones(5))
    (dummy_a, dummy_b), axis_size = vmap_dummy_args([0, 0], 5, a, b)
    assert dummy_a.shape == (3,)
    assert dummy_b.shape == ()
    assert axis_size == 5


def test_vmap_dummy_args2():
    a = makerv(np.ones((5, 3)))
    b = makerv(np.ones((3, 5)))
    (dummy_a, dummy_b), axis_size = vmap_dummy_args([0, 1], 5, a, b)
    assert dummy_a.shape == (3,)
    assert dummy_b.shape == (3,)
    assert axis_size == 5


def test_vmap_dummy_args3():
    a = makerv(np.ones((5, 3, 2)))
    b = makerv(np.ones((3, 1, 9)))
    (dummy_a, dummy_b), axis_size = vmap_dummy_args([1, None], None, a, b)
    assert dummy_a.shape == (5, 2)
    assert dummy_b.shape == (3, 1, 9)
    assert axis_size == 3


def test_vmap_generated_nodes1():
    # both inputs explicitly given
    a = AbstractRV(())
    b = AbstractRV(())
    f = lambda a, b: [normal(a, b)]
    nodes = vmap_generated_nodes(f, a, b)[0]
    assert len(nodes) == 1
    assert nodes[0].cond_dist == normal_scale


def test_vmap_generated_nodes2():
    # b captured as a closure
    a = AbstractRV(())
    b = AbstractRV(())
    f = lambda a: [normal(a, b)]
    nodes = vmap_generated_nodes(f, a)[0]
    assert len(nodes) == 1
    assert nodes[0].cond_dist == normal_scale


def test_vmap_generated_nodes3():
    # a captured as a closure
    a = AbstractRV(())
    b = AbstractRV(())
    f = lambda b: [normal(a, b)]
    nodes = vmap_generated_nodes(f, b)[0]
    assert len(nodes) == 1
    assert nodes[0].cond_dist == normal_scale


def test_vmap_generated_nodes4():
    # both a and b captured as a closure
    a = AbstractRV(())
    b = AbstractRV(())
    f = lambda: [normal(a, b)]
    nodes = vmap_generated_nodes(f)[0]
    assert len(nodes) == 1
    assert nodes[0].cond_dist == normal_scale


def test_vmap_generated_nodes5():
    def fun(a, b):
        loc = a + b
        scale = 1
        return [normal(loc, scale)]

    a = AbstractRV(())
    b = AbstractRV(())

    # both a and b given
    f = lambda a, b: fun(a, b)
    nodes = list(vmap_generated_nodes(f, a, b)[0])
    assert len(nodes) == 3
    assert nodes[0].cond_dist == add
    assert isinstance(nodes[1].cond_dist, Constant)
    assert nodes[2].cond_dist == normal_scale

    # b captured with closure
    f = lambda a: fun(a, b)
    nodes = list(vmap_generated_nodes(f, a)[0])
    assert len(nodes) == 3
    assert nodes[0].cond_dist == add
    assert isinstance(nodes[1].cond_dist, Constant)
    assert nodes[2].cond_dist == normal_scale

    # neither a nor b captured
    f = lambda: fun(a, b)
    nodes = list(vmap_generated_nodes(f)[0])
    assert len(nodes) == 3
    assert nodes[0].cond_dist == add
    assert isinstance(nodes[1].cond_dist, Constant)
    assert nodes[2].cond_dist == normal_scale


def test_vmap_eval1():
    "should fail because of incoherent axes sizes"
    try:
        y = vmap_eval(lambda loc, scale: normal_scale(loc, scale), [None, None], 5,
            np.zeros(3), np.ones(3), )
        assert False
    except AssertionError as e:
        assert True


def test_vmap_eval2():
    y = \
    vmap_eval(lambda loc, scale: [normal_scale(loc, scale)], [0, None], 3, np.zeros(3),
        1)[0]
    assert y.shape == (3,)


def test_vmap_eval3():
    def f(x):
        return [normal(x, x)]

    y = vmap_eval(f, [0], None, np.array([3.3, 4.4, 5.5]))[0]
    assert y.shape == (3,)


def test_vmap_eval4():
    def f(x):
        loc = x * 1.1
        scale = x ** 2.2
        return [normal(loc, scale)]

    y = vmap_eval(f, [0], None, np.array([3.3, 4.4, 5.5]))[0]
    assert y.shape == (3,)


def test_vmap_eval6():
    def f():
        return [normal(0, 1)]

    x = vmap_eval(f, [], 3)[0]
    assert x.shape == (3,)


def test_vmap_eval7():
    def f(x):
        loc = x * 1.1
        scale = x ** 2.2
        y = normal(loc, scale)
        x = normal(0, 1)
        z = normal(1, 2)
        return [y, x, z]

    y, x, z = vmap_eval(f, [0], 3, np.array([3.3, 4.4, 5.5]))
    assert y.shape == (3,)
    assert x.shape == (3,)
    assert z.shape == (3,)


def test_vmap_eval8():
    def f(x):
        loc = x * 1.1
        scale = x ** 2.2
        y = normal(loc, scale)
        x = normal(0, 1)
        z = normal(1, 2)
        return [y, x, z]

    y, x, z = vmap_eval(f, [0], None, np.array([3.3, 4.4, 5.5]))
    assert y.shape == (3,)
    assert x.shape == (3,)
    assert z.shape == (3,)


def test_vmap1():
    y = vmap(normal_scale, (0, None), 3)(np.zeros(3), np.array(1))
    assert y.shape == (3,)


def test_vmap2():
    def f(x):
        return normal(x, x)

    y = vmap(f, in_axes=0, axis_size=3)(np.array([3.3, 4.4, 5.5]))
    assert y.shape == (3,)


def test_vmap3():
    def f(x):
        loc = x * 1.1
        scale = x ** 2.2
        return normal(loc, scale)

    y = vmap(f, in_axes=0, axis_size=None)(np.array([3.3, 4.4, 5.5]))
    assert y.shape == (3,)


def test_vmap4():
    def f():
        return normal(np.array(1), np.array(2))

    y = vmap(f, in_axes=None, axis_size=3)()
    assert y.shape == (3,)


def test_vmap5():
    def f(loc, scale):
        x = normal(loc, scale)
        y = normal(x, scale)
        return (x, y)

    loc = np.array([1, 2, 3])
    scale = np.array([4, 5, 6])
    x, y = vmap(f, in_axes=(0, 0), axis_size=3)(loc, scale)
    assert x.shape == (3,)
    assert y.shape == (3,)


def test_vmap6():
    def f(loc, scale):
        x = normal(loc, scale)
        y = normal(x, scale)
        return (x, y)

    loc = np.array([1, 2, 3])
    scale = np.array([4, 5, 6])
    x, y = vmap(f, in_axes=(0, 0), axis_size=None)(loc, scale)
    assert x.shape == (3,)
    assert y.shape == (3,)


def test_vmap7():
    def f(loc, scale):
        x = normal(loc, scale)
        y = normal(x, scale)
        return (x, y)

    loc = np.array([1, 2, 3])
    scale = np.array([4, 5, 6])
    x, y = vmap(f, in_axes=0, axis_size=None)(loc, scale)
    assert x.shape == (3,)
    assert y.shape == (3,)


def test_vmap8():
    def f(stuff):
        x = stuff["x"]
        y, z = stuff["yz"]
        a = normal(x, 1)
        b = normal(a, y)
        c = normal(b, z)
        return ({"a": a}, b, c)

    stuff = {"x": 1.1, "yz": (2.2, 3.3)}
    out = vmap(f, in_axes=None, axis_size=5)(stuff)
    assert out[0]["a"].shape == (5,)
    assert out[1].shape == (5,)
    assert out[2].shape == (5,)


def test_vmap9():
    def f(stuff):
        x = stuff["x"]
        y, z = stuff["yz"]
        a = normal(x, 1)
        b = normal(a, y)
        c = normal(b, z)
        return ({"a": a}, b, c)

    stuff = {"x": 1.1, "yz": (2.2 * np.ones(5), 3.3)}
    # this doesn't work, for unclear jax reasons
    # in_axes = {'x': None, 'yz': (0, None)}
    # but this does
    in_axes = ({"x": None, "yz": (0, None)},)
    out = vmap(f, in_axes=in_axes, axis_size=5)(stuff)
    assert out[0]["a"].shape == (5,)
    assert out[1].shape == (5,)
    assert out[2].shape == (5,)


def test_vmap10():
    def f(x):
        loc = x * 1.1
        scale = x ** 2.2
        return {"y": normal(loc, scale)}

    stuff = vmap(f, 0, None)(np.array([3.3, 4.4, 5.5]))
    assert stuff["y"].shape == (3,)


def test_vmap11():
    def f(x):
        loc = x * 1.1
        scale = x ** 2.2
        y = normal(loc, scale)
        x = normal(0, 1)
        z = normal(1, 2)
        return {"y": y, "xz": (x, z)}

    stuff = vmap(f, 0, None)(np.array([3.3, 4.4, 5.5]))
    # fancy pattern matching
    match stuff:
        case {"y": y, "xz": (x, z)}:
            assert y.shape == (3,)
            assert x.shape == (3,)
            assert z.shape == (3,)
        case _:
            assert False, "should be impossible"


def test_vmap12():
    loc = 0.5

    def f(scale):
        return normal(loc, scale)

    x = vmap(f, 0, None)(np.array([2.2, 3.3, 4.4]))
    assert x.shape == (3,)


def test_vmap13():
    loc = 0.5
    scale = 1.3

    def f():
        return normal(loc, scale)

    x = vmap(f, None, 3)()
    assert x.shape == (3,)


def test_vmap14():
    x = normal(1.1, 2.2)
    y = vmap(lambda: normal(x, 1), None, 3)()
    assert y.shape == (3,)


def test_vmap15():
    x = normal(0, 1)
    y, z = vmap(
        lambda: (yi := normal(x, 2), zi := vmap(lambda: normal(yi, 3), None, 5)()),
        None, 3, )()


def test_plate1():
    x = normal(0, 1)
    y = plate(N=3)(lambda: normal(x, 1))
    assert x.shape == ()
    assert y.shape == (3,)


def test_plate2():
    x = normal(0, 1)
    y, z = plate(N=3)(
        lambda: (yi := normal(x, 1), zi := plate(N=5)(lambda: normal(yi, 1))))
    assert x.shape == ()
    assert y.shape == (3,)
    assert z.shape == (3, 5)


def test_plate3():
    x = normal(0, 1)
    y = plate(N=3)(lambda: normal(x, 1))
    z = plate(y, N=3)(lambda yi: plate(N=5)(lambda: normal(yi, 1)))
    assert x.shape == ()
    assert y.shape == (3,)
    assert z.shape == (3, 5)


def test_plate3a():
    x = normal(0, 1)
    y = plate(N=3)(lambda: normal(x, 1))
    z = plate(y)(lambda yi: plate(N=5)(lambda: normal(yi, 1)))
    assert x.shape == ()
    assert y.shape == (3,)
    assert z.shape == (3, 5)


def test_plate4():
    x = normal(0, 1)
    y = plate(N=3)(lambda: normal(x, 1))
    z = plate(y, N=3)(lambda yi: normal(yi, 1))
    assert x.shape == ()
    assert y.shape == (3,)
    assert z.shape == (3,)


def test_plate5():
    loc = np.array([2, 3, 4])
    scale = np.array([5, 6, 7])
    x = plate(loc, scale, N=3)(normal)
    assert x.shape == (3,)


def test_plate6():
    loc = np.array([2, 3, 4])
    scale = np.array(5)
    x = plate(loc, scale, N=3, in_axes=(0, None))(normal)
    assert x.shape == (3,)


def test_plate6a():
    "recommended implementation"
    loc = np.array([2, 3, 4])
    scale = np.array(5)
    x = plate(loc)(lambda loc_i: normal(loc_i, scale))
    assert x.shape == (3,)


def test_plate7():
    "not recommended but legal"  # recommended implmentation same as 6a
    loc = np.array([2, 3, 4])
    scale = np.array(5)
    x = plate(loc, scale, N=None, in_axes=(0, None))(normal)
    assert x.shape == (3,)


def test_plate8():
    loc = np.array(2)
    scale = np.array(5)
    x = plate(loc, scale, N=3, in_axes=(None, None))(normal)
    assert x.shape == (3,)


def test_plate8a():
    loc = np.array(2)
    scale = np.array(5)
    x = plate(N=3)(lambda: normal(loc, scale))
    assert x.shape == (3,)


def test_plate9():
    loc = np.array(2)
    scale = np.array(5)
    x = plate(loc, scale, N=3, in_axes=None)(normal)
    assert x.shape == (3,)


def test_plate10():
    loc = np.array(2)
    scale = np.array(5)
    x = plate(N=3)(lambda: normal(loc, scale))
    assert x.shape == (3,)


def test_plate11():
    x = plate(N=5)(lambda: normal(0, 1))
    y = plate(N=3)(lambda: normal(0, 1))
    z = plate(x)(lambda xi: plate(y)(lambda yi: normal(xi * yi, 1)))
    assert x.shape == (5,)
    assert y.shape == (3,)
    assert z.shape == (5, 3)


def test_plate12():
    stuff = plate(N=5)(lambda: {"x": normal(0, 1)})
    assert stuff["x"].shape == (5,)


def test_indexing1():
    # TODO: make this and all the following tests directly test against numpy functionality
    x = makerv([1, 2, 3])
    y = x[0]
    assert isinstance(y.cond_dist, Index)
    assert y.shape == ()


def test_indexing2():
    x = makerv([1, 2, 3])
    idx = makerv([0, 1])
    y = x[idx]
    assert isinstance(y.cond_dist, Index)
    assert y.shape == (2,)


def test_indexing3():
    x = makerv([1, 2, 3])
    idx = [0, 1]
    y = x[idx]
    assert isinstance(y.cond_dist, Index)
    assert y.shape == (2,)


def test_indexing4():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    idx = [0, 1, 1, 0]
    y = x[idx, :]
    assert isinstance(y.cond_dist, Index)
    assert y.shape == (4, 3)


def test_indexing5():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    idx = [[0, 1], [1, 1], [0, 0], [1, 0]]
    y = x[:, idx]
    assert isinstance(y.cond_dist, Index)
    assert y.shape == (2, 4, 2)


def test_indexing6():
    x = makerv([[1, 2, 3], [4, 5, 6]])
    idx = [0, 1, 1, 0]
    y = x[idx]
    assert isinstance(y.cond_dist, Index)
    assert y.shape == (4, 3)


# def test_indexing7():
#     x_numpy = np.random.randn(5, 6, 7, 8, 9)
#     y_numpy = x[:, [1, 2, 3]]


def test_eq_normal1():
    x = normal(0, 1)
    y = normal(0, 1)
    assert x != y
    assert x.cond_dist == y.cond_dist


def test_eq_normal2():
    x = normal(0, 1)
    y = normal(2, 3)
    assert x != y
    assert x.cond_dist == y.cond_dist


def test_eq_constant1():
    data = np.random.randn(3)
    c = Constant(data)
    d = Constant(data)
    assert c == d


def test_eq_constant2():
    data1 = np.random.randn(3)
    data2 = data1 + 0.0
    c = Constant(data1)
    d = Constant(data2)
    assert c == d


def test_eq_constant3():
    data1 = np.random.randn(3)
    data2 = data1 + 1e-9
    c = Constant(data1)
    d = Constant(data2)
    assert c != d


def test_eq_vmap1():
    d1 = VMapDist(normal_scale, in_axes=[0, 1], axis_size=5)
    d2 = VMapDist(normal_scale, in_axes=[0, 1], axis_size=5)
    assert d1 == d2


def test_eq_vmap2():
    d1 = VMapDist(normal_scale, in_axes=[0, 1], axis_size=5)
    d2 = VMapDist(normal_prec, in_axes=[0, 1], axis_size=5)
    assert d1 != d2


def test_eq_vmap3():
    d1 = VMapDist(normal_scale, in_axes=[0, 1], axis_size=5)
    d2 = VMapDist(normal_scale, in_axes=[1, 0], axis_size=5)
    assert d1 != d2


def test_eq_vmap4():
    d1 = VMapDist(normal_scale, in_axes=[0, 1], axis_size=5)
    d2 = VMapDist(normal_scale, in_axes=[0, 1], axis_size=4)
    assert d1 != d2


def test_eq_vmap5():
    d1 = VMapDist(normal_scale, in_axes=[0, 1], axis_size=5)
    d2 = VMapDist(d1, in_axes=[0, 1], axis_size=5)
    d3 = VMapDist(d1, in_axes=[0, 1], axis_size=5)
    assert d1 != d2
    assert d1 != d3
    assert d2 == d3


def test_eq_vmap6():
    d1 = VMapDist(normal_scale, in_axes=[0, 1], axis_size=5)
    d2 = VMapDist(normal_scale, in_axes=[0, 1], axis_size=5)
    d3 = VMapDist(d1, in_axes=[0, 1], axis_size=5)
    d4 = VMapDist(d2, in_axes=[0, 1], axis_size=5)
    assert d1 == d2
    assert d1 != d3
    assert d1 != d4
    assert d2 != d3
    assert d2 != d4
    assert d3 == d4


def test_eq_vmap7():
    d1 = VMapDist(normal_scale, in_axes=[0, 1], axis_size=5)
    d2 = VMapDist(normal_scale, in_axes=[0, 1], axis_size=5)
    d3 = VMapDist(d1, in_axes=[0, 1], axis_size=5)
    d4 = VMapDist(d2, in_axes=[0, 1], axis_size=4)
    assert d1 == d2
    assert d1 != d3
    assert d1 != d4
    assert d2 != d3
    assert d2 != d4
    assert d3 != d4


def test_eq_vmap8():
    d1 = VMapDist(normal_scale, in_axes=[0, 1])
    d2 = VMapDist(normal_scale, in_axes=[0, 1])
    d3 = VMapDist(d1, in_axes=[0, 1])
    d4 = VMapDist(d2, in_axes=[0, 1])
    assert d1 == d2
    assert d1 != d3
    assert d1 != d4
    assert d2 != d3
    assert d2 != d4
    assert d3 == d4


def test_log_prob1():
    x = makerv(1.0)
    l = log_prob(x)
    assert l is None


def test_log_prob2():
    x = normal(1.1, 2.2)
    l = log_prob(x)
    print_upstream(l)
    assert l.cond_dist == LogProb(normal_scale)
    assert l.parents[0].cond_dist == normal_scale
    assert l.parents[1].cond_dist == Constant(1.1)
    assert l.parents[2].cond_dist == Constant(2.2)


def test_log_prob3():
    # very simple model
    loc = 1.1
    scale = 2.2
    x = normal(loc, scale)

    l = log_prob(x)
    print_upstream(l)

    # sample with conditioning
    val = 3.3
    niter = 11
    ls = calc.sample(l, x, val, niter=niter)
    assert ls.shape == (niter,)
    expected = stats.norm.logpdf(val, loc, scale)
    assert np.allclose(ls, expected)

    # sample without conditioning
    niter = 10000
    ls, xs = calc.sample([l, x], niter=niter)
    expected = stats.norm.logpdf(xs, loc, scale)
    assert np.allclose(ls, expected)

    estimated_entropy = -np.mean(ls)
    expected_entropy = stats.norm.entropy(loc, scale)
    assert np.abs(estimated_entropy - expected_entropy) < 0.03

    # if you condition on x, then there is no probability to eval
    l = log_prob(x, x)
    assert l is None


def test_log_prob_joint():
    loc = 1.1
    scale_x = 2.2
    scale_y = 3.3
    x = normal(loc, scale_x)
    y = normal(x, scale_y)

    # test scalar log probs
    l = log_prob(x)  # should work (tested above)
    try:
        l = log_prob(y)  # should fail
        assert False, "failed to raise exception"
    except InvalidAncestorQuery as e:
        pass

    # get joint log prob
    l = log_prob([x, y])

    # sample with both x and y fixed
    val_x = 4.4
    val_y = 5.5
    niter = 11
    ls = calc.sample(l, [x, y], [val_x, val_y], niter=niter)
    expected = stats.norm.logpdf(val_x, loc, scale_x) + stats.norm.logpdf(val_y, val_x,
        scale_y)
    assert np.allclose(ls, expected)

    # sample with neither x nor y fixed
    [ls, xs, ys] = calc.sample([l, x, y], niter=niter)
    for li, xi, yi in zip(ls, xs, ys):
        expected = stats.norm.logpdf(xi, loc, scale_x) + stats.norm.logpdf(yi, xi,
            scale_y)
        assert np.allclose(li, expected)


def test_log_prob_branching():
    loc = 1.1
    scale = 2.2
    val_x = 3.3
    val_y = [4.4] * 10
    niter = 10

    x = normal(loc, scale)
    y = [normal(x, scale) for i in range(10)]
    l = log_prob(y, x)

    ls = calc.sample(l, (x, y), (val_x, val_y), niter=niter)
    assert ls.shape == (niter,)

    expected = 10 * stats.norm.logpdf(val_y[0], val_x, scale)

    assert np.allclose(ls, expected)


def test_vmap_log_prob():
    niter = 1000
    l = vmap(lambda: log_prob(normal(1.1, 2.2)), None, axis_size=5)()
    print_upstream(l)
    ls = calc.sample(l, niter=niter)
    assert ls.shape == (niter, 5)

    assert np.abs(-np.mean(ls) - stats.norm.entropy(1.1, 2.2)) < .03
