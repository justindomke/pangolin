from pangolin import ir
from pangolin.functional.tracing import trace_call


from pangolin.interface import cauchy
from pangolin.interface import constant
from pangolin.interface import normal


def test_trace_generated_nodes1():
    def fun(x, y):
        return [x * 2, y + 3]

    x0 = constant(1)
    y0 = constant(2)
    _, generated, _ = trace_call(fun, x0, y0)
    assert len(generated) == 4
    assert generated[0].op == ir.Constant(2)
    assert generated[1].op == ir.Mul()
    assert generated[1].parents == (x0, generated[0])
    assert generated[2].op == ir.Constant(3)
    assert generated[3].op == ir.Add()
    assert generated[3].parents[0] == y0
    assert generated[3].parents[1].op == ir.Constant(3)


def test_trace_generated_nodes2():
    def fun(x, y):
        tmp = normal(x, y)
        z = cauchy(tmp, y)
        return [z]

    x0 = constant(1)
    y0 = constant(2)
    out, generated, _ = trace_call(fun, x0, y0)
    assert len(generated) == 2
    assert generated[0].op == ir.Normal()
    assert generated[0].parents == (x0, y0)
    assert generated[1].op == ir.Cauchy()
    assert generated[1].parents == (generated[0], y0)
    assert out[0] == generated[1]


def test_trace_generated_nodes_closure():
    x = constant(1)

    def fun(y):
        tmp = x * 3
        z = y + tmp
        return [z]

    y0 = constant(2)
    out, generated, _ = trace_call(fun, y0)
    assert len(generated) == 3
    c, tmp, z = generated
    assert c.op == ir.Constant(3)
    assert tmp.op == ir.Mul()
    assert tmp.parents == (x, c)
    assert z.op == ir.Add()
    assert z.parents == (y0, tmp)
    assert out == [z]


def test_trace_generated_nodes_ignored_input():
    x = constant(1)

    def fun(y):
        return [2 * x]

    y0 = constant(3)
    out, generated, _ = trace_call(fun, y0)
    c, z = generated
    assert c.op == ir.Constant(2)
    assert z.op == ir.Mul()
    assert z.parents == (c, x)
    assert out == [z]


def test_trace_vmapped_nodes1():
    a = constant(0)
    b = constant(1)
    _, generated, _ = trace_call(lambda a, b: [normal(a, b)], a, b)
    assert len(generated) == 1
    assert generated[0].op == ir.Normal()


def test_trace_vmapped_nodes2():
    a = constant(0)
    b = constant(1)
    _, generated, _ = trace_call(lambda a: [normal(a, b)], a)
    assert len(generated) == 1
    assert generated[0].op == ir.Normal()


def test_trace_vmapped_nodes3():
    a = constant(0)
    b = constant(1)
    _, generated, _ = trace_call(lambda b: [normal(a, b)], b)
    assert len(generated) == 1
    assert generated[0].op == ir.Normal()


def test_trace_vmapped_nodes4():
    a = constant(0)
    b = constant(1)
    _, generated, _ = trace_call(lambda: [normal(a, b)])
    assert len(generated) == 1
    assert generated[0].op == ir.Normal()
    assert generated[0].parents == (a, b)


def test_trace_vmapped_nodes5():
    def fun(a, b):
        loc = a + b
        scale = 1
        return [normal(loc, scale)]

    a = constant(0)
    b = constant(1)

    for traced_fun, args in (
        (fun, (a, b)),
        (lambda a: fun(a, b), (a,)),
        (lambda: fun(a, b), ()),
    ):
        nodes = trace_call(traced_fun, *args)[1]
        assert len(nodes) == 3
        loc, scale, z = nodes
        assert loc.op == ir.Add()
        assert loc.parents == (a, b)
        assert scale.op == ir.Constant(1)
        assert z.op == ir.Normal()
        assert z.parents == (loc, scale)
