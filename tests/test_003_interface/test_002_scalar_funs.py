from pangolin.interface import normal, makerv, InfixRV, add, mul, div
from pangolin import ir


def test_wrong_number_of_args():
    for args in [(), (0,), (0, 1, 2), (0, 1, 2, 3)]:
        try:
            x = normal(*args)
            assert False
        # except TypeError as e:
        #     print(e)
        #     assert str(e) == f"Normal op got {len(args)} arguments but expected 2."
        except TypeError as e:  # wrong number of args
            if len(args) == 0:
                assert (
                    str(e)
                    == f"normal() missing 2 required positional arguments: 'mu' and 'sigma'"
                )
            elif len(args) == 1:
                assert (
                    str(e)
                    == f"normal() missing 1 required positional argument: 'sigma'"
                )
            else:
                assert (
                    str(e)
                    == f"normal() takes 2 positional arguments but {len(args)} were given"
                )


def test_normal_rv_rv():
    x = makerv(0)
    y = makerv(1)
    z = normal(x, y)
    assert z.op == ir.Normal()
    assert z.parents == (x, y)
    assert isinstance(z, InfixRV)


def test_normal_rv_scalar():
    x = makerv(0)
    y = 1
    z = normal(x, y)
    assert z.op == ir.Normal()
    assert z.parents[0] == x
    assert z.parents[1].op == ir.Constant(1)
    assert isinstance(z, InfixRV)


def test_normal_scalar_rv():
    x = 0
    y = makerv(1)
    z = normal(x, y)
    assert z.op == ir.Normal()
    assert z.parents[0].op == ir.Constant(0)
    assert z.parents[1] == y
    assert isinstance(z, InfixRV)


def test_normal_scalar_scalar():
    x = 0
    y = 1
    z = normal(x, y)
    assert z.op == ir.Normal()
    assert z.parents[0].op == ir.Constant(0)
    assert z.parents[1].op == ir.Constant(1)
    assert isinstance(z, InfixRV)


def test_add():
    x = 0
    y = 1
    z = add(x, y)
    assert z.op == ir.Add()
    assert z.parents[0].op == ir.Constant(0)
    assert z.parents[1].op == ir.Constant(1)
    assert isinstance(z, InfixRV)


def test_math():
    x = 0
    y = 1
    z = add(mul(x, y), div(x, y))
    assert z.op == ir.Add()
    assert z.parents[0].op == ir.Mul()
    assert z.parents[1].op == ir.Div()
    assert z.parents[0].parents[0].op == ir.Constant(0)
    assert z.parents[0].parents[1].op == ir.Constant(1)
    assert z.parents[1].parents[0].op == ir.Constant(0)
    assert z.parents[1].parents[1].op == ir.Constant(1)
    assert isinstance(z, InfixRV)

    x = makerv(0)
    y = makerv(1)
    z = add(mul(x, y), div(x, y))
    assert z.op == ir.Add()
    assert z.parents[0].op == ir.Mul()
    assert z.parents[1].op == ir.Div()
    assert z.parents[0].parents[0].op == ir.Constant(0)
    assert z.parents[0].parents[1].op == ir.Constant(1)
    assert z.parents[1].parents[0].op == ir.Constant(0)
    assert z.parents[1].parents[1].op == ir.Constant(1)
    assert isinstance(z, InfixRV)
    # these weren't true before because x and y just passed as ints, do different RVs created
    assert z.parents[0].parents[0] == z.parents[1].parents[0]
    assert z.parents[0].parents[1] == z.parents[1].parents[1]
