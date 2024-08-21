from pangolin.interface.vmap import vmap_subgraph, AbstractOp
from pangolin import ir
from pangolin.ir import RV, Constant
from pangolin.interface import OperatorRV, makerv


# vmap_subgraph(roots, dummy_roots, roots_axes, axis_size, dummy_nodes, dummy_outputs):


def test_simple_add():
    a = RV(Constant([1, 2, 3]))
    b = RV(Constant([4, 5, 6]))
    a_dummy = RV(AbstractOp((), False))
    b_dummy = RV(AbstractOp((), False))
    c_dummy = RV(ir.Add(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a, b], [a_dummy, b_dummy], [0, 0], 3, [c_dummy], [c_dummy])
    print(f"{c=}")
    assert c.op == ir.VMap(ir.Add(), (0, 0), 3)
    assert c.parents == (a, b)


def test_simple_add_non_abstract():
    a = RV(Constant([1, 2, 3]))
    b = RV(Constant([4, 5, 6]))
    a_dummy = RV(Constant(0))
    b_dummy = RV(Constant(0))
    c_dummy = RV(ir.Add(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a, b], [a_dummy, b_dummy], [0, 0], 3, [c_dummy], [c_dummy])
    print(f"{c=}")
    assert c.op == ir.VMap(ir.Add(), (0, 0), 3)
    assert c.parents == (a, b)


def test_simple_add_indices():
    a = OperatorRV(Constant([1, 2, 3]))
    b = OperatorRV(Constant([4, 5, 6]))
    a_dummy = a[0]
    b_dummy = b[0]
    c_dummy = RV(ir.Add(), a_dummy, b_dummy)
    [c] = vmap_subgraph([a, b], [a_dummy, b_dummy], [0, 0], 3, [c_dummy], [c_dummy])
    print(f"{c=}")
    assert c.op == ir.VMap(ir.Add(), (0, 0), 3)
    assert c.parents == (a, b)


def test_int_subclass():
    class SubInt(int):
        def __new__(cls, value, *args, **kwargs):
            return super(SubInt, cls).__new__(cls, value)

    x = SubInt(7)
    y = SubInt(9)
    print(f"{x=}")
    print(f"{y=}")
    x_plus_y = x + y
    print(f"{x_plus_y=}")
    assert isinstance(x_plus_y, int)
    assert not isinstance(x_plus_y, SubInt)

    assert isinstance(max(x, y), int)
    assert isinstance(max(x, y), SubInt)

    assert isinstance(sum([x, y]), int)
    assert not isinstance(sum([x, y]), SubInt)


def test_int_subclass2():
    class SubInt(int):
        def __new__(cls, value, *args, **kwargs):
            return super(SubInt, cls).__new__(cls, value)

        def __add__(self, other):
            return SubInt(int(self) + int(other))

        def __radd__(self, other):
            # needed because regular sum starts with 0
            return SubInt(int(self) + int(other))

    x = SubInt(7)
    y = SubInt(9)
    print(f"{x=}")
    print(f"{y=}")
    x_plus_y = x + y
    print(f"{x_plus_y=}")
    assert isinstance(x_plus_y, int)
    assert isinstance(x_plus_y, SubInt)

    assert isinstance(max(x, y), int)
    assert isinstance(max(x, y), SubInt)

    assert isinstance(sum([x, y]), int)
    assert isinstance(sum([x, y]), SubInt)


# consider code like this:
#
# x = LoopVar(10)
# with i = Loop(10):
#   x[i] = z[i]+y[i]
#
# How could this work? Here's one way:
# - i could be of a class that is a subclass of int, with value 0 (or should it be 9?)
# - So z[i] and y[i] are exactly the same as z[0] and y[0]
# - That is, they are Index RVs with index values represent what axis they are mapped over
# - Then x[i] should just basically remember that it is an output? It should also store that it is
#   "mapped by i"
# - When the context manager exits, it takes all the values that are mapped according to it as
#   outputs and does an upstream search to get the inputs
#
# So would become something like
# dummy_z = z[0]
# dummy_y = y[0]
# dummy_x = dummy_z + dummy_y
# [x] = vmap_subgraph([z, y], [dummy_z, dummy_y], [0, 0], 10, [dummy_x], [dummy_x])
#
# That seems *fairly* nice. But what if you do something like this?
#
# x = LoopVar(10)
# with i = Loop(10):
#   x[i] = z[i] + i/2
#
# I think this should be translated into
#
# i_range = RV(Constant(range(10)))
# dummy_z = z[0]
# dummy_i = i_range[0]
# dummy_tmp = dummy_i/2
# dummy_x = dummy_z + dummy_tmp
# [x] = vmap_subgraph([dummy_z, dummy_i], [z, i_range], [0, 0], 10, [dummy_tmp, dummy_x], [dummy_x])
#
# OK, but... Maybe we ONLY need a dummy i? Maybe that should be translated into
#
# i_range = RV(Constant(range(10)))
# zero = RV(Constant(0))
# dummy_i = RV(Index(None), i_range, zero)
# dummy_z = RV(Index(None), z, dummy_i)
# dummy_tmp = dummy_i/2
# dummy_x = dummy_z + dummy_tmp
# [x] = vmap_subgraph([dummy_i], [i_range], [0], 10, [dummy_z, dummy_tmp, dummy_i], [dummy_x])


# def test_loop_manual():
#     x = makerv([10, 20, 30])
#     myloop = Loop(3, auto_assign=False)
#     with myloop as i:
#         dummy_z = x[i]
#
#     assert isinstance(dummy_z.op, ir.Index)
#     assert dummy_z.parents == (x, myloop.i)
#     assert myloop.generated_rvs == [dummy_z]
#
#     [z] = vmap_subgraph([myloop.range], [myloop.i], [0], 3, [dummy_z], [dummy_z])
#
#     assert z.op == ir.VMap(ir.Index(None), (None, 0), 3)
#     assert z.parents == (x, myloop.range)
#
# def test_loop_manual_finalization():
#     x = makerv([10, 20, 30])
#     z = LoopVar()
#     myloop = Loop(3, auto_assign=False)
#     with myloop as i:
#         z[i] = x[i]
#
#     [new_z] = vmap_subgraph([myloop.range], [myloop.i], [0], 3, [z.dummy_rv], [z.dummy_rv])
#
#     z.finalize(new_z.op, *new_z.parents)
#
#     assert z.op == ir.VMap(ir.Index(None), (None, 0), 3)
#     assert z.parents == (x, myloop.range)
#
# def test_loop_auto_finalization1():
#     x = makerv([10, 20, 30])
#     z = LoopVar()
#     myloop = Loop(3)
#     with myloop as i:
#         z[i] = x[i]
#
#     assert z.op == ir.VMap(ir.Index(None), (None, 0), 3)
#     assert z.parents == (x, myloop.range)
#
# def test_loop_auto_finalization2():
#     x = makerv([10, 20, 30])
#     z = LoopVar()
#     myloop = Loop(3)
#     with myloop as i:
#         z[i] = x[i] + i
#
#     assert z.op == ir.VMap(ir.Add(), (0, 0), 3)
#     assert z.parents[0].op == ir.VMap(ir.Index(None), (None,0),3)
#     assert z.parents[1] == myloop.range
#     assert z.parents[0].parents == (x, myloop.range)


# def test_double_loop():
#     x = makerv([[1,2,3],[4,5,6]])
#     z = LoopVar()
#     myloop_i = Loop(2)
#     myloop_j = Loop(3)
#     with myloop_i as i:
#         z[i] = LoopVar()
#         with myloop_j as j:
#             z[i][j] = x[i]+x[j]

