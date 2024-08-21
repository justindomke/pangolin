"""WARNING: The loops module is highly experimental.
"""

from pangolin.ir import RV, Op, VMap, Constant, Index
from typing import Sequence, List, Self
from pangolin import util
from pangolin.interface import makerv, rv_factories, rv_factory, makerv_funs, OperatorRV
from pangolin.interface.vmap import AbstractOp
import numpy as np
from pangolin.interface.index import index_funs, standard_index_fun, eliminate_ellipses, pad_with_slices
from pangolin.interface import index

def looped_index(var, *idx):
    idx = eliminate_ellipses(var.ndim, idx)
    idx = pad_with_slices(var.ndim, idx)

    if any(isinstance(p, Loop) for p in idx):
        for p in idx:
            assert isinstance(p, Loop) or p == slice(
                None
            ), "can only mix Loop with full slice"
        return slice_existing_rv(var, idx, Loop.loops)

    return standard_index_fun(var,*idx)


def rv_maker(op: Op, *parents: OperatorRV):
    return make_sliced_rv(op, *parents, all_loops=Loop.loops)

def loop_makerv(x):
    if isinstance(x,Loop):
      return rv_factory(Constant(np.arange(x.length)))[x]
    if isinstance(x, RV):
        assert isinstance(x,OperatorRV)
        return x
    else:
        return rv_factory(Constant(x))

class Loop:
    """
    A Loop is a simple context manager that remembers what Loops have been created in what order.
    """

    loops: List[Self] = []
    "list of all Loops currently active as context managers."
    length: int | None

    def __init__(self, length=None):
        """
        Create a new Loop.

        Attributes
        ----------
        length:int | None
            the length for the loop (optional—can be None if loop length can be inferred from shapes)
        """
        self.length = length

    def __enter__(self, length=None):
        Loop.loops.append(self)
        rv_factories.append(rv_maker)
        index_funs.append(looped_index)
        makerv_funs.append(loop_makerv)

        return self

    # def __index__(self):
    #     range_rv = current_rv_class()(Constant(np.arange(self.length)))
    #     return SlicedRV(Index(None), range_rv, range_rv, full_rv=range_rv, loops=[range_rv], \
    #                                                                        loop_axes=[0])

    def __exit__(self, exc_type, exc_value, exc_tb):
        assert makerv_funs.pop() is loop_makerv
        assert index_funs.pop() is looped_index
        assert rv_factories.pop() is rv_maker
        assert Loop.loops.pop() is self

    def __add__(self, other):
        return makerv(self) + other

    def __radd__(self, other):
        return other + makerv(self)

    def __sub__(self, b):
        return makerv(self) - b

    def __rsub__(self, b):
        return makerv(b) - self

    def __mul__(self, other):
        return makerv(self) * other

    __rmul__ = __mul__

    def __truediv__(self, b):
        return makerv(self) / b

    def __rtruediv__(self, b):
        return makerv(b) / self

    def __pow__(self, b):
        return makerv(self) ** b

    def __rpow__(self, a):
        return makerv(a) ** self

    # def __matmul__(self, a):
    #     return matmul(self, a)
    #
    # def __rmatmul__(self, a):
    #     return matmul(a, self)

def remove_indices(sequence, indices):
    return type(sequence)(x for (n, x) in enumerate(sequence) if n not in indices)


# TODO: OperatorRV?
class SlicedRV(OperatorRV):
    """Represents a "slice" of a higher dimensional RV. Has the shape of just a single slice but has a reference to the
    full RV from which the slice was taken, and what Loops were applied to what axes to get the slice
    """

    def __init__(self, op, *parents, full_rv, loops, loop_axes):
        """
        Create a SlicedRV.

        Parameters
        ----------
        op: Op
            The op for the slice.
        parents: Sequence[RV]
            The parents for the slice.
        full_rv: RV
            The full (non-sliced) RV that the SlicedRV represents a slice of
        loops: Sequence[Loop]
            The Loops that have been applied to the full RV to get the slice
        loop_axes: Sequence[int]
            The axes along which the loops have been applied. The shape for the SlicedRV is the shape of full_rv except
            with the components from loop_axes deleted.
        """

        self.full_rv = full_rv
        self.loops = tuple(loops)
        self.loop_axes = tuple(loop_axes)
        self._loop2axis = dict(zip(loops, loop_axes))
        # TODO: check loop axes work with the full_rv
        super().__init__(op, *parents)
        assert self.shape == remove_indices(full_rv.shape, loop_axes)


def slice_existing_rv(full_rv, idx, all_loops) -> SlicedRV:
    """
    Given a full RV and a sequence of indices, create a sliced RV that
    "slices into" the full RV

    Parameters
    ----------
    full_rv: RV
        The full (unsliced) RV
    idx: Sequence[Loop | slice]
        Sequence with length len(full.rv.shape), where each entry is either a Loop
        or slice(None). The dimensions with slice(None) stay in the SlicedRV, the
        others are hidden.
    all_loops: Sequence[Loop]
        Context of all loops that are relevent to the new SlicedRV. Must contain
        all Loop objects that appear in idx. Is used solely to determine the order
        of the Loops in the new SlicedRV.

    Returns
    -------
    sliced_rv: SlicedRV
        New SlicedRV slicing into full_rv
    """

    assert isinstance(idx, Sequence)
    assert len(idx) == len(full_rv.shape)
    shape = full_rv.shape
    loops = [loop for loop in idx if isinstance(loop, Loop)]
    loops = sorted(loops, key=lambda loop: all_loops.index(loop))

    loop_axes = [-1] * len(loops)
    my_shape = []
    for n, (s, i) in enumerate(zip(shape, idx)):
        if isinstance(i, Loop):
            assert loop_axes[loops.index(i)] == -1
            loop_axes[loops.index(i)] = n
        else:
            assert i == slice(None, None, None), "with Loops slices must be full"
            my_shape.append(s)

    assert all(i >= 0 for i in loop_axes)

    my_shape = tuple(my_shape)
    my_random = full_rv.op.random

    op = AbstractOp(my_shape, my_random)
    parents = []

    return SlicedRV(
        op,
        *parents,
        full_rv=full_rv,
        loops=loops,
        loop_axes=loop_axes,
    )


def _loop2axis(var, loop, all_loops):
    assert isinstance(loop, Loop)

    if not isinstance(var, SlicedRV):
        return None
    elif loop not in var.loops:
        return None
    else:
        assert isinstance(loop, Loop)
        assert loop in var.loops

        where_loop = var.loops.index(loop)
        original_axis = var.loop_axes[where_loop]
        assert original_axis is not None

        future_loops = all_loops[: all_loops.index(loop)]

        axes_to_come = 0
        for future_loop in future_loops:
            if future_loop in var.loops:
                future_axis = var._loop2axis[future_loop]
                assert isinstance(future_axis, int)

                if future_axis < original_axis:
                    axes_to_come += 1

        correct_axis = original_axis - axes_to_come
        return correct_axis


def make_sliced_rv(op: Op, *parents: OperatorRV, all_loops: list[Loop]) -> OperatorRV:
    """
    Create a new SlicedRV from an Op, a set of parents, and a list of loops.
    Automatically creates a corresponding non-sliced RV.

    Parameters
    ----------
    op: Op
        The op distribution to apply to create the new SlicedRV
    *parents: tuple[OperatorRV,...]
        Parents for the sliced RV, which might be regular RVs or sliced RVs
    all_loops: list[Loop]
        Context of all relevant loops

    Returns
    -------
    sliced_rv: RV | SlicedRV
        New sliced RV
    """

    assert isinstance(op,Op)
    assert all(isinstance(p,OperatorRV) for p in parents)

    # parents = tuple(makerv(p) for p in parents)
    for p in parents:
        assert isinstance(p, OperatorRV)

    if all_loops == []:
        return OperatorRV(op, *parents)  # if no loops on stack, just give regular RV
    elif not op.random and not any(isinstance(p, SlicedRV) for p in parents):
        # TODO: random being used correctly?
        # special case, no need to create map
        return OperatorRV(op, *parents)
    else:
        vmap_op = op
        # all_loops = tuple(reversed(Loop.loops))
        for loop in reversed(all_loops):
            # print(f"{loop=}")
            # TODO: infer axis_size when possible
            axis_size = loop.length
            in_axes = tuple(_loop2axis(p, loop, all_loops) for p in parents)
            #print(f"{in_axes=}")
            #print(f"{axis_size=}")
            vmap_op = VMap(vmap_op, in_axes, axis_size)

        loop_axes = tuple(range(len(all_loops)))

        full_parents = [p.full_rv if isinstance(p, SlicedRV) else p for p in parents]

        full_rv = OperatorRV(vmap_op, *full_parents)
        #full_rv = rv_factory(vmap_op, *full_parents)
        #full_rv = all_loops[-1].rv_factory(vmap_op, *full_parents)

        return SlicedRV(
            op,
            *parents,
            full_rv=full_rv,
            loops=all_loops,
            loop_axes=loop_axes,
        )


class HideLoops:
    def __enter__(self):
        print("hiding loops")
        self.old_loops = Loop.loops
        Loop.loops = []

    def __exit__(self, exc_type, exc_value, exc_tb):
        print("done hiding loops")
        Loop.loops = self.old_loops


# class slot(OperatorRV):
#     """
#     A VMapRV is a "slot" into which you can assign inside of a Loop context. While
#     technically a RV, nothing is initialized until a single assigment call, which
#     must include all current loops, in order.
#
#     Examples
#     --------
#     >>> from pangolin.loops import VMapRV, Loop
#     >>> from pangolin import normal
#     >>> x = VMapRV()
#     >>> with Loop(3) as i:
#     >>>     x[i] = normal(0,1)
#     >>> x.shape
#     (3,)
#     """
#
#     def __init__(self, copy_rv=True):
#         # do not call super!
#         self.copy_rv = copy_rv
#         pass
#
#     def __setitem__(self, idx, value):
#         # TODO: somehow allow full slices for clarity?
#
#         if isinstance(value, Loop):
#             value = loop_makerv(value)
#
#         assert isinstance(value, SlicedRV)
#
#         if Loop.loops == []:
#             raise Exception("can't assign into VMapRV outside of Loop context")
#         elif len(Loop.loops) == 1:
#             assert Loop.loops == [idx]
#         else:
#             assert tuple(Loop.loops) == idx, "must assign using all loops, in order"
#
#         if self.copy_rv:
#             # try to "become" the RV being copied
#             super().__init__(value.full_rv.op, *value.full_rv.parents)
#         else:
#             # explicitly index from the RV being copied
#
#             with HideLoops():  # prevent indexing from seeing loops!
#                 copy = value.full_rv[:]
#             super().__init__(copy.op, *copy.parents)
#         return self

def slot():

    rv_type = type(makerv(1))
    #print(f"{rv_type=}")

    class VMapRV:
        """
        A VMapRV is a "slot" into which you can assign inside of a Loop context. While
        technically a RV, nothing is initialized until a single assigment call, which
        must include all current loops, in order.

        Examples
        --------
        >>> from pangolin.loops import VMapRV, Loop
        >>> from pangolin import normal
        >>> x = VMapRV()
        >>> with Loop(3) as i:
        >>>     x[i] = normal(0,1)
        >>> x.shape
        (3,)
        """

        def __init__(self, copy_rv=True):
            # do not call super!
            self.copy_rv = copy_rv
            pass

        def __setitem__(self, idx, value):
            # TODO: somehow allow full slices for clarity?

            if isinstance(value, Loop):
                value = loop_makerv(value)

            assert isinstance(value, SlicedRV)

            if Loop.loops == []:
                raise Exception("can't assign into VMapRV outside of Loop context")
            elif len(Loop.loops) == 1:
                assert Loop.loops == [idx]
            else:
                assert tuple(Loop.loops) == idx, "must assign using all loops, in order"

            if self.copy_rv:
                # try to "become" the RV being copied
                # super().__init__(self, value.full_rv.op, *value.full_rv.parents)
                # ultra evil changing of self type!
                self.__class__ = type(value.full_rv)
                assert isinstance(self, type(value.full_rv))
                #print(f"1 {self.__class__=}")
                type(value.full_rv).__init__(self, value.full_rv.op, *value.full_rv.parents)
            else:
                # explicitly index from the RV being copied

                with HideLoops():  # prevent indexing from seeing loops!
                    copy = value.full_rv[:]
                super().__init__(copy.op, *copy.parents)
            return self

    return VMapRV()

# https://stackoverflow.com/questions/7940470/is-it-possible-to-overwrite-self-to-point-to-another-object-inside-self-method
