"""WARNING: The loops module is highly experimental.
"""


from .ir import RV, AbstractCondDist, VMapDist, makerv, CondDist
from typing import Sequence, List, Self
from . import util


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
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        assert Loop.loops.pop() is self


class SlicedRV(RV):
    """Represents a "slice" of a higher dimenstional RV. Has the shape of just a single slice but has a reference to the
    full RV from which the slice was taken, and what Loops were applied to what axes to get the slice
    """

    def __init__(self, cond_dist, *parents, full_rv, loops, loop_axes):
        """
        Create a SlicedRV.

        Parameters
        ----------
        cond_dist: CondDist
            The conditional distribution for the slice.
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
        super().__init__(cond_dist, *parents)
        assert self.shape == util.remove_indices(full_rv.shape, loop_axes)


def slice_existing_rv(full_rv, idx, all_loops):
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

    cond_dist = AbstractCondDist(my_shape)
    parents = []

    return SlicedRV(
        cond_dist,
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
        print(f"{original_axis=} {axes_to_come=} {correct_axis}")
        return correct_axis


def make_sliced_rv(cond_dist, *parents, all_loops):
    """
    Create a new SlicedRV from a CondDist, a set of parents, and a list of loops.
    Automatically creates a corresponding non-sliced RV.

    Parameters
    ----------
    cond_dist: CondDist
        The conditional distribution to apply to create the new SlicedRV
    *parents: RV
        Parents for the sliced RV, which might be regular RVs or sliced RVs
    all_loops: Sequence[Loop]
        Context of all relevant loops

    Returns
    -------
    sliced_rv: RV | SlicedRV
        New sliced RV
    """

    # parents = tuple(makerv(p) for p in parents)
    for p in parents:
        assert isinstance(p, RV)

    if all_loops == []:
        return RV(cond_dist, *parents)  # if no loops on stack, just give regular RV
    elif not cond_dist.random and not any(isinstance(p, SlicedRV) for p in parents):
        # special case, no need to create map
        return RV(cond_dist, *parents)
    else:
        vmap_cond_dist = cond_dist
        # all_loops = tuple(reversed(Loop.loops))
        for loop in reversed(all_loops):
            # print(f"{loop=}")
            # TODO: infer axis_size when possible
            axis_size = loop.length
            in_axes = tuple(_loop2axis(p, loop, all_loops) for p in parents)
            vmap_cond_dist = VMapDist(vmap_cond_dist, in_axes, axis_size)

        loop_axes = tuple(range(len(all_loops)))

        full_parents = [p.full_rv if isinstance(p, SlicedRV) else p for p in parents]

        full_rv = RV(vmap_cond_dist, *full_parents)

        return SlicedRV(
            cond_dist,
            *parents,
            full_rv=full_rv,
            loops=all_loops,
            loop_axes=loop_axes,
        )


class HideLoops:
    def __enter__(self):
        self.old_loops = Loop.loops
        Loop.loops = []

    def __exit__(self, exc_type, exc_value, exc_tb):
        Loop.loops = self.old_loops


class VMapRV(RV):
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
        if Loop.loops == []:
            raise Exception("can't assign into VMapRV outside of Loop context")
        elif len(Loop.loops) == 1:
            assert Loop.loops == [idx]
        else:
            assert tuple(Loop.loops) == idx

        if self.copy_rv:
            # try to "become" the RV being copied
            super().__init__(value.full_rv.cond_dist, *value.full_rv.parents)
        else:
            # explicitly index from the RV being copied

            with HideLoops():  # prevent indexing from seeing loops!
                copy = value.full_rv[:]
            super().__init__(copy.cond_dist, *copy.parents)


# https://stackoverflow.com/questions/7940470/is-it-possible-to-overwrite-self-to-point-to-another-object-inside-self-method
