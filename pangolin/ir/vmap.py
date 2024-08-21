from pangolin.ir import Op, RV
import numpy as np
from pangolin import util
import jax.tree_util


class VMap(Op):
    """
    Represents a `VMap` Op. That's *one specific* op vectorized over some number of arguments.
    """

    def __init__(self, base_op: Op, in_axes: tuple[int | None, ...] | list[int | None, ...], axis_size: int | None = None):
        """
        Create a `VMap` Op. All arguments here are heavily inspired by [`jax.lax.vmap`](
        https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) although note that
        `VMap` only maps a single `Op`. (The `vmap` function defined elsewhere takes an arbitrary
        function and transforms it into a graph of RVs with `VMap` `Op`s.)
        Parameters
        ----------
        base_op: Op
            The `Op` to be mapped
        in_axes: tuple[int | None, ...] | list[int | None, ...]
            What axis to map for each argument of the op (each can be a non-negative int or
            `None`)
        axis_size: int | None
            The size of the mapped axis/axes. Optional unless all elements of `in_axes` are `None`.
        """

        assert isinstance(base_op, Op)
        if isinstance(in_axes, list):
            in_axes = tuple(in_axes)
        assert isinstance(in_axes, tuple), "in_axes must be tuple"
        if axis_size is None:
            assert any(
                axis is not None for axis in in_axes
            ), "if axis_size=None, at least one axis must be mapped"
        else:
            if not isinstance(axis_size, (int,np.integer)):
                raise Exception(f"axis_size must be None or int was {type(axis_size)}")

        self.base_op = base_op
        self.in_axes = in_axes
        self.axis_size = axis_size
        super().__init__(name="Map", random=base_op.random)

    def _get_shape(self, *parents_shapes):
        remaining_shapes, axis_size = get_sliced_shapes(
            parents_shapes, self.in_axes, self.axis_size
        )
        dummy_shape = self.base_op.get_shape(*remaining_shapes)
        return (axis_size,) + dummy_shape

    def __repr__(self):
        out = f"VMap({repr(self.base_op)},{repr(self.in_axes)}"
        if self.axis_size:
            out += f",{repr(self.axis_size)}"
        out += ")"
        return out

    def __str__(self):
        """
        Return a string representation of the VMap op. Just like `__repr__`` except (1) uses str
        for calling the recursive distribution and (2) uses a symbol '∅' instead of `None` for
        representing unmapped args
        """
        # this is kind of overkill but whatever...
        new_in_axes = jax.tree_util.tree_map(
            lambda x: "∅" if x is None else x,
            self.in_axes,
            is_leaf=util.is_leaf_with_none,
        )
        out = f"VMap({str(self.base_op)},{str(new_in_axes)}"
        if self.axis_size:
            out += f",{repr(self.axis_size)}"
        out += ")"
        return out

    def __eq__(self, other):
        if isinstance(other, VMap):
            return (
                self.base_op == other.base_op
                and self.in_axes == other.in_axes
                and self.axis_size == other.axis_size
            )
        return False

    def __hash__(self):
        return hash((self.base_op, self.in_axes, self.axis_size))


def split_shape(shape, i):
    if i is None:
        new_shape = shape
        new_axis_size = None
    else:
        lo, mid, hi = (shape[:i], shape[i], shape[i + 1 :])
        new_shape = lo + hi
        new_axis_size = shape[i]
    return new_shape, new_axis_size


def get_sliced_shapes(shapes, in_axes, axis_size):
    axis_size = axis_size
    remaining_shapes = []
    for i, shape in zip(in_axes, shapes):
        new_shape, new_axis_size = split_shape(shape, i)
        remaining_shapes.append(new_shape)
        if axis_size is None:
            axis_size = new_axis_size
        elif new_axis_size is not None:
            assert axis_size == new_axis_size, "incoherent axis size"
    return remaining_shapes, axis_size
