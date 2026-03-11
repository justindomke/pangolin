from jax._src.core import pp_toplevel_jaxpr
from .base import InfixRV, AbstractOp, generated_nodes, constant, vmap_subgraph, vmap, print_upstream
from pangolin import ir
from typing import Sequence, Callable
import warnings
import inspect
from pangolin import dag, util
from jaxtyping import PyTree
import jax.tree_util
from pangolin import interface as pi


"""
How this works:

1) An Axis is a special scalar InfixRV type. It should never appear in a final model. Users can create models like this:

>>> a = pi.constant([1.1, 2.2, 3.3])
>>> b = pi.constant(2.2)
>>> i = InfixRV(AbstractOp(()))
>>> xi = normal(a[i], b)
>>> yi = normal(x_i, b)
>>> [x, y] = vmap_axis([xi, yi], i)

This creates an independent *copy* of xi and yi that is vmapped. xi and yi cannot themselves be used in inference because i is abstract

- A Reassigned is a special Op type that throws an 

"""


class AxisOp(ir.Op):
    """
    A scalar value but "special".

    An AxisOp should *never* appear in a final model. Backends should treat the presence of an AxisOp in a model passed for sampling as an error.
    """

    _random = False
    _get_shape = lambda *args: ()

    def __init__(self, size: int):
        self.range = pi.constant(range(size))
        self.size = size
        super().__init__()


def index_if_necessary(base, *indices):
    if all(idx.op == ir.Constant(range(size)) for size, idx in zip(base.shape, indices)):
        return base
    else:
        return InfixRV(ir.Index(), base, *indices)


def popout(rv: InfixRV[ir.Index], ax: InfixRV[AxisOp]):
    """
    - rv is some rv that's been indexed with some number of axes
    - we want to "recover" one of those axes, i.e. remove that axis from the indexing statement
    - and we want to know where to find it
    """

    where = 0
    base = rv.parents[0]
    indices = rv.parents[1:]
    for n, idx in enumerate(indices):
        if idx is ax:
            if base.shape[n] != ax.op.size:
                raise ValueError(f"Axis size {ax.op.size} does not match dim {n} of shape {base.shape}")

            new_idx = ax.op.range
            new_indices = indices[:n] + (new_idx,) + indices[n + 1 :]
            return index_if_necessary(base, *new_indices), where
        else:
            where += idx.ndim
    raise ValueError(f"axis rv {ax} not found in Index rv {rv}")


def extract(output: PyTree[InfixRV], ax: InfixRV[AxisOp]):
    """
    Given some RV or group of RVs, extract the information necessary to vmap.

    """

    output_rvs, output_treedef = jax.tree_util.tree_flatten(output)

    def node_block(var: InfixRV):
        return var._n < ax._n

    def edge_block(v1: InfixRV, v2: InfixRV):
        return v1.op == ir.Index() and v2 is ax

    all_rvs = dag.upstream_nodes(output_rvs, node_block=node_block, edge_block=edge_block)

    indexed_rvs: list[InfixRV] = []
    indexed_dims: list[int] = []
    for rv in all_rvs:
        if rv is ax:
            indexed_rvs.append(ax.op.range)
            indexed_dims.append(0)
            continue

        if isinstance(rv.op, ir.Index):
            rv_indices = rv.parents[1:]
            if ax in rv_indices:
                new_rv, where = popout(rv, ax)
                indexed_rvs.append(new_rv)
                indexed_dims.append(where)

    def replay(indexed_rvs_replayed: list[InfixRV]):
        rv_to_replayed: dict[InfixRV, InfixRV] = {}

        n = 0
        for rv in all_rvs:
            if rv is ax:
                rv_to_replayed[rv] = indexed_rvs_replayed[n]
                n += 1
                continue

            if isinstance(rv.op, ir.Index):
                if ax in rv.parents[1:]:
                    rv_to_replayed[rv] = indexed_rvs_replayed[n]
                    n += 1
                    continue

            new_parents = []
            for p in rv.parents:
                if p in rv_to_replayed:
                    new_parents.append(rv_to_replayed[p])
                else:
                    new_parents.append(p)

            replayed_rv = InfixRV(rv.op, *new_parents)
            rv_to_replayed[rv] = replayed_rv

        results_flat = [rv_to_replayed[out_rv] for out_rv in output_rvs]
        return jax.tree_util.tree_unflatten(output_treedef, results_flat)

    return indexed_rvs, indexed_dims, replay


def vmap_axis(output: Sequence[InfixRV], ax: InfixRV[AxisOp]):
    """
    Examples
    --------

    >>> x = constant([1.1, 2.2, 3.3])
    >>> y = constant([4.4, 5.5])
    >>> with caxis(3) as i:
    ...     with caxis(2) as j:
    ...         zij = x[i] * y[j]
    ...     [zi] = vmap_context_axis([zij], j)
    >>> [z] = vmap_context_axis([zi], i)
    >>> print(z.op)
    vmap(vmap(mul, [None, 0], 2), [0, None], 3)
    >>> z.parents == (x, y)
    True
    """

    arrays, dims, replay = extract(output, ax)

    axis_size = ax.op.size

    for array, dim in zip(arrays, dims, strict=True):
        if array.shape[dim] != axis_size:
            raise ValueError(f"Axis size {axis_size} does not match dim {dim} of shape {array.shape}")

    if len(arrays) == 0:
        warnings.warn("len(arrays)=0...")

    in_axes = dims
    new_fun = vmap(replay, in_axes, axis_size)
    return new_fun(arrays)


class Reassigned(ir.Op):
    """
    An Op indicating an RV that has been reassigned to a Slot
    """

    _random = False

    @property
    def random(self):
        raise ValueError("Can't get random for Reassigned Op")

    def _get_shape(self, *parent_shapes):
        raise ValueError("Can't get shape for Reassigned Op")


class Taker(InfixRV):
    def __init__(self):
        self.initialized = False
        # do NOT call super().__init__()

    def take(self, value: InfixRV):
        # "become" value
        super().__init__(value.op, *value.parents)
        # destroy value
        value.__dict__["op"] = Reassigned()
        value.__dict__["parents"] = ()


def split_key(key):
    i = 0
    while i < len(key) and isinstance(key[i], InfixRV):
        i += 1

    axes = key[:i]
    slices = key[i:]

    for a in axes:
        if not isinstance(a, InfixRV) and isinstance(a.op, AxisOp):
            raise ValueError("Non Axis Op found")

    for s in slices:
        if not s == slice(None):
            raise ValueError("Non full slice found")

    return axes, slices


class Slot(InfixRV):
    def __init__(self):
        # do NOT call super().__init__()
        pass

    def __setitem__(self, key, value: pi.RVLike):
        if hasattr(self, "value"):
            raise ValueError("Can't assign to a Slot twice")

        value = pi.makerv(value)

        if not isinstance(key, tuple):
            key = [key]
        else:
            key = list(key)

        axes, slices = split_key(key)

        self.axes = axes

        if len(slices) != value.ndim:
            raise ValueError("number of full slices does not match value.ndim")

        self.value = pi.InfixRV(value.op, *value.parents)

        # destroy assigned value
        value.__dict__["op"] = Reassigned()
        value.__dict__["parents"] = ()

    def __getitem__(self, key):  # type: ignore
        if not hasattr(self, "value"):
            raise ValueError("Can't get from a Slot before assigning")

        initialized = hasattr(self, "op")
        if initialized:
            return super().__getitem__(key)

        if not isinstance(key, tuple):
            key = [key]
        else:
            key = list(key)

        expected_key = self.axes + [slice(None)] * self.value.ndim

        if key != expected_key:
            raise ValueError("Can't index a non-initialized Slot with anything except initial key")

        return self.value


def update_slots(slots: list[Slot], ax: InfixRV[AxisOp]):
    for slot in slots:
        assert hasattr(slot, "value")
        assert ax in slot.axes

    old_values: list[InfixRV] = [s.value for s in slots]
    new_values = vmap_axis(old_values, ax)

    for s, v in zip(slots, new_values, strict=True):
        s.value = v
        s.axes = [a for a in s.axes if a is not ax]

        if s.axes == []:
            parents = []
            for p in s.value.parents:
                if p in new_values:
                    i = new_values.index(p)
                    parents.append(slots[i])
                else:
                    parents.append(p)

            super(Slot, s).__init__(s.value.op, *parents)


# def asssign_value(self, value: InfixRV):
#     self.value = InfixRV(value.op, *value.parents)
#     self.value_assigned = True

# def get_value(self):
#     if not self.value_assigned:
#         raise ValueError("Cannot get_value if not assigned")
#     return self.value


# class Axis(InfixRV[AxisOp]):
#     """
#     A special RV that always contains an AxisOp and that can act as a context manager.

#     Args:
#         op: The axis be be based on

#     """

#     active_axes = []

#     def __init__(self, size: int):
#         self.assigned_slots: list[Slot] = []
#         axis_op = AxisOp(size)
#         super().__init__(axis_op)

#     def __enter__(self):
#         # if hasattr(self, "enter_n"):
#         #    raise ValueError("AxisRV cannot be entered twice")

#         Axis.active_axes.append(self)

#         self.__dict__["enter_n"] = InfixRV._n  # get around frozen
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):  # get around frozen
#         # if hasattr(self, "exit_n"):
#         #    raise ValueError("AxisRV cannot be exited twice")

#         if not Axis.active_axes:
#             raise Exception(f"Exiting with no active axes. (Pangolin bug.)")

#         if Axis.active_axes[-1] is not self:
#             raise Exception(
#                 f"Context Exit Order Violation: {self} is not the most recent active context. (Pangolin bug.)"
#             )

#         # Remove from the end
#         Axis.active_axes.pop()

#         # self.__dict__["exit_n"] = InfixRV._n

#         # update_slots(self.assigned_slots, self)

#     # def var_in_n_range(self, rv: InfixRV):
#     #     if not hasattr(self, "enter_n"):
#     #         raise ValueError("ContextRV was not entered")
#     #     if not hasattr(self, "exit_n"):
#     #         raise ValueError("ContextRV was not exited")

#     #     enter_n = self.__dict__["enter_n"]
#     #     exit_n = self.__dict__["exit_n"]

#     #     if rv._n < enter_n:
#     #         return False
#     #     elif rv._n < exit_n:
#     #         return True
#     #     else:
#     #         raise ValueError(f"rv created after ContextRV exit ({enter_n, exit_n}) vs {rv._n}")


# def axis(size):
#     return Axis(size, True)


# def axis_debug(size):
#     return Axis(size, False)


# class Slot(InfixRV):
#     """
#     The job of a Slot is as follows:

#     - When created, record all active Axis context managers

#     - Initially, throw an error if the user tries to do anything except __setitem__

#     - If the user DOES do __setitem__, then make sure that:
#       1. The indices are all context managers activated since Slot __init__ (in same order) plus full slices
#       2. Record self onto slot_list for all context manager indices
#       3. Create a local "copy" of value, pointing to value's parents
#       4. Invalidate the existing copy of value, leading to an error if re-used

#     - At this point, continue throwing an error if the user tries to do anything with the Slot. The only legal thing is to do __getitem__ with exactly the same sequence.

#     - When one of the context managers exits, all the slots should be (together) vmapped over that axis, and the axis removed from all of them. This changes

#     - Once ALL the context mangers have exited, finally you can do __init__ and act like a real InfixRV

#     Danger: What happens if you do this?

#     x = Slot()
#     y = Slot()
#     with Axis(3) as i:
#         x[i] = normal(0,1)
#         y[i] = x[i]

#     Or how about this

#     x = Slot()
#     y = Slot()
#     with Axis(3) as i:
#         a = normal(0,1)
#         x[i] = a
#         y[i] = a

#     I think the only reliable answer is that before vmapping an axis, you always create a
#     """

#     def __init__(self):
#         self.assigned = False
#         self.initialized = False
#         self.axes: list[InfixRV] = []
#         self.active_axes_when_created = [ax for ax in Axis.active_axes]  # copy!
#         # do NOT call super().__init__()

#     def expected_axes(self):
#         if not util.starts_with(Axis.active_axes, self.active_axes_when_created):
#             raise ValueError(
#                 f"Active axes {Axis.active_axes} does not start active axes when slot created: {self.active_axes_when_created}"
#             )
#         return Axis.active_axes[len(self.active_axes_when_created) :]

#     def expected_key(self, value):
#         return self.expected_axes() + [slice(None)] * value.ndim

#     def __setitem__(self, key, value: pi.RVLike):
#         if self.assigned:
#             raise ValueError("Can't assign to a Slot twice")

#         value = pi.makerv(value)

#         if not isinstance(key, tuple):
#             key = [key]
#         else:
#             key = list(key)

#         if key != self.expected_key(value):
#             raise ValueError(f"key {key} does not match expected {self.expected_key(value)}")

#         for ax in self.expected_axes():
#             ax.assigned_slots.append(self)

#         self.axes = self.expected_axes()
#         # self.value = InfixRV(ir.Identity(), value)
#         # self.value = value

#         if not value._n >= self.axes[-1]._n:
#             raise ValueError("Cannot assign a value to a slot not created in innermost context manager")
#         self.value = InfixRV(value.op, *value.parents)
#         value.__dict__["op"] = Reassigned()  # invalidate existing value
#         value.__dict__["parents"] = []
#         self.assigned = True

#     def __getitem__(self, key):  # type: ignore
#         if self.initialized:
#             return super().__getitem__(key)

#         if not self.assigned:
#             raise ValueError("Can't read from non-assigned Slot")

#         if not isinstance(key, tuple):
#             key = [key]
#         else:
#             key = list(key)

#         if key != self.expected_key(self.value):
#             raise ValueError(f"key {key} does not match expected {self.expected_key(self.value)}")

#         # TODO: We return a reference to the OG value, not the assigned value
#         # is that what we want?
#         # return self.value.parents[0]
#         return self.value
#         # return super().__getitem__(key)


# # def update_slots(slots: list[Slot], ax: Axis):
# #     for slot in slots:
# #         assert slot.assigned
# #         assert ax in slot.axes

# #     old_values: list[InfixRV] = [s.value for s in slots]
# #     new_values = vmap_axis(old_values, ax)

# #     for s, v in zip(slots, new_values, strict=True):
# #         s.value = v
# #         s.axes = [a for a in s.axes if a is not ax]
# #         if s.axes == []:
# #             s.initialized = True
# #             super(Slot, s).__init__(s.value.op, *s.value.parents)


# TODO:
# - Transform indices into "pure scalars" when

# Danger zone:
# - Accessing a variable without a loop axis created inside a loop (should we actively "kill"?)
# - Doing "math" with
