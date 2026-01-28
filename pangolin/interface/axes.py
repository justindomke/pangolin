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


class Axis(ir.Op):
    """
    A scalar value but "special"
    """

    _random = False
    _get_shape = lambda *args: ()

    def __init__(self, size: int):
        self.size = size
        super().__init__()


class AxisRV(InfixRV[Axis]):
    active_axes = []

    def __init__(self, op: Axis):
        self.assigned_slots: list[Slot] = []
        super().__init__(op)

    def __enter__(self):
        if hasattr(self, "enter_n"):
            raise ValueError("AxisRV cannot be entered twice")

        AxisRV.active_axes.append(self)

        self.__dict__["enter_n"] = InfixRV._n  # get around frozen
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # get around frozen
        if hasattr(self, "exit_n"):
            raise ValueError("AxisRV cannot be exited twice")

        if not AxisRV.active_axes:
            raise Exception(f"Exiting with no active axes. (Pangolin bug.)")

        if AxisRV.active_axes[-1] is not self:
            raise Exception(
                f"Context Exit Order Violation: {self} is not the most recent active context. (Pangolin bug.)"
            )

        # Remove from the end
        AxisRV.active_axes.pop()

        self.__dict__["exit_n"] = InfixRV._n


    def var_in_n_range(self, rv: InfixRV):
        if not hasattr(self, "enter_n"):
            raise ValueError("ContextRV was not entered")
        if not hasattr(self, "exit_n"):
            raise ValueError("ContextRV was not exited")

        enter_n = self.__dict__["enter_n"]
        exit_n = self.__dict__["exit_n"]

        if rv._n < enter_n:
            return False
        elif rv._n < exit_n:
            return True
        else:
            raise ValueError(f"rv created after ContextRV exit ({enter_n, exit_n}) vs {rv._n}")

    @property
    def range(self):
        return pi.constant(range(self.op.size))


def axis(size):
    axis_op = Axis(size)
    return AxisRV(axis_op)


def index_if_necessary(base, *indices):
    if all(idx.op == ir.Constant(range(size)) for size, idx in zip(base.shape, indices)):
        return base
    else:
        return InfixRV(ir.Index(), base, *indices)


def popout(rv: InfixRV[ir.Index], ax: InfixRV[Axis]):
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

            new_idx = pi.constant(range(ax.op.size))
            new_indices = indices[:n] + (new_idx,) + indices[n + 1 :]
            return index_if_necessary(base, *new_indices), where
        else:
            where += idx.ndim
    raise ValueError(f"axis rv {ax} not found in Index rv {rv}")


def extract(output: PyTree[InfixRV], ax: AxisRV):
    """
    Given some RV or group of RVs, extract the information necessary to

    """

    # don't visit nodes created before context manager enter
    # throw error if see nodes created after context manager exit
    def node_block(var: InfixRV):
        return not ax.var_in_n_range(var)

    output_rvs, output_treedef = jax.tree_util.tree_flatten(output)

    all_rvs = dag.upstream_nodes(output_rvs, node_block=node_block)

    indexed_rvs = []
    indexed_dims = []
    for rv in all_rvs:
        if isinstance(rv.op, ir.Index):
            rv_indices = rv.parents[1:]
            if ax in rv_indices:
                new_rv, where = popout(rv, ax)
                indexed_rvs.append(new_rv)
                indexed_dims.append(where)

    def replay(indexed_rvs_replayed: list[InfixRV[ir.Index]], ax_replayed: InfixRV):
        rv_to_replayed: dict[InfixRV, InfixRV] = {ax: ax_replayed}

        n = 0
        for rv in all_rvs:
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


def vmap_axis(output: Sequence[InfixRV], ax: AxisRV):
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

    # in_axes = dims
    # new_fun = vmap(replay, in_axes, axis_size)
    # return new_fun(arrays)

    in_axes = (dims, 0)
    new_fun = vmap(replay, in_axes, axis_size)
    return new_fun(arrays, ax.range)


class Slot(InfixRV):
    """
    The job of a Slot is as follows:

    - When created, record all active Axis context managers

    - Initially, throw an error if the user tries to do anything except __setitem__

    - If the user DOES do __setitem__, then make sure that:
      1. The indices are all context managers activated since Slot __init__ (in same order) plus full slices
      2. Record self onto slot_list for all context manager indices
      3. Store the value

    - At this point, continue throwing an error if the user tries to do anything with the Slot. The only legal thing is to do __getitem__ with exactly the same sequence.

    - When one of the context managers exits, all the slots should be (together) vmapped over that axis, and the axis removed from all of them. This changes

    - Once ALL the context mangers have exited, finally you can do __init__ and act like a real InfixRV

    Danger: What happens if you do this?

    x = Slot()
    y = Slot()
    with Axis(3) as i:
        x[i] = normal(0,1)
        y[i] = x[i]

    Or how about this

    x = Slot()
    y = Slot()
    with Axis(3) as i:
        a = normal(0,1)
        x[i] = a
        y[i] = a

    I think the only reliable answer is that before vmapping an axis, you always create a
    """

    def __init__(self):
        self.assigned = False
        self.axes: list[InfixRV] = []
        self.active_axes_when_created = [ax for ax in AxisRV.active_axes]  # copy!
        # do NOT call super().__init__()

    def expected_axes(self):
        if not util.starts_with(AxisRV.active_axes, self.active_axes_when_created):
            raise ValueError(
                f"Active axes {AxisRV.active_axes} does not start active axes when slot created: {self.active_axes_when_created}"
            )
        return AxisRV.active_axes[len(self.active_axes_when_created) :]

    def expected_key(self, value):
        return self.expected_axes() + [slice(None)] * value.ndim

    def __setitem__(self, key, value: pi.RVLike):
        if self.assigned:
            raise ValueError("Can't assign to a Slot twice")

        value = pi.makerv(value)

        if not isinstance(key, tuple):
            key = [key]
        else:
            key = list(key)

        if key != self.expected_key(value):
            raise ValueError(f"key {key} does not match expected {self.expected_key(value)}")

        for ax in self.expected_axes():
            ax.assigned_slots.append(self)

        self.axes = self.expected_axes()
        self.value = InfixRV(ir.Identity(), value)
        # self.value = value
        self.assigned = True

    def __getitem__(self, key):  # type: ignore
        if not self.assigned:
            raise ValueError("Can't read from non-assigned Slot")

        if not isinstance(key, tuple):
            key = [key]
        else:
            key = list(key)

        if key != self.expected_key(self.value):
            raise ValueError(f"key {key} does not match expected {self.expected_key(self.value)}")

        # TODO: We return a reference to the OG value, not the assigned value
        # is that what we want?
        return self.value.parents[0]


def update_slots(slots: list[Slot], ax: AxisRV):
    for slot in slots:
        assert slot.assigned
        assert ax in slot.axes

    old_values: list[InfixRV] = [s.value for s in slots]
    new_values = vmap_axis(old_values, ax)

    for s, v in zip(slots, new_values, strict=True):
        s.value = v
        s.axes = [a for a in s.axes if a is not ax]
        if s.axes == []:
            super(Slot, s).__init__(s.value.op, *s.value.parents)


# TODO:
# - Transform indices into "pure scalars" when

# Danger zone:
# - Accessing a variable without a loop axis created inside a loop (should we actively "kill"?)
# - Doing "math" with
