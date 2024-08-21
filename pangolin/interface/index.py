from pangolin.ir import RV, Constant, Index
from pangolin.util import most_specific_class
from pangolin.interface import interface


# TODO: some day merge "indexes into indexes"
# - this seems to NOT be very simple—all depends on shapes of inputs and stuff)
# - for jax backends I guess you get views and it's fine?
# - but for Stan or JAGS seems problematic?
# - (or should we solve in the codegen phase?)


def eliminate_ellipses(ndim, idx):
    num_ellipsis = len([i for i in idx if i is ...])
    if num_ellipsis > 1:
        raise ValueError("an index can only have a single ellipsis ('...')")
    elif num_ellipsis == 1:
        where = idx.index(...)
        slices_needed = ndim - (len(idx) - 1)  # sub out ellipsis
        if where > 0:
            idx_start = idx[:where]
        else:
            idx_start = ()
        idx_mid = (slice(None),) * slices_needed
        idx_end = idx[where + 1 :]
        idx = idx_start + idx_mid + idx_end
    return idx


def pad_with_slices(ndim, idx):
    num_full_slices_needed = ndim - len(idx)
    return idx + (slice(None),) * num_full_slices_needed


def convert_constants(idx):
    from pangolin.interface.interface import rv_factory  # avoid circular import

    return tuple(i if isinstance(i, (slice, RV)) else rv_factory(Constant(i)) for i in idx)


def simplify_indices(ndim:int, idx):
    """
    Converts ellipses to slices and pads with slices but does not convert constants to RVs.
    """

    idx = eliminate_ellipses(ndim, idx)

    if ndim == 0:
        raise Exception("can't index scalar RV")
    elif len(idx) > ndim:
        raise Exception("RV indexed with more dimensions than exist")

    idx = pad_with_slices(ndim, idx)

    return idx


def standard_index_fun(var: RV, *indices: RV | slice):
    """
    Convenience function to create a new indexed RV. Typically, users would not call this function
    directly but use normal indexing notation like `x[y]` and rely on operator overloading to
    call this function.

    Parameters
    ----------
    var: RV
        the RV to be indexed
    indices
        each element of `indices` must either be (a) a slice with fixed integer indices, (b) an RV
        with integer values, or (c) something that can be cast to a constant. The length of
        `indices` is permitted to be less than `var.ndim`. If so, extra full slices are added for
        all missing dimensions.
    Returns
    -------
    new_rv: RV
        The new indexed RV, conceptually equivalent to `var[*indices]`.
    """

    from pangolin.interface.interface import rv_factory  # avoid circular import

    indices = simplify_indices(var.ndim, indices)

    indices = convert_constants(indices)

    slices = [i if isinstance(i, slice) else None for i in indices]
    parents = [i for i in indices if not isinstance(i, slice)]

    return rv_factory(Index(*slices), var, *parents)


index_funs = [standard_index_fun]


def index(var: RV, *indices: RV | slice):
    return index_funs[-1](var, *indices)
