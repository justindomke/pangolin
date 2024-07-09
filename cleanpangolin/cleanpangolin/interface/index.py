from cleanpangolin.ir import RV, Constant, Index
from cleanpangolin.util import most_specific_class

# TODO: some day merge "indexes into indexes"
# - this seems to NOT be very simple—all depends on shapes of inputs and stuff)
# - for jax backends I guess you get views and it's fine?
# - but for Stan or JAGS seems problematic?
# - (or should we solve in the codegen phase?)

def index(var:RV, *indices:RV | slice):
    """
    Convenience function to create a new indexed RV.

    Note: Users are not expected to call this function directly. Instead they should just use
    normal indexing notation like `x[y]`.

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

    indices = tuple(i if isinstance(i, (slice,RV)) else RV(Constant(i)) for i in indices)

    non_slice_indices = (i for i in indices if isinstance(i,RV))
    # don't pass OperatorRV here, but I think OK because user should never explicitly call this
    # function
    rv_class = most_specific_class(var, *non_slice_indices)

    print(f"{rv_class=}")

    # add extra full slices
    num_full_slices_needed = var.ndim - len(indices)
    indices = indices + (slice(None),) * num_full_slices_needed

    slices = []
    parents = []
    for i in indices:
        if isinstance(i, slice):
            slices.append(i)
        else:
            parents.append(i)
            slices.append(None)

    return rv_class(Index(*slices), var, *parents)
