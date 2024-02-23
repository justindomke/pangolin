from .. import util
from .. import dag
from ..ir import RV

from typing import Protocol, Iterable, Sequence, Tuple, Callable


# The only type checking we have for transforms is this Protocol
class Transform(Protocol):
    """
    Protocol for the call signature of a valid "Transform". A Transform is
    any function that obeys the signature
    `new_vars, new_given, new_vals = tform(vars, given, vals)`, where all the inputs
    and outputs are flat sequences (no pytrees). The fundamental property a
    Transform must guarantee is that the conditional distribution of new_vars |
    new_given == new_vals is the same as the conditional distribution of vars |
    given == vals.

    Parameters
    ----------
    vars : Sequence[RV]
        query variables
    given : Sequence[RV]
        conditioning variables
    vals : Sequence[numeric]
        conditioning values (same length/shapes as **given**)

    Returns
    -------
    new_vars : Sequence[RV]
        new query variables (same length/shapes as **vars**.)
    new_given : Sequence[RV]
        new conditioning variables (may have different length/shapes from **given**.)
    new_vals : Sequence[RV]
        new conditioning values (same length/shapes as **new_given**)

    Raises
    ------
    InapplicableTransform
        If the transform cannot (or "should not") be applied
    """

    def __call__(
        self, vars: Sequence[RV], given: Sequence[RV], vals: Sequence
    ) -> Tuple[Sequence[RV], Sequence[RV], Sequence]:
        pass


def apply_transforms(transforms: Sequence[Transform], tree_vars, tree_given, tree_vals):
    """
    Given a sequence of transforms, apply them to a query triple. Keeps applying
    transformations until all of them return `InapplicableTransform`. Also takes
    care of flattening and unflattening so individual transforms never need to worry
    about it.

    Parameters
    ----------
    transforms: Sequence[Transform]
        sequence of transforms to apply (each must obey Transform protocol but don't
        need to explicitly inherit or declare anything)
    tree_vars
        pytree of query variables
    tree_given
        pytree of conditioning variables
    tree_vals
        pytree of conditioning values (must match treedef/shapes of **tree_given**)

    Returns
    -------
    new_tree_vars
        pytree of new query variables (must match treedef/shapes of **tree_vars**)
    new_tree_given
        pytree of new conditioning variables
    new_tree_vals
        pytree of new conditioning values (must match treedef/shapes of
        **new_tree_given**)
    """

    vars, given, vals, unflatten_vars = util.flatten_args(
        tree_vars, tree_given, tree_vals
    )

    progress = True
    while progress:
        progress = False
        for t in transforms:
            try:
                vars, given, vals = t(vars, given, vals)
                progress = True
            except InapplicableTransform:
                continue

    return unflatten_vars(vars), given, vals


class InapplicableTransform(Exception):
    """
    Transforms should return an exception of this type if they are called on inputs
    where either they should not be applied or where applying them would be
    counterproductive.
    """

    pass
