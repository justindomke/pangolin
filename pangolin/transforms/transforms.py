from .. import util
from .. import dag
from ..ir import RV


class InapplicableTransform(Exception):
    pass


def apply_transforms(transforms, tree_vars, tree_given, tree_vals):
    """
    Inputs:
    * `vars` - a pytree of `RV`s
    * `given` - a pytree of `RV`s
    * `vals` - a pytree of values matching `given`

    Outputs:
    * `new_vars`
    * `new_given`
    * `new_vals`
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
