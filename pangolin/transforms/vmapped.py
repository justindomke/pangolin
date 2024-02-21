from .. import interface
from ..interface import VMapDist
from .transforms import InapplicableTransform
from .local_transforms import LocalTransform

from .. import inference_numpyro

from jax import tree_map, tree_util


# Thoughts:
# it's really awkward to need to extract the parents here and pass them to regenerate
# just so the cond_dists can be examined by the transformer


def vmap_regenerator(base_regenerator):
    def new_regenerator(vars, parents_of_vars, *, pars_included, has_observed_descendent):
        def check_vmap_dist(var):
            if not isinstance(var.cond_dist, VMapDist):
                raise InapplicableTransform(f"dist not vmapped {var.cond_dist}")

        tree_map(check_vmap_dist, vars)

        # all vars that point TO EACH OTHER must map over axis 0 only
        def check_axes(var, included):
            for in_axis, in_included in zip(var.cond_dist.in_axes, included):
                if in_axis != 0 and in_included:
                    raise InapplicableTransform(
                        "regenerated parent not mapped over axis 0"
                    )

        tree_map(check_axes, vars, pars_included)

        vars_in_axes = tree_map(lambda node: 0, vars)
        parents_in_axes = tree_map(lambda node: node.cond_dist.in_axes, vars)
        in_axes = vars_in_axes, parents_in_axes

        flat_vars, _ = tree_util.tree_flatten(vars)
        axis_sizes = tuple(
            var.cond_dist.axis_size
            for var in flat_vars
            if var.cond_dist.axis_size is not None
        )

        if len(axis_sizes) > 0:
            assert all(axis_size == axis_sizes[0] for axis_size in axis_sizes)
            axis_size = axis_sizes[0]
        else:
            axis_size = None

        def myfun(vars, info_vars):
            return base_regenerator(
                vars,
                info_vars,
                pars_included=pars_included,
                has_observed_descendent=has_observed_descendent,
            )

        return interface.vmap(myfun, in_axes, axis_size)(vars, parents_of_vars)

    return new_regenerator


def vmap_local_transform(base_tform):
    assert isinstance(base_tform, LocalTransform)
    extractor = base_tform.extractor
    regenerator = vmap_regenerator(base_tform.regenerator)
    return LocalTransform(extractor, regenerator)
