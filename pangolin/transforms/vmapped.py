from .. import interface
from ..interface import VMapDist
from .transforms import InapplicableTransform
from .local_transforms import LocalTransform, LocalTransformEZ

from .. import inference_numpyro

from jax import tree_map, tree_util


# Thoughts:
# it's really awkward to need to extract the parents here and pass them to regenerate
# just so the cond_dists can be examined by the transformer


def vmap_transformer(base_transformer):
    def new_transformer(*cond_dists, pars_included, has_observed_descendent):
        if any(not isinstance(cond_dist, VMapDist) for cond_dist in cond_dists):
            raise InapplicableTransform("dist not vmapped")

        # check: for all the cond_dists that point TO EACH OTHER
        # they must map over axis 0 only
        for cond_dist, included in zip(cond_dists, pars_included):
            # if not isinstance(cond_dist, VMapDist):
            #    raise InapplicableTform("dist not vmapped")
            for in_axis, in_included in zip(cond_dist.in_axes, included):
                if in_axis != 0 and in_included:
                    raise InapplicableTransform(
                        "regenerated parent not mapped over axis 0"
                    )

        base_cond_dists = tuple(cond_dist.base_cond_dist for cond_dist in cond_dists)
        in_axes = tuple(cond_dist.in_axes for cond_dist in cond_dists)

        axis_sizes = tuple(
            cond_dist.axis_size
            for cond_dist in cond_dists
            if cond_dist.axis_size is not None
        )

        if len(axis_sizes) > 0:
            assert all(axis_size == axis_sizes[0] for axis_size in axis_sizes)
            axis_size = axis_sizes[0]
        else:
            axis_size = None

        print(f"{base_cond_dists=}")

        base_tform = base_transformer(
            *base_cond_dists,
            pars_included=pars_included,
            has_observed_descendent=has_observed_descendent,
        )

        def regenerate(*nodes_parents):
            return interface.vmap(base_tform, in_axes, axis_size)(*nodes_parents)

        return regenerate

    return new_transformer


def vmap_local_transform(base_tform):
    assert isinstance(base_tform, LocalTransform)
    extractor = base_tform.extractor
    transformer = vmap_transformer(base_tform.transformer)
    return LocalTransform(extractor, transformer)


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


def vmap_local_transform_ez(base_tform):
    assert isinstance(base_tform, LocalTransformEZ)
    extractor = base_tform.extractor
    regenerator = vmap_regenerator(base_tform.regenerator)
    return LocalTransformEZ(extractor, regenerator)
