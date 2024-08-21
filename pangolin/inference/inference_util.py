from pangolin import util
import numpy as np

class Calculate:
    def __init__(self, sample_flat, **frozen_options):
        """
        Create a `Calculate` object.

        Inputs:
        * `flat`: function that can perform flat sampling
        * `**options`: options to "freeze" in
        """

        self.sample_flat = sample_flat
        self.frozen_options = frozen_options  # options for that engine

    def sample(
        self,
        vars,
        given_vars=None,
        given_vals=None,
        reduce_fn=None,
        **options,
    ):
        """
        Draw samples!

        Inputs:
        * `vars`: a pytree of `RV`s to sample. (Can be any pytree)
        * `given_vars`: a pytree of `RV`s to condition on. `None` is no conditioning
        variables. (
        Can be any pytree)
        * `given_vals`: a pytree of observed values. (Pytree must match `given_vars`.)
        * `reduce_fn` (optional) will apply a function to the samples for each `RV` in
        `vars` before returning samples. (This is used to define `E`, `var`, etc.
        below.)

        Outputs:
        * A `pytree` of `RV`s matching `vars` with one extra dimension, containing
        the samples.

        Example:
        ```python
        x = normal(0,1)
        y = normal(x,1)
        sample(x,y,2) # returns something close to 1.0
        ```
        """

        for key in options:
            if key in self.frozen_options:
                raise ValueError(f"key {key} in both options and frozen_options")

        (
            flat_vars,
            flat_given_vars,
            flat_given_vals,
            unflatten,
            unflatten_given,
        ) = util.flatten_args(vars, given_vars, given_vals)

        flat_samps = self.sample_flat(
            flat_vars, flat_given_vars, flat_given_vals, **options, **self.frozen_options
        )

        if reduce_fn is not None:
            flat_samps = map(reduce_fn, flat_samps)

        return unflatten(flat_samps)

    # def sample(self, vars, given_vars=None, given_vals=None, reduce_fn=None):
    #     """
    #     Draw samples!
    #
    #     Inputs:
    #     * `vars`: a pytree of `RV`s to sample. (Can be any pytree)
    #     * `given_vars`: a pytree of `RV`s to condition on. `None` is no conditioning
    #     variables. (
    #     Can be any pytree)
    #     * `given_vals`: a pytree of observed values. (Pytree must match `given_vars`.)
    #     * `reduce_fn` (optional) will apply a function to the samples for each `RV` in
    #     `vars` before returning samples. (This is used to define `E`, `var`, etc.
    #     below.)
    #
    #     Outputs:
    #     * A `pytree` of `RV`s matching `vars` with one extra dimension, containing
    #     the samples.
    #
    #     Example:
    #     ```python
    #     x = normal(0,1)
    #     y = normal(x,1)
    #     sample(x,y,2) # returns something close to 1.0
    #     ```
    #     """
    #
    #     given_vals = util.assimilate_vals(given_vars, given_vals)
    #
    #     flat_vars, vars_treedef = jax.tree_util.tree_flatten(vars)
    #     flat_given_vars, given_vars_treedef = jax.tree_util.tree_flatten(given_vars)
    #     flat_given_vals, given_vals_treedef = jax.tree_util.tree_flatten(given_vals)
    #     assert given_vars_treedef == given_vals_treedef
    #
    #     if self.engine == "jags":
    #         inference = inference_jags
    #     elif self.engine == "numpyro":
    #         inference = inference_numpyro
    #     elif self.engine == "stan":
    #         inference = inference_stan
    #     else:
    #         raise Exception(f"inference must be in {engines}")
    #
    #     flat_samps = inference.sample_flat(
    #         flat_vars, flat_given_vars, flat_given_vals, **self.options
    #     )
    #     if reduce_fn is not None:
    #         flat_samps = map(reduce_fn, flat_samps)
    #
    #     return jax.tree_util.tree_unflatten(vars_treedef, flat_samps)

    def E(self, vars, given_vars=None, given_vals=None, **options):
        return self.sample(vars, given_vars, given_vals, lambda x: np.mean(x, axis=0), **options)

    def var(self, vars, given_vars=None, given_vals=None, **options):
        return self.sample(vars, given_vars, given_vals, lambda x: np.var(x, axis=0), **options)

    def std(self, vars, given_vars=None, given_vals=None,**options):
        return self.sample(vars, given_vars, given_vals, lambda x: np.std(x, axis=0), **options)


# TODO:
# -offer ancestor_sample() and ancestor_log_prob() routines


# def get_non_flat_sampler(sample_flat):
#     """
#     Given a sampling function that does "flat" sampling, get a new one that does non-flat sampling
#     """
#
#     def sample(
#             vars,
#             given_vars=None,
#             given_vals=None,
#             #reduce_fn=None,
#             **vargs,
#     ):
#         """
#         Draw samples!
#
#         Inputs:
#         * `vars`: a pytree of `RV`s to sample. (Can be any pytree)
#         * `given_vars`: a pytree of `RV`s to condition on. `None` is no conditioning
#         variables. (
#         Can be any pytree)
#         * `given_vals`: a pytree of observed values. (Pytree must match `given_vars`.)
#         * `reduce_fn` (optional) will apply a function to the samples for each `RV` in
#         `vars` before returning samples. (This is used to define `E`, `var`, etc.
#         below.)
#
#         Outputs:
#         * A `pytree` of `RV`s matching `vars` with one extra dimension, containing
#         the samples.
#
#         Example:
#         ```python
#         x = normal(0,1)
#         y = normal(x,1)
#         sample(x,y,2) # returns something close to 1.0
#         ```
#         """
#
#         (
#             flat_vars,
#             flat_given_vars,
#             flat_given_vals,
#             unflatten,
#             unflatten_given,
#         ) = util.flatten_args(vars, given_vars, given_vals)
#
#         flat_samps = sample_flat(
#             flat_vars, flat_given_vars, flat_given_vals, **vargs
#         )
#
#         #if reduce_fn is not None:
#         #    flat_samps = map(reduce_fn, flat_samps)
#
#         return unflatten(flat_samps)
#     return sample
