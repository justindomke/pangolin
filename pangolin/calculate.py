from typing import Any, Callable, TypeAlias, TYPE_CHECKING, Type, Sequence, List, Optional, Iterable
from jaxtyping import PyTree
from pangolin.ir import Op, RV, ArrayLike
from pangolin import util, ir
import numpy as np


class Calculate:
    """
    A `Calculate` object just remembers a set of options and then offers inference methods.

    The idea is that the user provides a `sample_flat` function. This can be any function that takes a triple of `(vars, given, values)` where `vars` and `given` are sequences of `RV` and `values` is a sequence of constants (arrays / floats) with the same length and shapes as `given`. The function then returns a list of arrays of the same length as `vars` but each has one more dimension at the beginning corresponding to the samples.

    Given this, `Calculate` provides three conveniences:

    1. It allows for binding default and/or frozen values for the `sample_flat` function.
    2. It "lifts" `sample_flat` into a function `sample` that can run on arbitrary pytrees rather than lists.
    3. It provides convenience functions `E`, `var`, `std`, `sample_arviz` that automatically take expectations and so on without manual intervention from the user.

    Parameters
    ----------
    sample_flat : callable
        function that performs inference.
    frozen : iterable of str, optional
        names of options that cannot be overridden at call time
    **options
        default option values for sample_flat

    """

    def __init__(self, sample_flat: Callable, *, frozen: Iterable[str] = (), **options):
        extra = frozen - options.keys()
        if extra:
            raise ValueError(
                f"frozen keys {sorted(extra)} have no corresponding option "
                f"(did you misspell one of {sorted(options)}?)"
            )

        self.sample_flat = sample_flat
        self.default = options
        self.frozen = frozen

    def sample(
        self,
        vars: PyTree[RV],
        given_vars: PyTree[RV] = None,
        given_vals: PyTree[ArrayLike] = None,
        reduce_fn: Optional[Callable] = None,
        **options,
    ):
        """
        Draw samples!

        Args:
            vars: A `RV` or list/tuple of `RV` or pytree of `RV` to sample.
            given_vars: A `RV` or list/tuple of `RV` or pytree of `RV` to condition on.
                ``None`` indicates no conditioning variables.
            given_vals: An ``ArrayLike`` or list/tuple of ``ArrayLike`` or pytree of
                ``ArrayLike`` representing observed values. Must match the structure and
                shape of ``given_vars``.
            reduce_fn:  Function to apply to each leaf node in samples before returning.
                This is used to create `E`, `var`, etc. (If ``None``, does nothing.)
            options: extra options to pass to sampler

        Returns:
            Pytree of JAX arrays matching structure and shape of ``vars`` but with one
            extra dimension at the start, containing the samples.

        Examples
        --------
        >>> from pangolin.blackjax import sample_flat, run_nuts
        >>> zero    = ir.RV(ir.Constant(0))
        >>> one     = ir.RV(ir.Constant(1))
        >>> x       = ir.RV(ir.Normal(), zero, one)
        >>> y       = ir.RV(ir.Normal(), x, one)
        >>> calc    = Calculate(sample_flat, run_inf=run_nuts, num_samples=529)
        >>> x_samps = calc.sample(x,y,2)
        >>> x_samps.shape
        (529,)
        >>> np.mean(x_samps) # something close to 1.0
        Array(...)
        """

        locked = self.frozen & options.keys()
        if locked:
            raise ValueError(f"options {sorted(locked)} are frozen")

        options = self.default | options  # overrides defaults

        (
            flat_vars,
            flat_given_vars,
            flat_given_vals,
            unflatten,
            unflatten_given,
        ) = util.flatten_args(vars, given_vars, given_vals)

        flat_samps = self.sample_flat(
            flat_vars,
            flat_given_vars,
            flat_given_vals,
            **options,
        )

        if reduce_fn is not None:
            flat_samps = map(reduce_fn, flat_samps)

        return unflatten(flat_samps)

    def E(
        self,
        vars: PyTree[RV],
        given_vars: PyTree[RV] = None,
        given_vals: PyTree[ArrayLike] = None,
        **options,
    ):
        """
        Compute (conditional) expected values. This is just a thin wrapper that calls
        `sample` and then reduces by taking the mean.

        Args:
            vars: A `RV` or list/tuple of `RV` or pytree of `RV` to sample.
            given_vars:  A `RV` or list/tuple of `RV` or pytree of `RV` to condition on.
                ``None`` indicates no conditioning variables.
            given_vals: An ``ArrayLike`` or list/tuple of ``ArrayLike`` or pytree of
                ``ArrayLike`` representing observed values. Must match the structure and
                shape of ``given_vars``.
            reduce_fn:  Function to apply to each leaf node in samples before returning.
                This is used to create `E`, `var`, etc. (If ``None``, does nothing.)
            options: extra options to pass to sampler

        Returns:
            Pytree of JAX arrays matching structure and shape of ``vars``, containing
                the expectations.


        Examples
        --------
        >>> from pangolin.blackjax import sample_flat, run_nuts
        >>> zero    = ir.RV(ir.Constant(0))
        >>> one     = ir.RV(ir.Constant(1))
        >>> x       = ir.RV(ir.Normal(), zero, one)
        >>> y       = ir.RV(ir.Normal(), x, one)
        >>> calc    = Calculate(sample_flat, run_inf=run_nuts, num_samples=529)
        >>> calc.E(x,y,2) # something close to 1.0
        Array(...)
        """

        return self.sample(vars, given_vars, given_vals, lambda x: np.mean(x, axis=0), **options)

    def var(
        self,
        vars: PyTree[RV],
        given_vars: PyTree[RV] = None,
        given_vals: PyTree[ArrayLike] = None,
        **options,
    ):
        return self.sample(vars, given_vars, given_vals, lambda x: np.var(x, axis=0), **options)

    def std(
        self,
        vars: PyTree[RV],
        given_vars: PyTree[RV] = None,
        given_vals: PyTree[ArrayLike] = None,
        **options,
    ):
        return self.sample(vars, given_vars, given_vals, lambda x: np.std(x, axis=0), **options)

    def sample_arviz(
        self,
        vars: dict[str, RV],
        given_vars: PyTree[RV] = None,
        given_vals: PyTree[ArrayLike] = None,
        **options,
    ):
        """This is an **experimental** function to draw samples in
        `ArviZ <https://www.arviz.org/en/latest/>`__ format.

        Note: ArviZ is not installed with pangolin by default: You must install it
        manually.

        Args:
            vars: dictionary mapping names to individual random variables
                given_vars: A `RV` or list/tuple of `RV` or pytree of `RV` to condition on.
                ``None`` indicates no conditioning variables.
            given_vars: A `RV` or list/tuple of `RV` or pytree of `RV` to condition on.
                given_vals: An ``ArrayLike`` or list/tuple of ``ArrayLike`` or pytree of
                ``ArrayLike`` representing observed values. Must match the structure and
                shape of ``given_vars``.
            reduce_fn:  Function to apply to each leaf node in samples before returning.
                This is used to create `E`, `var`, etc. (If ``None``, does nothing.)
            options: extra options to pass to sampler
        """

        try:
            from arviz import convert_to_inference_data
        except ImportError:
            raise ImportError("To use this method you must install arviz manually")

        samps = self.sample(vars, given_vars, given_vals, **options)
        samps_with_none = {key: samps[key][None, ...] for key in samps}
        dataset = convert_to_inference_data(samps_with_none)
        return dataset
