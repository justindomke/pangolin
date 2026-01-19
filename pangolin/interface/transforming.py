from . import InfixRV
from pangolin.ir import Op, VMap, Constant, print_upstream, Bijector, Transformed
from pangolin import dag, ir, util
from collections.abc import Callable
from .base import (
    makerv,
    create_rv,
    RVLike,
    constant,
    exp,
    log,
    normal,
    abs,
    exponential,
    beta,
    uniform,
    cholesky,
    matmul,
    diag,
    diag_matrix,
    config,
    override,
    sum,
)
from .indexing import vector_index
from typing import Sequence
from typing import Any, Self
from jaxtyping import PyTree
from .compositing import make_composite, generated_nodes, Composite
from .base import AbstractOp
from . import base
from .util import fill_tril, extract_tril


Shape = ir.Shape


class Transform:
    """
    Create a `Transform` object. A `Transform` can be used to wrap a function that produces a single `RV` with a random `Op` and turn it into a function the produces a single `RV` with type `ir.Transformed`.

    Mathematically, the idea is that if ``P(X)`` is some density and ``Y=T(X)`` is a diffeomorphism, then ``P(Y=y) = P(X=T⁻¹(y)) × |det ∇T⁻¹(y)|``. The goal here is that if you have some function that will produce the random variable ``X``, this transform can be as a wrapper to create ``Y`` instead.

    Three functions must be provided as inputs:

    1. ``y = forward(x, b1, ..., bN)`` performs the forward transformation. This takes a single `RVLike` ``x`` along with some number of `RVLike` parameters ``b1, ..., bN`` and produces a single **random** `InfixRV` ``y`` as output. Any number of intermediate `InfixRV` can be created internally, but these must all be non-random. Must be a diffeomorphism with respect to ``x`` for any fixed values of ``b1, ..., bN``.
    2. ``x = inverse(y, b1, ..., bN)`` performs the inverse transformation, with similar properties as ``forward``. Must be an inverse in the sense that ``inverse(forward(x,*args),*args)==x``.
    3.  ``log_jac_det(x, y, b1, ..., bN)`` computes the log-determinant Jacobian ``log|det ∇forward(x,*args)| == -log|det ∇backwards(y,*args)|``. Both ``x`` and ``y`` are provided since it may be more convenient to use one rather than the other.

    When a transform is called, by default it returns a wrapped function that produces only a transformed random variable (``y``). However, if the keyword-only argument ``orig`` is ``True``, then the wrapped function also returns a deterministic version of the original variable. That is, it returns ``x`` that is a deterministic function of ``y``.


    Args:
        forward: Function to compute forward transformation.
        inverse: Function to compute inverse transformation.
        log_det_jac: Function to compute log-determinant-Jacobian.

    Returns:
        Wrapper function that takes a function ``fun(*parents) -> InfixRV[Op]`` and returns a function ``wrapped(*args,*parents) -> InfixRV[Wrapped[Op,Bijector]]``. (The bijector is constructed automatically.) The final `Op` (``Wrapped[Op,Bijector]``) is always random.

    Examples
    --------

    Create an `inverse-exponential <https://en.wikipedia.org/wiki/Inverse_distribution#Inverse_exponential_distribution>`__ distributed random variable:

    >>> reciprocal = Transform(lambda x: 1/x, lambda y: 1/y, lambda x, y: -2*log(abs(x)))
    >>> print(exponential(1.5))
    exponential(1.5)
    >>> print(reciprocal(exponential)(1.5)) # doctest: +NORMALIZE_WHITESPACE
    transformed(exponential,
                bijector(composite(1, [1, div], [[], [1, 0]]),
                         composite(1, [1, div], [[], [1, 0]]),
                         composite(2, [abs, log, -2, mul], [[0], [2], [], [4, 3]])))(1.5)

    Create an inverse-exponential-distributed random variable. But also return a deterministic random variable where the inverse is un-done.

    >>> reciprocal = Transform(lambda x: 1/x, lambda y: 1/y, lambda x, y: -2*log(abs(x)))
    >>> y = reciprocal(exponential)(1.5)
    >>> x = reciprocal.inverse(y)
    >>> x.op
    Div()
    >>> y in x.parents
    True

    Do the same as the above in a single call.

    >>> reciprocal = Transform(lambda x: 1/x, lambda y: 1/y, lambda x, y: -2*log(abs(x)))
    >>> y, x = reciprocal.apply_and_invert(exponential)(1.5)
    >>> x.op
    Div()
    >>> y in x.parents
    True

    Create a random variable distributed like ``x*4.4`` for ``x ~ normal(3.3, 3.3)``

    >>> myfun = lambda a: normal(a,a)
    >>> scale = Transform(lambda x, b: x*b, lambda y, b: y/b, lambda x, y, b: log(b), 1)
    >>> x = myfun(3.3)
    >>> print(x)
    normal(3.3, 3.3)
    >>> y = scale(myfun, 4.4)(3.3)
    >>> print(y) # doctest: +NORMALIZE_WHITESPACE
    transformed(normal,
                bijector(mul,
                        div,
                        composite(3, [log], [[2]])), 1)(4.4, 3.3, 3.3)

    Create a lognormal distribution parameterized in terms of the precision of the original normal.

    >>> exp_tform = Transform(exp, log, lambda x,y: y)
    >>> def normal_precision(mean, precision):
    ...     return normal(mean, 1/precision ** 0.5)
    >>> x = exp_tform(normal_precision)(1.1, 5.5)
    >>> print(x) # doctest: +NORMALIZE_WHITESPACE
    transformed(normal,
                bijector(exp,
                        log,
                        composite(2, [identity], [[1]])))(1.1, div(1, pow(5.5, 0.5)))

    See also: `tforms`

    """

    def __init__[*Ts](
        self,
        forward: Callable[..., InfixRV],
        inverse: Callable[..., InfixRV],
        log_det_jac: Callable[..., InfixRV],
        n_biject_args: int = 0,
    ):
        self.forward = forward
        self.inverse = inverse
        self.log_det_jac = log_det_jac
        self.n_biject_args = n_biject_args

    def apply_and_invert[O: Op](self, fun: Callable[..., InfixRV[O]], *biject_args: RVLike):
        """
        Args:
            fun: Original function. Should take some number of `RVLike` arguments and return a single *random* `InfixRV`.
            biject_args: Arguments to the transform.

        Returns:
            Wrapped version of ``fun`` that creates a tuple containing (1) A random `InfixRV` with a `Transformed` `Op`. (2) A deterministic `InfixRV` that depends on the first one that undoes the original transform.
        """

        biject_args = tuple(makerv(a) for a in biject_args)
        biject_arg_shapes = [a.shape for a in biject_args]
        if len(biject_args) != self.n_biject_args:
            raise ValueError(f"{len(biject_args)=} does not match {self.n_biject_args=}")

        def transformed_fun(*args: RVLike) -> tuple[InfixRV[Transformed[O, Bijector]], InfixRV]:
            args = tuple(makerv(a) for a in args)
            x: InfixRV[O] = fun(*args)
            bijector = make_bijector(self.forward, self.inverse, self.log_det_jac, x.shape, *biject_arg_shapes)
            if not x.op.random:
                raise ValueError(f"Cannot transform non-random op {x.op}")

            transformed_op = Transformed(x.op, bijector, self.n_biject_args)
            y = InfixRV(transformed_op, *biject_args, *x.parents)
            x = self.inverse(y, *biject_args)
            return y, x

        return transformed_fun

    def __call__[O: Op](self, fun: Callable[..., InfixRV[O]], *biject_args: RVLike):
        """
        Call the transform to wrap a function that creates a single `InfixRV` to get a new function that creates a transformed `InfixRV`.

        Args:
            fun: Original function. Should take some number of `RVLike` arguments and return a single *random* `InfixRV`.
            biject_args: Arguments to the transform.

        Returns:
            Wrapped version of ``fun`` that creates an `InfixRV` with a `Transformed` `Op`.
        """

        get_y_x = self.apply_and_invert(fun, *biject_args)

        def transformed_fun(*args: RVLike) -> InfixRV[Transformed[O, Bijector]]:
            return get_y_x(*args)[0]

        return transformed_fun

    @property
    def reverse(self) -> "Transform":
        """
        A `Transform` that performs the inverse / reverse of this `Transform`. Achieved by just swapping around ``forward`` and ``inverse`` and then creating a new ``log_det_jac`` function that has reversed arguments and reversed sign.
        """
        return Transform(self.inverse, self.forward, lambda x, y, *args: -self.log_det_jac(y, x, *args))

    def __repr__(self) -> str:
        return f"Transform({self.forward}, {self.inverse}, {self.log_det_jac})"


def maybe_make_composite[LastOp: Op](
    flat_fun: Callable[..., InfixRV[LastOp]], *input_shapes: Shape
) -> tuple[LastOp | Composite[LastOp] | Composite[ir.Identity], list[InfixRV]]:
    dummy_args = tuple(InfixRV(AbstractOp(shape)) for shape in input_shapes)

    # generated_nodes runs on "flat" functions that return arrays, not single RVs
    test_out = flat_fun(*dummy_args)
    if test_out in dummy_args:
        where_out = dummy_args.index(test_out)
        num_inputs = len(dummy_args)
        return Composite(num_inputs, (ir.Identity(),), [[where_out]]), []

    f = lambda *args: [flat_fun(*args)]
    all_vars, [out] = generated_nodes(f, *dummy_args)
    assert isinstance(out, InfixRV), "output of function must be a single InfixRV"

    single_op = all(a in dummy_args for a in out.parents)
    same_parents = out.parents == dummy_args
    if single_op and same_parents:
        assert all_vars == [out]
        return out.op, []

    return make_composite(flat_fun, *input_shapes)


def make_bijector(forward_fn, inverse_fn, log_det_jac_fn, x_shape: Shape, *biject_args_shapes: Shape) -> Bijector:
    """
    Examples
    --------
    >>> bijector = make_bijector(lambda x: exp(x), lambda y: log(y), lambda x, y: y, ())
    >>> print(bijector)
    bijector(exp, log, composite(2, [identity], [[1]]))
    """

    forward_op, forward_constants = maybe_make_composite(forward_fn, x_shape, *biject_args_shapes)
    y_shape = forward_op.get_shape(x_shape, *biject_args_shapes)
    inverse_op, inverse_constants = maybe_make_composite(inverse_fn, y_shape, *biject_args_shapes)
    log_det_jac_op, log_det_jac_constants = maybe_make_composite(log_det_jac_fn, x_shape, y_shape, *biject_args_shapes)

    if len(forward_constants):
        raise ValueError("foward_fn cannot capture closure variables")
    if len(inverse_constants):
        raise ValueError("inverse_fn cannot capture closure variables")
    if len(log_det_jac_constants):
        raise ValueError("log_det_jac_fn cannot capture closure variables")

    x_shape_pred = inverse_op.get_shape(y_shape, *biject_args_shapes)
    if x_shape != x_shape_pred:
        raise ValueError(f"x_shape {x_shape} does not match inverted shape {x_shape_pred}")

    bijector = Bijector(forward_op, inverse_op, log_det_jac_op)
    return bijector


########################################################################################
# Composing transforms
########################################################################################


def compose_transforms(transforms: Sequence[Transform], log_det_direction: str = "forward") -> Transform:
    """
    Composes a sequence of Transform objects into a single Transform.

    The resulting Transform applies ``y = Tn(...T2(T1(x))...)``.yield

    If the individual transforms have arguments, these are in linear order. E.g. if you have ``T1(x,a)`` and ``T2(y,b)`` then ``compose_transforms([T1,T2])(x,a,b)`` applies ``z = T2(T1(x,a),b)``.

    Args:
        transforms: Sequence of `Transform` objects
        log_det_direction: Should the new ``log_det_jac`` function loop through the transforms in forward or inverse order?

    Returns:
        Composite `Transform` object.
    """

    import builtins

    total_biject_args = builtins.sum(t.n_biject_args for t in transforms)

    def check_args(*args):
        if len(args) != total_biject_args:
            raise TypeError(f"Expected {total_biject_args} args, got {len(args)}")

    def composed_forward(x, *args):
        check_args(*args)

        current_x = x
        arg_idx = 0
        for t in transforms:
            n = t.n_biject_args
            t_args = args[arg_idx : arg_idx + n]
            current_x = t.forward(current_x, *t_args)
            arg_idx += n
        return current_x

    def composed_inverse(y, *args):
        check_args(*args)

        current_y = y
        arg_idx = total_biject_args
        for t in reversed(transforms):
            n = t.n_biject_args
            t_args = args[arg_idx - n : arg_idx]
            current_y = t.inverse(current_y, *t_args)
            arg_idx -= n
        return current_y

    def _log_det_forward(x, y, *args):
        """Iterates Forward (T1 -> Tn). Relies on x."""
        check_args(*args)

        log_det_sum = constant(0.0)
        current_x = x
        arg_idx = 0
        for t in transforms:
            n = t.n_biject_args
            t_args = args[arg_idx : arg_idx + n]
            next_x = t.forward(current_x, *t_args)
            log_det_sum += t.log_det_jac(current_x, next_x, *t_args)
            current_x = next_x
            arg_idx += n

        return log_det_sum

    def _log_det_inverse(x, y, *args):
        """Iterates Backward (Tn -> T1). Relies on y."""
        check_args(*args)

        log_det_sum = constant(0.0)
        current_y = y
        arg_idx = total_biject_args
        for t in reversed(transforms):
            n = t.n_biject_args
            t_args = args[arg_idx - n : arg_idx]
            previous_x = t.inverse(current_y, *t_args)
            log_det_sum += t.log_det_jac(previous_x, current_y, *t_args)
            current_y = previous_x
            arg_idx -= n
        return log_det_sum

    if log_det_direction == "forward":
        composed_log_det_jac = _log_det_forward
    elif log_det_direction == "inverse":
        composed_log_det_jac = _log_det_inverse
    else:
        raise ValueError("log_det_direction must be 'forward' or 'inverse'")

    return Transform(composed_forward, composed_inverse, composed_log_det_jac, total_biject_args)


########################################################################################
# Library of specific transforms
########################################################################################


def _cholesky_log_det_jac(X, Y):
    # X is original matrix
    # Y is cholesky factor
    k = Y.shape[0]
    powers = constant(range(k, 0, -1))
    return -k * log(2) - matmul(powers, log(diag(Y)))


def _exp_diagonal(X: RVLike):
    X = makerv(X)
    x = diag(X)
    with override(broadcasting="simple"):
        return X + diag_matrix(exp(x) - x)


def _log_diagonal(X: RVLike):
    X = makerv(X)
    x = diag(X)
    with override(broadcasting="simple"):
        return X + diag_matrix(log(x) - x)


def _exp_diagonal_log_det_jac(X: RVLike, Y: RVLike):
    X = makerv(X)
    x = diag(X)
    return sum(x, axis=0)


class tforms:
    """
    A namespace containing a bunch of pre-baked `Transform` instances for common transforms of distributions.

    Examples
    --------
    Create a lognormal, ``y = exp(x), x ~ normal(0.5,1.5)``

    >>> y = tforms.exp(normal)(0.5,1.5)
    >>> print(y)
    transformed(normal, bijector(exp, log, composite(2, [log], [[1]])))(0.5, 1.5)

    Create ``y = log(x), x ~ exponential(1.5)``, i.e. an unconstrained version of an exponential distribution:

    >>> y = tforms.exp.reverse(exponential)(1.5)
    >>> print(y)
    transformed(exponential, bijector(log, exp, composite(2, [log, -1, mul], [[0], [], [2, 3]])))(1.5)


    Create ``y = logit(x), x ~ beta(2.2, 2.2)``, i.e. an unconstrained version of a beta distribution with one parameter.

    >>> y = tforms.logit(lambda u: beta(u,u))(2.2)

    Create ``y = scaled_logit(x, 3.5, 5.5), x ~ uniform(3.5, 5.5)`` i.e. an unconstrained version of a uniform distribution.

    >>> y = tforms.scaled_logit(uniform, 3.5, 5.5)(3.0, 5.0)
    """

    # f(x) = exp(x)  <==>  df/dx = exp(x) = y  <==>  log df/dx = log(y)
    exp = Transform(exp, log, lambda x, y: log(y))
    """
    A `Transform` instance that applies the exp bijector ``y = exp(x)``. Commonly used to transform from reals to positive reals.
    """

    log = exp.reverse

    logit = Transform(base.logit, base.inv_logit, lambda x, y: -log(x) - log(1 - x))
    """
    A `Transform` instance that applies the logit bijector ``y = logit(x)``. Commonly used to transform from [0,1] to reals.
    """

    inv_logit = logit.reverse
    """
    A `Transform` instance that applies the inverse logit.
    """

    scaled_logit = Transform(
        lambda x, a, b: base.logit((x - a) / (a - b)),
        lambda y, a, b: a + (b - a) * base.inv_logit(y),
        lambda x, y, a, b: base.log(x - a) + base.log(b - x) - base.log(b - a),  # should use softplus
        n_biject_args=2,
    )
    """
    A `Transform` instance that applies the scaled logit ``y = logit((y-a)/(a-b)``. Commonly used to transform from [a,b] to reals.
    """

    cholesky = Transform(
        lambda X: cholesky(X),
        lambda Y: base.matmul(Y, base.transpose(Y)),
        _cholesky_log_det_jac,
    )
    """
    A `Transform` instance that applies a Cholesky decomposition. Commonly used to transform from symmetric positive definite matrices into triangular matrices. 
    """

    fill_tril = Transform(fill_tril, extract_tril, lambda x, y: constant(0.0))
    """
    A `Transform` instance that fills a lower-triangular matrix from a vector. Used to transform from real vectors to lower-triangular matrices.
    """

    extract_tril = fill_tril.reverse
    """
    A `Transform` instance that extracts the lower-triangular part of a matrix. Commonly used to transform from triangular lower-triangular matrices to real vectors.
    """

    exp_diagonal = Transform(_exp_diagonal, _log_diagonal, _exp_diagonal_log_det_jac)
    """
    A `Transform` instance that exponentiates the diagonal of a matrix. Commonly used to transform real lower-triangular matrices into Cholesky factors.
    """

    log_diagonal = exp_diagonal.reverse
    """
    A `Transform` instance that takes the logarithm of the diagonal of a matrix. Commonly used to transform real lower-triangular matrices into Cholesky factors.
    """

    unconstrain_spd = compose_transforms([cholesky, log_diagonal, extract_tril])
    """
    A `Transform` instance that transforms a symmetric positive definite into the space of unconstrained reals. Accomplished by (1) taking a Cholesky decomposition (2) taking the logarithm of the diagonal (3) extracting the lower-triangular entries.
    """

    def __init__(self):
        raise TypeError("Use tforms as a static namespace, do not instantiate.")
