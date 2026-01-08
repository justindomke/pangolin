from . import InfixRV
from pangolin.ir import Op, VMap, Constant, print_upstream, Bijector, Transformed
from pangolin import dag, ir, util
from collections.abc import Callable
from .base import makerv, create_rv, RVLike, constant, exp, log, normal, abs, exponential, beta
from typing import Sequence
from typing import Any, Self
from jaxtyping import PyTree
from .compositing import make_composite, generated_nodes, Composite
from .vmapping import AbstractOp
from . import base


Shape = ir.Shape


class Transform:
    """
    Create a `Transform` object. A `Transform` can be used to wrap a function that produces a single `RV` with a random `Op` and turn it into a function the produces a single `RV` with type `ir.Transformed`.

    Mathematically, the idea is that if ``P(X)`` is some density and ``Y=T(X)`` is a diffeomorphism, then ``P(Y=y) = P(X=T⁻¹(y)) × |det ∇T⁻¹(y)|``. The goal here is that if you have some function that will produce the random variable ``X``, this transform can be as a wrapper to create ``Y`` instead.

    Three functions must be provided as inputs:

    1. ``y = forward(x, b1, ..., bN)`` performs the forward transformation. This takes a single `RVLike` ``x`` along with some number of `RVLike` parameters ``b1, ..., bN`` and produces a single **random** `InfixRV`` ``y`` as output. Any number of intermediate `InfixRV` can be created internally, but these must all be non-random. Must be a diffeomorphism with respect to ``x`` for any fixed values of ``b1, ..., bN``.
    2. ``x = inverse(y, b1, ..., bN)`` performs the inverse transformation, with similar properties as ``forward``. Must be an inverse in the sense that ``inverse(forward(x,*args),*args)==x``.
    3.  ``log_jac_det(x, y, b1, ..., bN)`` computes the log-determinant Jacobian ``log|det ∇forward(x,*args)| == -log|det ∇backwards(y,*args)|``. Both ``x`` and ``y`` are provided since it may be more convenient to use one rather than the other.


    Args:
        forward: Function to compute forward transformation.
        inverse: Function to compute inverse transformation.
        log_det_jac: Function to compute log-determinant-Jacobian.

    Returns:
        Wrapper function that takes a function ``fun(*parents) -> InfixRV[Op]`` and returns a function ``wrapped(*args,*parents) -> InfixRV[Wrapped[Op,Bijector]]``. (The bijector is constructed automatically.) The final `Op` (``Wrapped[Op,Bijector]``) is always random.

    Examples
    --------

    Create an `inverse-exponential <https://en.wikipedia.org/wiki/Inverse_distribution#Inverse_exponential_distribution>`__ distributed random variable:

    >>> myfun = exponential
    >>> reciprocal = Transform(lambda x: 1/x, lambda y: 1/y, lambda x, y: -2*log(abs(x)))
    >>> print(myfun(1.5))
    exponential(1.5)
    >>> print(reciprocal(myfun)(1.5)) # doctest: +NORMALIZE_WHITESPACE
    transformed(exponential,
                bijector(composite(1, [1, div], [[], [1, 0]]),
                         composite(1, [1, div], [[], [1, 0]]),
                         composite(2, [abs, log, -2, mul], [[0], [2], [], [4, 3]])))(1.5)

    Create a random variable distributed like ``x*4.4`` for ``x ~ normal(3.3, 3.3)``

    >>> myfun = lambda a: normal(a,a)
    >>> scale = Transform(lambda x, b: x*b, lambda y, b: y/b, lambda x, y, b: log(b))
    >>> x = myfun(3.3)
    >>> print(x)
    normal(3.3, 3.3)
    >>> y = scale(myfun, 4.4)(3.3)
    >>> print(y) # doctest: +NORMALIZE_WHITESPACE
    transformed(normal,
                bijector(mul,
                        div,
                        composite(3, [log], [[2]])), 1)(4.4, 3.3, 3.3)

    It's not clear if this is a good idea, but a Transform can also be used as a decorator. The following creates a lognormal distribution parameterized in terms of the precision of the original normal.

    >>> @Transform(exp, log, lambda x,y: y)
    ... def lognormal_precision(mean, precision):
    ...     return normal(mean, 1/precision ** 0.5)
    >>> x = lognormal_precision(1.1, 5.5)
    >>> print(x) # doctest: +NORMALIZE_WHITESPACE
    transformed(normal,
                bijector(exp,
                        log,
                        composite(2, [identity], [[1]])))(1.1, div(1, pow(5.5, 0.5)))
    """

    def __init__[*Ts](
        self,
        forward: Callable[..., InfixRV],
        inverse: Callable[..., InfixRV],
        log_det_jac: Callable[..., InfixRV],
    ):
        self.forward = forward
        self.inverse = inverse
        self.log_det_jac = log_det_jac

    def __call__[O: Op](
        self, fun: Callable[..., InfixRV[O]], *biject_args: RVLike
    ) -> Callable[..., InfixRV[Transformed[O, Bijector]]]:

        biject_args = tuple(makerv(a) for a in biject_args)
        biject_arg_shapes = [a.shape for a in biject_args]
        n_biject_args = len(biject_args)

        def transformed_fun(*args: RVLike) -> InfixRV[Transformed[O, Any]]:
            args = tuple(makerv(a) for a in args)
            x: InfixRV[O] = fun(*args)
            bijector = make_bijector(self.forward, self.inverse, self.log_det_jac, x.shape, *biject_arg_shapes)
            if not x.op.random:
                raise ValueError(f"Cannot transform non-random op {x.op}")

            transformed_op = Transformed(x.op, bijector, n_biject_args)
            return InfixRV(transformed_op, *biject_args, *x.parents)

        return transformed_fun

    @property
    def inv(self) -> "Transform":
        """
        A `Transform` that performs the inverse of this `Transform`.
        """
        return Transform(self.inverse, self.forward, lambda x, y, *args: self.log_det_jac(y, x, *args))

    def __repr__(self) -> str:
        return f"Transform({self.forward}, {self.inverse}, {self.log_det_jac})"


class transforms:
    """
    Container for pre-defined `Transform` instances.

    Examples
    --------

    Create a lognormal, ``y = exp(x), x ~ normal(0,1)``

    >>> a = transforms.exp(normal)(0,1)

    Create ``y = log(x), x ~ exponential(1)``, i.e. sample from an unconstrained version of an exponential distribution:

    >>> b = transforms.exp.inv(exponential)(1)

    Create ``y = logit(x), x ~ beta(z, z), z ~ exponential(2.2)``, i.e. sample from an unconstrained version of a beta distribution with one parameter.

    >>> z = exponential(2.2)
    >>> c = transforms.logit(lambda u: beta(u,u))(z)

    """

    exp = Transform(exp, log, lambda x, y: y + 0)
    """
    A `Transform` instance that applies the exp bijector ``y = exp(x)``. Commonly used to transform from reals to positive reals.
    """

    logit = Transform(base.logit, base.inv_logit, lambda x, y: -log(x) - log(1 - x))
    """
    A `Transform` instance that applies the logit bijector ``y = logit(x)``. Commonly used to transform from [0,1] to reals.
    """

    scaled_logit = Transform(
        lambda x, a, b: base.logit((x - a) / (a - b)),
        lambda y, a, b: a + (b - a) * base.inv_logit(y),
        lambda x, y, a, b: base.log(x - a) + base.log(b - x) - base.log(b - a),  # should use softplus
    )
    """
    A `Transform` instance that applies the scaled logit ``y = logit((y-a)/(a-b)``. Commonly used to transform from [a,b] to reals.
    """


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
