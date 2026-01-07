from . import InfixRV
from pangolin.ir import Op, VMap, Constant, print_upstream, Bijector, Transformed
from pangolin import dag, ir, util
from collections.abc import Callable
from .base import makerv, create_rv, RVLike, constant, exp, log, normal
from typing import Sequence, Type
from typing import Protocol, TypeVar, Any
from numpy.typing import ArrayLike
import numpy as np
from jax import numpy as jnp
from typing import Protocol
from jaxtyping import PyTree
from .compositing import make_composite
from .base import exp, log, mul, constant

"""
The desired API is:

```
x = normal(0,1)
y = transform(x, exp_bijector) # y = exp(x)
```

Or

```
x = normal(0,1)
y = transform(x, inverse=log_bijector) # x = log(y)
```

In either case, y should follow a lognormal distribution.
"""

Shape = ir.Shape

# will need some kind of make-bijector method


def make_bijector(forward_fn, inverse_fn, log_det_jac_fn, x_shape: Shape, *biject_args_shapes: Shape) -> Bijector:
    """
    Examples
    --------
    >>> bijector = make_bijector(lambda x: exp(x), lambda y: log(y), lambda x, y: y+0, ())
    >>> print(bijector)
    bijector(composite(1, [exp], [[0]]), composite(1, [log], [[0]]), composite(2, [0, add], [[], [1, 2]]))
    """

    forward_op, forward_constants = make_composite(forward_fn, x_shape, *biject_args_shapes)
    y_shape = forward_op.get_shape(x_shape, *biject_args_shapes)
    inverse_op, inverse_constants = make_composite(inverse_fn, y_shape, *biject_args_shapes)
    log_det_jac_op, log_det_jac_constants = make_composite(log_det_jac_fn, x_shape, y_shape, *biject_args_shapes)

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


# def transform[O: Op, B: Bijector](
#     fun: Callable[..., InfixRV[O]], bijector: B, *biject_args: RVLike
# ) -> Callable[..., InfixRV[Transformed[O, B]]]:
#     """
#     Given a function that produces an RV, get a function the produces a transformed RV

#     Examples
#     --------
#     >>> x = transform(normal, ir.ExpBijector())(0,1)
#     >>> x
#     InfixRV(Transformed(Normal(), ExpBijector()), InfixRV(Constant(0)), InfixRV(Constant(1)))
#     >>> print(x)
#     transformed(normal, exp_bijector)(0, 1)
#     """

#     biject_args = tuple(makerv(a) for a in biject_args)

#     def myfun(*args: RVLike):
#         args = tuple(makerv(a) for a in args)

#         x: InfixRV[O] = fun(*args)

#         if not x.op.random:
#             raise ValueError(f"Cannot transform non-random op {x.op}")

#         n_biject_args = len(biject_args)
#         transformed_op = Transformed(x.op, bijector, n_biject_args)
#         return InfixRV(transformed_op, *biject_args, *x.parents)

#     return myfun


class Transform[B: Bijector]:
    """
    A class to conveniently apply bijectors

    Args:
        bijector: The `Bijector` to apply

    Returns:
        Function that takes (1) a Callable producing a single RV and (2) some number of RVLike Arguments and returns a Transformed RV.

    Examples
    --------
    >>> transform = Transform(ir.ExpBijector())
    >>> x = transform(normal)(0,1)
    >>> x
    InfixRV(Transformed(Normal(), ExpBijector()), InfixRV(Constant(0)), InfixRV(Constant(1)))
    >>> print(x)
    transformed(normal, exp_bijector)(0, 1)
    """

    def __init__(self, bijector: B):
        if not isinstance(bijector, Bijector):
            raise ValueError(f"bijector was {type(bijector)} rather than Bijector")

        self.bijector = bijector

    def __call__[O: Op](
        self, fun: Callable[..., InfixRV[O]], *biject_args: RVLike
    ) -> Callable[..., InfixRV[Transformed[O, B]]]:

        biject_args = tuple(makerv(a) for a in biject_args)

        def myfun(*args: RVLike):
            args = tuple(makerv(a) for a in args)

            x: InfixRV[O] = fun(*args)

            if not x.op.random:
                raise ValueError(f"Cannot transform non-random op {x.op}")

            n_biject_args = len(biject_args)
            transformed_op = Transformed(x.op, self.bijector, n_biject_args)
            return InfixRV(transformed_op, *biject_args, *x.parents)

        return myfun


exp_transform = Transform(ir.ExpBijector())
"""
A `Transform` instance that applies the exp bijector.

Examples
--------
>>> x = exp_transform(normal)(0,1)
>>> print(x)
transformed(normal, exp_bijector)(0, 1)
"""

mul_bijector = make_bijector(lambda x, a: x * a, lambda y, a: y / a, lambda x, y, a: log(a), (), ())


def make_transform(forward_fn, inverse_fn, log_det_jac_fn):
    """
    Examples
    --------
    >>> transform = make_transform(lambda x: exp(x), lambda y: log(y), lambda x, y: y+0)
    >>> x = transform(normal)(0,1)
    >>> print(x)
    transformed(normal, bijector(composite(1, [exp], [[0]]), composite(1, [log], [[0]]), composite(2, [0, add], [[], [1, 2]])))(0, 1)
    """

    def transform[O: Op](fun: Callable[..., InfixRV[O]], *biject_args: RVLike):
        biject_args = tuple(makerv(a) for a in biject_args)
        biject_arg_shapes = [a.shape for a in biject_args]
        n_biject_args = len(biject_args)

        def apply(*args: RVLike):
            args = tuple(makerv(a) for a in args)
            x: InfixRV[O] = fun(*args)
            bijector = make_bijector(forward_fn, inverse_fn, log_det_jac_fn, x.shape, *biject_arg_shapes)
            if not x.op.random:
                raise ValueError(f"Cannot transform non-random op {x.op}")

            transformed_op = Transformed(x.op, bijector, n_biject_args)
            return InfixRV(transformed_op, *biject_args, *x.parents)

        return apply

    return transform


# def exp_transform[O: Op](fun: Callable[..., InfixRV[O]]) -> Callable[..., InfixRV[Transformed[O, ir.ExpBijector]]]:
#     """
#     Examples
#     --------
#     >>> x = exp_transform(normal)(0,1)
#     >>> x
#     InfixRV(Transformed(Normal(), ExpBijector()), InfixRV(Constant(0)), InfixRV(Constant(1)))
#     >>> print(x)
#     transformed(normal, exp_bijector)(0, 1)
#     """

#     return transform(fun, ir.ExpBijector())


# def transform[O: Op, B: Bijector](var: InfixRV[O], bijector: B, *biject_args: RVLike) -> InfixRV[Transformed[O, B]]:
#     """
#     Examples
#     --------
#     >>> x = normal(0,1)
#     >>> y = transform(x, ir.ExpBijector())
#     >>> y
#     InfixRV(Transformed(Normal(), ExpBijector()), InfixRV(Constant(0)), InfixRV(Constant(1)))
#     >>> print(y)
#     transformed(normal, exp_bijector)(0, 1)
#     >>> y.parents == x.parents
#     True

#     >>> x = normal(0,1)
#     >>> y = transform(x, ir.MulBijector(), 7.21)
#     >>> y
#     InfixRV(Transformed(Normal(), MulBijector(), 1), InfixRV(Constant(7.21)), InfixRV(Constant(0)), InfixRV(Constant(1)))
#     >>> print(y)
#     transformed(normal, mul_bijector, 1)(7.21, 0, 1)
#     >>> y.parents[0].op
#     Constant(7.21)
#     >>> y.parents[1:] == x.parents
#     True
#     """

#     biject_args = tuple(makerv(a) for a in biject_args)

#     if not var.op.random:
#         raise ValueError(f"Cannot transform non-random op {var.op}")

#     n_biject_args = len(biject_args)
#     transformed_op = Transformed(var.op, bijector, n_biject_args)
#     return InfixRV(transformed_op, *biject_args, *var.parents)
