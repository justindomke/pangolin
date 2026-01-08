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
from .compositing import make_composite, generated_nodes, Composite
from .base import exp, log, mul, constant
from .vmapping import AbstractOp
from . import base

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
    if single_op:
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


# TODO: Allow forward and inverse to accept PyTree biject_args? (obscure...)


class Transform:
    def __init__(
        self, forward: Callable[..., InfixRV], inverse: Callable[..., InfixRV], log_det_jac: Callable[..., InfixRV]
    ):
        self.forward = forward
        self.inverse = inverse
        self.log_det_jac = log_det_jac

    @property
    def inv(self):
        return Transform(self.inverse, self.forward, lambda x, y, *args: self.log_det_jac(y, x, *args))

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


exp_transform = Transform(exp, log, lambda x, y: y + 0)
"""
A `Transform` instance that applies the exp bijector. Commonly used to transform from reals to positive reals.
"""

logit_transform = Transform(base.logit, base.inv_logit, lambda x, y: -log(x) - log(1 - x))
"""
A `Transform` instance that applies the logit bijector. Commonly used to transform from [0,1] to reals.
"""

scaled_logit_transform = Transform(
    lambda x, a, b: base.logit((x - a) / (a - b)),
    lambda y, a, b: a + (b - a) * base.inv_logit(y),
    lambda x, y, a, b: base.log(x - a) + base.log(b - x) - base.log(b - a),  # should use softplus
)
"""
A `Transform` instance that applies the scaled logit. Commonly used to transform from [a,b] to reals.
"""

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


# class Transform[B: Bijector]:
#     """
#     A class to conveniently apply bijectors

#     Args:
#         bijector: The `Bijector` to apply

#     Returns:
#         Function that takes (1) a Callable producing a single RV and (2) some number of RVLike Arguments and returns a Transformed RV.

#     Examples
#     --------
#     >>> transform = Transform(ir.ExpBijector())
#     >>> x = transform(normal)(0,1)
#     >>> x
#     InfixRV(Transformed(Normal(), ExpBijector()), InfixRV(Constant(0)), InfixRV(Constant(1)))
#     >>> print(x)
#     transformed(normal, exp_bijector)(0, 1)
#     """

#     def __init__(self, bijector: B):
#         if not isinstance(bijector, Bijector):
#             raise ValueError(f"bijector was {type(bijector)} rather than Bijector")

#         self.bijector = bijector

#     def __call__[O: Op](
#         self, fun: Callable[..., InfixRV[O]], *biject_args: RVLike
#     ) -> Callable[..., InfixRV[Transformed[O, B]]]:

#         biject_args = tuple(makerv(a) for a in biject_args)

#         def myfun(*args: RVLike):
#             args = tuple(makerv(a) for a in args)

#             x: InfixRV[O] = fun(*args)

#             if not x.op.random:
#                 raise ValueError(f"Cannot transform non-random op {x.op}")

#             n_biject_args = len(biject_args)
#             transformed_op = Transformed(x.op, self.bijector, n_biject_args)
#             return InfixRV(transformed_op, *biject_args, *x.parents)

#         return myfun


# exp_transform = Transform(ir.ExpBijector())
# """
# A `Transform` instance that applies the exp bijector.

# Examples
# --------
# >>> x = exp_transform(normal)(0,1)
# >>> print(x)
# transformed(normal, exp_bijector)(0, 1)
# """

# mul_bijector = make_bijector(lambda x, a: x * a, lambda y, a: y / a, lambda x, y, a: log(a), (), ())


# def make_transform(forward_fn, inverse_fn, log_det_jac_fn):
#     """
#     Examples
#     --------
#     >>> transform = make_transform(lambda x: exp(x), lambda y: log(y), lambda x, y: y+0)
#     >>> x = transform(normal)(0,1)
#     >>> print(x)
#     transformed(normal, bijector(composite(1, [exp], [[0]]), composite(1, [log], [[0]]), composite(2, [0, add], [[], [1, 2]])))(0, 1)
#     """

#     def transform[O: Op](fun: Callable[..., InfixRV[O]], *biject_args: RVLike):
#         biject_args = tuple(makerv(a) for a in biject_args)
#         biject_arg_shapes = [a.shape for a in biject_args]
#         n_biject_args = len(biject_args)

#         def apply(*args: RVLike):
#             args = tuple(makerv(a) for a in args)
#             x: InfixRV[O] = fun(*args)
#             bijector = make_bijector(forward_fn, inverse_fn, log_det_jac_fn, x.shape, *biject_arg_shapes)
#             if not x.op.random:
#                 raise ValueError(f"Cannot transform non-random op {x.op}")

#             transformed_op = Transformed(x.op, bijector, n_biject_args)
#             return InfixRV(transformed_op, *biject_args, *x.parents)

#         return apply

#     return transform


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
