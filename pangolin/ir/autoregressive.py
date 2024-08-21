from pangolin.ir import Op, RV, Constant
import numpy as np
from pangolin import util
from typing import Sequence
from pangolin import ir

# TODO:
# maybe remove convenient form for in_axes—make everything explicit
# interface can simplify

class Autoregressive(Op):
    def __init__(
        self,
        base_op: Op,
        length: int,
        in_axes: tuple[int | None, ...] | list[int | None],
        where_self: int = 0,
    ):
        """
        base_cond_dist - what distribution to repeat on
        num_constants - number of constant arguments (default 0)
        length - the number of times to repeat (optional if there are )
        """
        self.base_op = base_op
        # self.num_constants = num_constants
        self.length = length
        self.in_axes = tuple(in_axes)
        self.where_self = where_self
        super().__init__(name=f"Autoregressive({base_op.name})", random=base_op.random)

    def _get_shape(self, start_shape, *other_shapes):
        # const_shapes = other_shapes[: self.num_constants]
        # other_shapes = other_shapes[self.num_constants :]

        # if self.length is None and other_shapes == ():
        #     raise ValueError("Can't create Autoregressive with length=None and no mapped arguments")
        #
        # if self.length is None:
        #     my_length = other_shapes[0][0]
        # else:
        #     my_length = self.length

        base_input_shapes = []

        for n, (s, ax) in enumerate(zip(other_shapes,self.in_axes,strict=True)):
            if ax is None:
                print('NONE')
                base_input_shapes.append(s)
            else:
                print('NOT')
                assert isinstance(ax,int)
                base_input_shapes.append(s[:ax] + s[ax + 1 :])

        # insert self
        # if n == self.where_self:
        #    base_input_shapes.append(start_shape)
        base_input_shapes = (
            base_input_shapes[: self.where_self]
            + [start_shape]
            + base_input_shapes[self.where_self :]
        )

        # for s in other_shapes:
        #    assert s[0] == my_length
        # base_other_shapes = tuple(s[1:] for s in other_shapes)

        # base_input_shapes = (start_shape,) + const_shapes + base_other_shapes
        base_output_shape = self.base_op.get_shape(*base_input_shapes)
        output_shape = (self.length,) + base_output_shape
        return output_shape

    def __eq__(self, other):
        if isinstance(other, Autoregressive):
            return (
                self.base_op == other.base_op
                and self.length == other.length
                and self.in_axes == other.in_axes
                and self.where_self == other.where_self
            )
        return False

    def __hash__(self):
        return hash((self.base_op, self.length, self.in_axes, self.where_self))

    def __str__(self):
        return f"autoregressive({self.base_op},{self.length},{self.in_axes},{self.where_self})"

    def __repr__(self):
        return f"Autoregressive({self.base_op},{self.length},{self.in_axes},{self.where_self})"


# class Autoregressive(Op):
#     def __init__(self, base_op:Op, length=None, num_constants=0):
#         """
#         base_cond_dist - what distribution to repeat on
#         num_constants - number of constant arguments (default 0)
#         length - the number of times to repeat (optional if there are )
#         """
#         self.base_op = base_op
#         self.num_constants = num_constants
#         self.length = length
#         super().__init__(name=f"Autoregressive({base_op.name})", random=base_op.random)
#
#     def _get_shape(self, start_shape, *other_shapes):
#         const_shapes = other_shapes[:self.num_constants]
#         other_shapes = other_shapes[self.num_constants:]
#
#         if self.length is None and other_shapes == ():
#             raise ValueError("Can't create Autoregressive with length=None and no mapped arguments")
#
#         if self.length is None:
#             my_length = other_shapes[0][0]
#         else:
#             my_length = self.length
#
#         for s in other_shapes:
#             assert s[0] == my_length
#         base_other_shapes = tuple(s[1:] for s in other_shapes)
#
#         base_input_shapes = (start_shape,) + const_shapes + base_other_shapes
#         base_output_shape = self.base_op.get_shape(*base_input_shapes)
#         output_shape = (my_length,) + base_output_shape
#         return output_shape
