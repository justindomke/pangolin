from cleanpangolin.ir import Op, RV, Constant
import numpy as np
from cleanpangolin import util
from typing import Sequence
from cleanpangolin import ir

class Composite(Op):
    def __init__(self, num_inputs: int, ops: tuple[Op,...], par_nums: tuple[tuple[int, ...],...]):
        assert isinstance(num_inputs, int)
        assert all(isinstance(d, Op) for d in ops)
        for my_par_nums in par_nums:
            assert all(isinstance(i, int) for i in my_par_nums)
        for d in ops[:-1]:
            assert not d.random, "all but last op for Composite must be non-random"
        self.num_inputs = num_inputs
        self.ops = tuple(ops)
        self.par_nums = tuple(par_nums)
        super().__init__(name=f"Composite({ops[-1].name})", random=ops[-1].random)

    def _get_shape(self, *parents_shapes):
        all_shapes = list(parents_shapes)
        for my_op, my_par_nums in zip(self.ops, self.par_nums):
            my_parents_shapes = [all_shapes[i] for i in my_par_nums]
            my_shape = my_op.get_shape(*my_parents_shapes)
            all_shapes.append(my_shape)
        return all_shapes[-1]

    def __str__(self):
        return f"composite({self.num_inputs},{self.ops},{self.par_nums})"

    def __repr__(self):
        return f"Composite({self.num_inputs},{self.ops},{self.par_nums})"