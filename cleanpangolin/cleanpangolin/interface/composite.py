from cleanpangolin.ir import Composite
from cleanpangolin.interface import OperatorRV
from .vmap import generated_nodes, AbstractOp

def make_composite(fun, *input_shapes):
    """
    Inputs:
    - fun: A function that takes RVs as inputs and returns a single RV
    - *input_shapes: shapes for each explicit input to fun
    Outputs:
    - op: a Composite op representing the function
    - consts: a list of constants that fun captures as a closure
    The final op expects as inputs *first* all the elements of consts and then the explicit inputs

    Fun should not examine ops of its inputs.
    """

    # TODO: take pytree of arguments
    # TODO: don't like passing random=False
    dummy_args = [OperatorRV(AbstractOp(shape,random=False)) for shape in input_shapes]

    f = lambda *args: [fun(*args)] # vmap_generated_nodes runs on "flat" functions
    all_vars, [out] = generated_nodes(f, *dummy_args)
    assert isinstance(out, OperatorRV), "output of function must be a single OperatorRV"

    ops = []
    par_nums = []
    linear_order = {}

    for var in dummy_args:
        linear_order[var] = dummy_args.index(var)

    current_position = len(dummy_args)

    consts = []
    for var in all_vars:
        for p in var.parents:
            if p not in all_vars and p not in linear_order:
                linear_order[p] = current_position
                current_position += 1
                consts.append(p)

    num_inputs = current_position

    for var in all_vars:
        my_op = var.op
        my_par_nums = tuple(linear_order[p] for p in var.parents)
        ops.append(my_op)
        par_nums.append(my_par_nums)
        linear_order[var] = current_position
        current_position += 1

    return Composite(num_inputs, tuple(ops), tuple(par_nums)), consts

def composite(fun):
    from cleanpangolin.interface.interface import rv_factory
    def myfun(*inputs):
        input_shapes = [x.shape for x in inputs]
        op, consts = make_composite(fun,*input_shapes)
        return rv_factory(op, *consts, *inputs)
    return myfun
