from pangolin.ir import Composite
from pangolin.interface import OperatorRV, makerv
from .vmap import generated_nodes, AbstractOp
from pangolin import util
import jax.tree_util

def make_composite(fun, *input_shapes):
    """
    Inputs:
    - fun: A function that takes RVs as inputs and returns a single RV
    - *input_shapes: shapes for each explicit input to fun
    Outputs:
    - op: a Composite op representing the function
    - consts: a list of constants that fun captures as a closure
    The final op expects as inputs *first* all the explicit inputs and then all the elements of consts

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

def composite_flat(fun):
    from pangolin.interface.interface import rv_factory
    def myfun(*inputs):
        input_shapes = [x.shape for x in inputs]
        op, consts = make_composite(fun,*input_shapes)
        return rv_factory(op, *consts, *inputs)
    return myfun

def composite(fun):
    from pangolin.interface.interface import rv_factory
    def myfun(*inputs):
        # this casts at the SMALLEST level - [0,0,0] becomes three scalars, not a vector
        # can't do more because level of granularity unclear
        inputs = jax.tree_util.tree_map(makerv, inputs)
        flat_fun, flatten_input, unflatten_output = util.flatten_fun(fun, *inputs)

        # remove first output, not list
        new_flat_fun = lambda *args: flat_fun(*args)[0]

        flat_inputs = flatten_input(*inputs)

        test_out = new_flat_fun(*flat_inputs)
        print(f"{test_out=}")

        flat_input_shapes = [x.shape for x in flat_inputs]
        op, consts = make_composite(new_flat_fun, *flat_input_shapes)
        return rv_factory(op, *flat_inputs, *consts)

    return myfun

