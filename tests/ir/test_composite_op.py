import numpy as np
import pytest

from pangolin import ir
from pangolin.ir import Composite


def test_plain_normal():
    op = Composite(2, (ir.Normal(),), ((0, 1),))
    assert op.get_shape((), ()) == ()
    assert op.random


def test_single_input():
    op = Composite(1, (ir.Normal(),), ((0, 0),))
    assert op.get_shape(()) == ()
    assert op.random


def test_diag_normal():
    # def f(x0,x1):
    #     x2 = np.arange(5)
    #     x3 = x1 * x2
    #     x4 = normal(x0, x3)
    op = Composite(
        2, (ir.Constant(np.arange(5)), ir.VMap(ir.Mul(), (None, 0)), ir.VMap(ir.Normal(), (0, 0))), ((), (1, 2), (0, 3)),
    )
    assert op.get_shape((5,), ()) == (5,)
    assert op.random


def test_multivariate_normal():
    "Multivariate normal with mean zero and cov a*I"
    scalar_matrix_mul = ir.VMap(ir.VMap(ir.Mul(), (None, 0)), (None, 0))

    op = Composite(
        1, (ir.Constant(np.ones(5)), ir.Constant(np.eye(5)), scalar_matrix_mul, ir.MultiNormal()), ((), (), (0, 2), (1, 3))
    )
    assert op.get_shape(()) == (5,)
    assert op.random


def test_accepts_zero_inputs_and_references_to_previous_outputs():
    op = Composite(0, (ir.Constant(2), ir.Exp()), ((), (0,)))

    assert op.get_shape() == ()


def test_accepts_repeated_and_reordered_parent_references():
    repeated = Composite(1, (ir.Add(),), ((0, 0),))
    reordered = Composite(2, (ir.Sub(),), ((1, 0),))

    assert repeated.get_shape(()) == ()
    assert reordered.get_shape((), ()) == ()


def test_normalizes_numpy_integers_and_mutable_parent_sequences():
    ops = (ir.Identity(),)
    par_nums = [[np.int64(0)]]
    op = Composite(np.int64(1), ops, par_nums)

    par_nums[0].clear()

    assert op.num_inputs == 1
    assert op.ops == (ir.Identity(),)
    assert op.par_nums == ((0,),)


def test_copies_a_mutable_ops_sequence():
    ops = [ir.Identity()]
    op = Composite(1, ops, [[0]])
    ops.clear()
    assert op.ops == (ir.Identity(),)
    assert op == Composite(1, (ir.Identity(),), [[0]])
    assert hash(op) == hash(Composite(1, (ir.Identity(),), [[0]]))


def test_repr_uses_lists_and_round_trips():
    op = Composite(2, [ir.Add(), ir.Normal()], [[0, 1], [2, 1]])
    assert repr(op) == "Composite(2, [Add(), Normal()], [[0, 1], [2, 1]])"
    assert eval(repr(op), dict(vars(ir))) == op


@pytest.mark.parametrize("num_inputs", [True, 1.5, "1"])
def test_requires_integer_num_inputs(num_inputs):
    with pytest.raises(TypeError, match="num_inputs must be an integer"):
        Composite(num_inputs, (ir.Identity(),), ((0,),))


def test_rejects_negative_num_inputs():
    with pytest.raises(ValueError, match="num_inputs must be non-negative"):
        Composite(-1, (ir.Identity(),), ((0,),))


def test_requires_nonempty_ops():
    with pytest.raises(ValueError, match="at least one Op"):
        Composite(0, (), ())


def test_requires_ops_to_be_operations():
    with pytest.raises(TypeError, match="is not an Op"):
        Composite(1, (ir.Identity(), "exp"), ((0,), (1,)))  # type: ignore[arg-type]


@pytest.mark.parametrize("par_nums", [(), ((0,), (1,))])
def test_requires_one_parent_entry_per_op(par_nums):
    with pytest.raises(ValueError, match="1 Ops but .* par_nums entries"):
        Composite(1, (ir.Identity(),), par_nums)


@pytest.mark.parametrize("par_nums", [None, (0,)])
def test_requires_par_nums_to_be_nested_sequences(par_nums):
    with pytest.raises(TypeError, match="par_nums must be a sequence of sequences"):
        Composite(1, (ir.Identity(),), par_nums)


@pytest.mark.parametrize("parent_number", [True, 0.5, "0"])
def test_requires_integer_parent_numbers(parent_number):
    with pytest.raises(TypeError, match="each parent number must be an integer"):
        Composite(1, (ir.Identity(),), ((parent_number,),))


@pytest.mark.parametrize(
    ("num_inputs", "ops", "par_nums", "operation_number", "value_number"),
    [
        (1, (ir.Identity(),), ((-1,),), 0, -1),
        (1, (ir.Identity(),), ((1,),), 0, 1),
        (1, (ir.Identity(), ir.Identity()), ((0,), (2,)), 1, 2),
    ],
)
def test_rejects_unavailable_parent_references(num_inputs, ops, par_nums, operation_number, value_number):
    with pytest.raises(
        ValueError,
        match=f"operation {operation_number} refers to value {value_number}",
    ):
        Composite(num_inputs, ops, par_nums)


def test_requires_intermediate_ops_to_be_deterministic():
    with pytest.raises(ValueError, match="all but the last Op must be deterministic"):
        Composite(1, (ir.Normal(), ir.Identity()), ((0, 0), (1,)))


@pytest.mark.parametrize("parent_shapes", [(), ((),), ((), ()), ((), (), ())])
def test_get_shape_requires_exact_input_count(parent_shapes):
    op = Composite(1, (ir.Identity(),), ((0,),))

    if len(parent_shapes) == 1:
        assert op.get_shape(*parent_shapes) == ()
    else:
        with pytest.raises(ValueError, match=f"expects 1 input shapes but received {len(parent_shapes)}"):
            op.get_shape(*parent_shapes)
