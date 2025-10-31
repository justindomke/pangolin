from pangolin.torch_backend import (
    log_prob_op,
    sample_op,
    eval_op,
    ancestor_sample_flat,
    ancestor_sample,
)
from pangolin import ir
import numpy as np
import jax
from pangolin.util import tree_allclose
import torch


def test_constant():
    op = ir.Constant(2.0)
    out = eval_op(op, [])
    assert np.allclose(out, 2.0)


def test_normal():
    op = ir.Normal()
    parent_values = [0.5, 1.1]
    value = -0.3

    log_prob = log_prob_op(op, value, parent_values)
    expected = torch.distributions.Normal(*parent_values).log_prob(torch.tensor(value))
    assert np.allclose(log_prob, expected)

    out = sample_op(op, parent_values)
    assert out.shape == ()


def test_add():
    op = ir.Add()
    parent_values = [0.5, 1.1]

    out = eval_op(op, parent_values)
    expected = 1.6
    assert np.allclose(out, expected)


def test_ancestor_sample_flat():
    a = ir.RV(ir.Constant(3.0))
    b = ir.RV(ir.Constant(0.00000001))
    c = ir.RV(ir.Normal(), a, b)

    out = ancestor_sample_flat([a, b, c])
    assert np.allclose(out, [3.0, 0.0, 3.0])


def test_ancestor_sample():
    a = ir.RV(ir.Constant(3.0))
    b = ir.RV(ir.Constant(0.000000001))
    c = ir.RV(ir.Normal(), a, b)

    out1 = ancestor_sample(c)
    assert np.allclose(out1, 3.0)

    out2 = ancestor_sample({"dog": c})
    assert tree_allclose(out2, {"dog": 3.0})

    out3 = ancestor_sample([a, {"dog": (b, c)}])
    assert tree_allclose(out3, [3.0, {"dog": (0.0, 3.0)}])


def test_ancestor_sample_multiple():
    a = ir.RV(ir.Constant(3.0))
    b = ir.RV(ir.Constant(0.000000001))
    c = ir.RV(ir.Normal(), a, b)

    out1 = ancestor_sample(c, 2)
    assert np.allclose(out1, [3.0, 3.0])

    out2 = ancestor_sample({"dog": c}, 2)
    assert tree_allclose(out2, {"dog": torch.tensor([3.0, 3.0])})

    out3 = ancestor_sample([a, {"dog": (b, c)}])
    assert tree_allclose(
        out3,
        [
            torch.tensor([3.0, 3.0]),
            {"dog": (torch.tensor([0.0, 0.0]), torch.tensor([3.0, 3.0]))},
        ],
    )
