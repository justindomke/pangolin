import numpy as np
from pangolin import ir

from pangolin.torch_backend import (
    ancestor_sample_flat,
    ancestor_log_prob_flat,
    eval_op,
    sample_op,
    log_prob_op,
)
import torch


def test_add():
    # x -> x+x

    op = ir.Composite(1, [ir.Add()], [[0, 0]])

    out = eval_op(op, [1.5])
    expected = 3.0
    assert np.allclose(out, expected)

    x = ir.RV(ir.Constant(1.5))
    y = ir.RV(op, x)
    assert isinstance(y.op, ir.Composite)

    [out] = ancestor_sample_flat([y], None)
    assert np.allclose(expected, out)


def test_add_mul():
    # x,y -> (x+x)*y
    op = ir.Composite(2, [ir.Add(), ir.Mul()], [[0, 0], [2, 1]])

    x = ir.RV(ir.Constant(3.3))
    y = ir.RV(ir.Constant(4.4))
    z = ir.RV(op, x, y)
    assert isinstance(z.op, ir.Composite)

    expected = (3.3 + 3.3) * 4.4
    out = eval_op(z.op, [3.3, 4.4])
    assert np.allclose(expected, out)

    [out] = ancestor_sample_flat([z], None)
    assert np.allclose(expected, out)


def test_bernoulli():
    op = ir.Composite(1, [ir.Bernoulli()], [[0]])

    x = ir.RV(ir.Constant(0.7))
    y = ir.RV(op, x)
    assert isinstance(y.op, ir.Composite)

    value = sample_op(y.op, [0.7])

    l = log_prob_op(y.op, value, [0.7])

    torch_dist = torch.distributions.Bernoulli(0.7)
    assert isinstance(torch_dist, torch.distributions.Distribution)
    expected = torch_dist.log_prob(value)

    assert np.allclose(l, expected)


def test_exponential():
    op = ir.Composite(1, [ir.Exponential()], [[0]])

    x = ir.RV(ir.Constant(0.7))
    y = ir.RV(op, x)
    assert isinstance(y.op, ir.Composite)

    value = sample_op(y.op, [0.7])

    l = log_prob_op(y.op, value, [0.7])

    torch_dist = torch.distributions.Exponential(0.7)
    expected = torch_dist.log_prob(value)

    assert np.allclose(l, expected)


def test_add_normal():
    # z ~ Normal(x+y, y)
    op = ir.Composite(2, [ir.Add(), ir.Normal()], [[0, 1], [2, 1]])

    x = ir.RV(ir.Constant(0.3))
    y = ir.RV(ir.Constant(0.1))
    z = ir.RV(op, x, y)

    state = torch.get_rng_state()
    value = sample_op(z.op, [0.3, 0.1])
    torch.set_rng_state(state)
    [value2] = ancestor_sample_flat([z])

    assert value == value2

    l = log_prob_op(z.op, value, [0.3, 0.1])
    l2 = ancestor_log_prob_flat([z], [value])

    tmp = 0.3 + 0.1
    torch_dist = torch.distributions.Normal(tmp, 0.1)
    expected = torch_dist.log_prob(value)
    assert np.allclose(l, expected)
    assert np.allclose(l2, expected)
