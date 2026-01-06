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

# handle_vmap_nonrandom(op: ir.VMap, *numpyro_parents, is_observed):


def test_handle_nonrandom_exp():
    op = ir.VMap(ir.Exp(), (None,), 5)
    x = np.array(1.0)
    out = eval_op(op, [x])
    expected = np.exp(1.0) * np.ones(5)
    assert np.allclose(out, expected)

    y = ir.RV(op, ir.RV(ir.Constant(x)))
    [out2] = ancestor_sample_flat([y], None)
    assert np.allclose(out2, expected)


def test_handle_nonrandom_exp_2d():
    op = ir.VMap(ir.VMap(ir.Exp(), (None,), 5), (0,))
    x = np.array([1.0, 2.0])
    out = eval_op(op, [x])
    expected = np.exp(np.array([1.0, 2.0]))[:, None] * np.ones((2, 5))
    assert np.allclose(out, expected)


def test_handle_nonrandom_add():
    op = ir.VMap(
        ir.Add(),
        (
            None,
            0,
        ),
    )
    x = np.array(1.0)
    y = np.array([2.0, 3.0, 4.0])
    z = eval_op(op, [x, y])
    expected = x + y
    assert np.allclose(z, expected)


def test_handle_nonrandom_add_2d():
    op = ir.VMap(ir.VMap(ir.Add(), (None, 0)), (None, 0))
    x = np.array(1.0)
    y = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    z = eval_op(op, [x, y])
    expected = x + y
    assert np.allclose(z, expected)


def test_normal_iid():
    op = ir.VMap(ir.Normal(), [None, None], 5)
    mu = torch.tensor(1.0)
    sigma = torch.tensor(1.0)

    def my_sample(dummy):
        return torch.distributions.Normal(mu, sigma).sample()

    state = torch.get_rng_state()
    y = sample_op(op, [mu, sigma])
    torch.set_rng_state(state)
    expected = torch.vmap(my_sample, randomness="different")(torch.ones(5))
    assert np.allclose(y, expected)

    def my_log_prob(v):
        return torch.distributions.Normal(mu, sigma, validate_args=False).log_prob(v)

    l = log_prob_op(op, y, [mu, sigma])
    expected_l = torch.sum(torch.vmap(my_log_prob)(y))
    assert np.allclose(l, expected_l)


def test_normal_non_iid():
    op = ir.VMap(ir.Normal(), [0, 0])
    mu = torch.tensor([1.0, 2.0, 3.0])
    sigma = torch.tensor([3.0, 2.0, 1.0])

    def my_sample(mu, sigma):
        return torch.distributions.Normal(mu, sigma, validate_args=False).sample()

    state = torch.get_rng_state()
    y = sample_op(op, [mu, sigma])
    torch.set_rng_state(state)
    expected = torch.vmap(my_sample, randomness="different")(mu, sigma)
    assert np.allclose(y, expected)

    def my_log_prob(v, mu, sigma):
        return torch.distributions.Normal(mu, sigma, validate_args=False).log_prob(v)

    l = log_prob_op(op, y, [mu, sigma])
    expected_l = torch.sum(torch.vmap(my_log_prob)(y, mu, sigma))
    assert np.allclose(l, expected_l)


def test_normal_iid_2d():
    op = ir.VMap(ir.VMap(ir.Normal(), [None, None], 5), [None, None], 3)
    mu = torch.tensor(1.0)
    sigma = torch.tensor(1.0)
    state = torch.get_rng_state()
    y = sample_op(op, [mu, sigma])

    def my_sample0(dummy):
        return torch.distributions.Normal(mu, sigma, validate_args=False).sample()

    def my_sample1(dummy):
        return torch.vmap(my_sample0, randomness="different")(torch.ones(5))

    dummy = torch.ones(3)
    torch.set_rng_state(state)
    expected = torch.vmap(my_sample1, randomness="different")(dummy)

    assert np.allclose(y, expected)
