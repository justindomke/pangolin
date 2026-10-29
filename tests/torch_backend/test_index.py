"""Check scalar indexing validation, batching, and derivatives."""

import pytest
import torch

from pangolin import ir, torch_backend


def test_scalar_and_negative_indices():
    array = torch.arange(6).reshape(2, 3)
    result = torch_backend.eval_op(ir.Index(), (array, -1, 1))
    assert result.shape == ()
    assert result == 4
    assert torch_backend.index(torch.tensor(7)) == 7


@pytest.mark.parametrize("indices", [(), (0,), (0, 0, 0)])
def test_helper_rejects_wrong_index_count(indices):
    with pytest.raises(ValueError, match="one index per array dimension"):
        torch_backend.index(torch.ones((2, 3)), *(torch.tensor(i) for i in indices))


@pytest.mark.parametrize("indices", [([0], 1), (0, [1, 2]), ([[0]], 1)])
def test_helper_rejects_nonscalar_indices(indices):
    with pytest.raises(ValueError, match="scalar indices"):
        torch_backend.index(torch.ones((2, 3)), *(torch.tensor(i) for i in indices))


def test_helper_rejects_float_indices():
    with pytest.raises((IndexError, TypeError)):
        torch_backend.index(torch.arange(3), torch.tensor(1.0))


def test_vmap_with_dynamic_indices():
    array = torch.arange(6).reshape(2, 3)
    rows, columns = torch.tensor([1, 0]), torch.tensor([2, 0, 1])
    op = ir.VMap(ir.VMap(ir.Index(), (None, None, 0), 3), (None, 0, None), 2)
    result = torch_backend.eval_op(op, (array, rows, columns))
    torch.testing.assert_close(result, torch.tensor([[5, 3, 4], [2, 0, 1]]))
    paired = torch.vmap(torch_backend.index, in_dims=(None, 0, 0))
    torch.testing.assert_close(paired(array, rows, torch.tensor([0, 2])), torch.tensor([3, 2]))


def test_grad_accumulates_repeated_selections():
    array = torch.arange(4.0, requires_grad=True)
    indices = torch.tensor([2, 0, 2])
    op = ir.VMap(ir.Index(), (None, 0), 3)
    torch_backend.eval_op(op, (array, indices)).sum().backward()
    torch.testing.assert_close(array.grad, torch.tensor([1.0, 0.0, 2.0, 0.0]))
