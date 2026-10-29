"""Check non-scalar primitive wrappers, coercion, and interface preservation."""

import inspect

import numpy as np
import pytest

from pangolin import functional, ir
from pangolin import interface as pi
from pangolin.interface import Interface

CASES = [
    ("matmul", ir.Matmul(), ([1, 2], [3, 4]), {}, ()),
    ("matmul", ir.Matmul(), ([[1, 2], [3, 4]], [5, 6]), {}, (2,)),
    ("matmul", ir.Matmul(), ([1, 2], [[3, 4, 5], [6, 7, 8]]), {}, (3,)),
    ("matmul", ir.Matmul(), ([[1, 2]], [[3, 4, 5], [6, 7, 8]]), {}, (1, 3)),
    ("transpose", ir.Transpose(), ([[1, 2, 3], [4, 5, 6]],), {}, (3, 2)),
    ("diag", ir.Diag(), ([[1, 2], [3, 4]],), {}, (2,)),
    ("diag_matrix", ir.DiagMatrix(), ([1, 2, 3],), {}, (3, 3)),
    ("inv", ir.Inv(), ([[1, 0], [0, 2]],), {}, (2, 2)),
    ("cholesky", ir.Cholesky(), ([[2, 0], [0, 3]],), {}, (2, 2)),
    ("softmax", ir.Softmax(), ([1, 2, 3],), {}, (3,)),
    ("sum", ir.Sum(-1), ([[1, 2, 3], [4, 5, 6]],), {"axis": -1}, (2,)),
    ("concatenate", ir.Concatenate(1), ([[1], [2]], [[3, 4], [5, 6]]), {"axis": 1}, (2, 3)),
    ("multi_normal", ir.MultiNormal(), ([0, 0], [[1, 0], [0, 1]]), {}, (2,)),
    ("categorical", ir.Categorical(), ([0.2, 0.3, 0.5],), {}, ()),
    ("multinomial", ir.Multinomial(), (10, [0.2, 0.3, 0.5]), {}, (3,)),
    ("dirichlet", ir.Dirichlet(), ([1, 2, 3],), {}, (3,)),
    ("wishart", ir.Wishart(), (5, [[1, 0], [0, 1]]), {}, (2, 2)),
]


@pytest.mark.parametrize("mode", ["off", "simple", "rank", "numpy"])
@pytest.mark.parametrize("name,op,args,kwargs,shape", CASES)
def test_coerced_inputs_build_one_op(mode, name, op, args, kwargs, shape):
    interface = Interface(broadcasting=mode)
    result = getattr(interface, name)(*args, **kwargs)
    assert type(result) is ir.RV
    assert result.op == op
    assert result.shape == shape
    assert result.interface is interface
    assert result.op.random == op.random
    for parent, value in zip(result.parents, args, strict=True):
        assert isinstance(parent.op, ir.Constant)
        np.testing.assert_array_equal(parent.op.value, value)
        assert parent.interface is interface


@pytest.mark.parametrize("name,op,args,kwargs,shape", CASES)
def test_foreign_and_raw_parents_keep_identity(name, op, args, kwargs, shape):
    interface, foreign = Interface(), Interface(broadcasting="numpy")
    parents = tuple(
        foreign.constant(value) if i % 2 == 0 else ir.RV(ir.Constant(value))
        for i, value in enumerate(args)
    )
    result = getattr(interface, name)(*parents, **kwargs)
    assert result.op == op
    assert result.shape == shape
    assert result.interface is interface
    assert all(actual is parent for actual, parent in zip(result.parents, parents, strict=True))
    assert parents[0].interface is foreign


@pytest.mark.parametrize("name,op,args,kwargs,shape", CASES)
def test_default_exports(name, op, args, kwargs, shape):
    assert name in pi.__all__
    function = getattr(pi, name)
    assert function == getattr(pi.default, name)
    assert inspect.signature(function) == inspect.signature(getattr(Interface(), name))
    result = function(*args, **kwargs)
    assert result.op == op
    assert result.shape == shape
    assert result.interface is pi.default


@pytest.mark.parametrize("mode", ["off", "simple", "rank", "numpy"])
@pytest.mark.parametrize(
    "name,args,kwargs",
    [
        ("matmul", (1, [1]), {}),
        ("matmul", ([1, 2], [1, 2, 3]), {}),
        ("matmul", (np.ones((2, 2, 2)), np.eye(2)), {}),
        ("transpose", ([1, 2],), {}),
        ("diag", ([[1, 2, 3], [4, 5, 6]],), {}),
        ("diag_matrix", ([[1, 2]],), {}),
        ("inv", ([[1, 2, 3], [4, 5, 6]],), {}),
        ("cholesky", ([1, 2],), {}),
        ("softmax", ([[1, 2], [3, 4]],), {}),
        ("sum", ([1, 2],), {"axis": 1}),
        ("sum", (1,), {"axis": 0}),
        ("concatenate", (), {}),
        ("concatenate", ([[1, 2]], [[3]]), {}),
        ("multi_normal", ([0, 0], np.eye(3)), {}),
        ("multi_normal", ([[0, 0]], np.eye(2)), {}),
        ("categorical", ([[0.2, 0.8]],), {}),
        ("multinomial", ([10, 20], [0.2, 0.8]), {}),
        ("dirichlet", (1,), {}),
        ("wishart", (5, [1, 2]), {}),
    ],
)
def test_invalid_shapes_are_not_broadcast(mode, name, args, kwargs):
    with pytest.raises(ValueError):
        getattr(Interface(broadcasting=mode), name)(*args, **kwargs)


def test_matrix_syntax_uses_receiver_interface():
    interface, foreign = Interface(broadcasting="off"), Interface(broadcasting="numpy")
    matrix, vector = interface.constant([[1, 2], [3, 4]]), foreign.constant([5, 6])
    result = matrix @ vector
    assert result.op == ir.Matmul()
    assert result.parents == (matrix, vector)
    assert result.interface is interface
    assert matrix.T.op == ir.Transpose()
    assert matrix.T.parents == (matrix,)
    assert matrix.T.interface is interface
    result = [5, 6] @ matrix
    assert result.op == ir.Matmul()
    assert result.parents[1] is matrix
    assert result.parents[0].interface is interface
    assert result.interface is interface


def test_keyword_parameters_and_axis_stay_outside_graph():
    interface = Interface()
    result = interface.multi_normal(cov=np.eye(2), mean=[0, 1])
    assert tuple(parent.shape for parent in result.parents) == ((2,), (2, 2))
    result = interface.sum(axis=-1, x=[[1, 2], [3, 4]])
    assert result.op == ir.Sum(-1)
    assert len(result.parents) == 1
    result = interface.concatenate([1, 2], [3], axis=-1)
    assert result.op == ir.Concatenate(-1)
    assert len(result.parents) == 2


def test_random_and_deterministic_parents_can_be_concatenated():
    interface = Interface()
    x = interface.multi_normal([0, 0], np.eye(2))
    y = interface.softmax(x)
    result = interface.concatenate(x, y)
    assert result.shape == (4,)
    assert result.parents == (x, y)
    assert y.parents == (x,)
    assert result.interface is interface


def test_distribution_can_be_explicitly_vmapped():
    interface = Interface(broadcasting="off")
    means, cov = interface.constant(np.zeros((3, 2))), interface.constant(np.eye(2))
    result = functional.vmap(interface.multi_normal, (0, None))(means, cov)
    assert result.op == ir.VMap(ir.MultiNormal(), (0, None), 3)
    assert result.parents == (means, cov)
    assert result.shape == (3, 2)
    assert result.interface is interface


def test_array_operation_can_be_composited():
    interface = Interface()
    vector = interface.constant([1, 2])
    result = functional.composite(interface.diag_matrix, simplify=False)(vector)
    assert isinstance(result.op, ir.Composite)
    assert result.shape == (2, 2)
    assert result.parents == (vector,)
    assert result.interface is interface
