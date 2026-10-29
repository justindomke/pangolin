# pyright: reportUnnecessaryTypeIgnoreComment=error

"""Check the user-facing signatures of non-scalar operations."""

from typing import TYPE_CHECKING, assert_type

from pangolin import ir
from pangolin import interface as pi


def test_non_scalar_return_types():
    for interface in (pi.default, pi.Interface(broadcasting="off")):
        vector = interface.constant([1, 2])
        matrix = interface.constant([[1, 0], [0, 1]])
        assert_type(interface.matmul(matrix, vector), ir.RV)
        assert_type(interface.transpose(matrix), ir.RV)
        assert_type(interface.diag(matrix), ir.RV)
        assert_type(interface.diag_matrix(vector), ir.RV)
        assert_type(interface.inv(matrix), ir.RV)
        assert_type(interface.cholesky(matrix), ir.RV)
        assert_type(interface.softmax(vector), ir.RV)
        assert_type(interface.sum(vector, axis=0), ir.RV)
        assert_type(interface.concatenate(vector, [3], axis=0), ir.RV)
        assert_type(interface.multi_normal(cov=matrix, mean=vector), ir.RV)
        assert_type(interface.categorical([0.5, 0.5]), ir.RV)
        assert_type(interface.multinomial(n=10, p=[0.5, 0.5]), ir.RV)
        assert_type(interface.dirichlet(vector), ir.RV)
        assert_type(interface.wishart(nu=5, scale=matrix), ir.RV)
        assert_type(matrix @ vector, ir.RV)
        assert_type(matrix.T, ir.RV)

    assert_type(pi.multi_normal([0, 0], [[1, 0], [0, 1]]), ir.RV)
    assert_type(pi.concatenate([1], [2]), ir.RV)


if TYPE_CHECKING:
    pi.matmul([1])  # type: ignore
    pi.multi_normal(mean=[0, 0], covariance=[[1, 0], [0, 1]])  # type: ignore
    pi.sum([1, 2])  # type: ignore
    pi.sum([1, 2], axis="0")  # type: ignore
    pi.concatenate([1], axis="0")  # type: ignore
    pi.softmax([1, 2], axis=0)  # type: ignore
    pi.wishart(5, [[1]], unexpected=1)  # type: ignore
