"""
Various assorted multivariate deterministic functions
"""

__docformat__ = "numpy"

import numpy as np
from pangolin.ir import Op


class MatMul(Op):
    """
    A class that does matrix multiplication, following the rules of `numpy.matmul`.
    Currently only 1d and 2d arrays are supported.
    """

    def __init__(self):
        super().__init__(name="MatMul", random=False)

    def _get_shape(self, a_shape, b_shape):
        # could someday generalize to handle more dimensions
        assert len(a_shape) >= 1, "args to @ must have at least 1 dim"
        assert len(b_shape) >= 1, "args to @ must have at least 1 dim"
        assert len(a_shape) <= 2, "args to @ must have at most 2 dims"
        assert len(b_shape) <= 2, "args to @ must have at most 2 dims"

        # https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
        # The behavior depends on the arguments in the following way.
        # * If both arguments are 2-D they are multiplied like conventional matrices.
        # * If either argument is N-D, N > 2, it is treated as a stack of matrices
        #   residing in the last two indexes and broadcast accordingly.
        # * If the first argument is 1-D, it is promoted to a matrix by prepending a
        #   1 to its dimensions. After matrix multiplication the prepended 1 is removed.
        # * If the second argument is 1-D, it is promoted to a matrix by appending a
        #   1 to its dimensions. After matrix multiplication the appended 1 is removed.

        if len(a_shape) == 1 and len(b_shape) == 1:
            # inner product
            assert a_shape == b_shape
            return ()
        elif len(a_shape) == 1 and len(b_shape) == 2:
            # vector-matrix product
            assert a_shape[0] == b_shape[0]
            return (b_shape[1],)
        elif len(a_shape) == 2 and len(b_shape) == 1:
            # matrix-vector product
            assert a_shape[1] == b_shape[0]
            return (a_shape[0],)
        elif len(a_shape) == 2 and len(b_shape) == 2:
            # matrix-matrix product
            assert a_shape[1] == b_shape[0]
            return (a_shape[0], b_shape[1])
        else:
            raise Exception("bug: should be impossible")


class Inv(Op):
    """
    Take the inverse of a square matrix
    """

    def __init__(self):
        super().__init__(name="Inv", random=False)

    def _get_shape(self, *parents):
        assert len(parents) == 1
        p_shape = parents[0]
        assert len(p_shape) == 2, "inverse only applies to 2d arrays"
        assert p_shape[0] == p_shape[1], "inverse only for square 2d arrays"
        return p_shape


class Softmax(Op):
    """
    Softmax
    """

    def __init__(self):
        super().__init__(name="Softmax", random=False)

    def _get_shape(self, *parents):
        assert len(parents) == 1
        p_shape = parents[0]
        assert len(p_shape) == 1, "input to softmax would be 1d"
        return p_shape


class Sum(Op):
    """Take the sum of an array over some axis"""

    def __init__(self, axis):
        """
        Create a Sum instance
        Parameters
        ----------
        axis: int
            What axis to sum over.
        """
        if isinstance(axis, np.ndarray) and axis.shape == ():
            axis = int(axis)
        if not isinstance(axis, int):
            raise ValueError("axis argument for Sum must be a fixed integer")
        self.axis = axis
        super().__init__(name="Sum", random=False)

    def _get_shape(self, x_shape):
        if self.axis is None:
            return ()
        else:
            return x_shape[: self.axis] + x_shape[self.axis + 1 :]

    def __repr__(self):
        return f"Sum(axis={self.axis})"

    def __str__(self):
        return f"sum(axis={self.axis})"

    def __eq__(self, other):
        if isinstance(other, Sum):
            return self.axis == other.axis
        return False

    def __hash__(self):
        return hash(self.axis)
