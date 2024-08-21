__docformat__ = "numpy"

import numpy as np
from pangolin.ir import Op


class VecMatOp(Op):
    """
    Convenience class to create "vec mat" distributions that take as input a vector of
    length N, a matrix of size NxN and is a vector of length N
    """

    def __init__(self, name):
        super().__init__(name=name, random=True)

    def _get_shape(self, vec_shape, mat_shape):
        if len(vec_shape) != 1:
            raise ValueError("first parameter must be a vector.")
        if len(mat_shape) != 2:
            raise ValueError("second parameter must be a matrix.")
        N = vec_shape[0]
        if mat_shape != (N,N):
            raise ValueError("second parameter must be matrix with size matching first parameter")
        return (N,)


class MultiNormal(VecMatOp):
    """
    MultiNormal distribution parameterized in terms of the mean and covariance.
    """
    def __init__(self):
        """
        Create a MultiNormal instance. Takes no parameters.
        """
        super().__init__(name="MultiNormal")




class Categorical(Op):
    """
    Categorical distribution parameterized in terms of a 1-d vector of weights.
    """

    def __init__(self):
        """
        Create a Categorical instance. Takes no parameters.
        """
        super().__init__(name="categorical", random=True)

    def _get_shape(self, weights_shape):
        """"""
        assert isinstance(weights_shape,tuple)
        if len(weights_shape) != 1:
            raise ValueError(f"Categorical op got input with {len(weights_shape)} dims but "
                             f"expected 1.")
        return ()


class Multinomial(Op):
    """
    Multinomial distribution parameterized in terms of the number of observations `n` (a scalar)
    and a vector of probabilities `p` (1-D).
    """

    def __init__(self):
        """
        Create a Multinomial instance. Takes no parameters.
        """
        super().__init__(name="multinomial", random=True)

    def _get_shape(self, n_shape, p_shape):
        if n_shape != ():
            raise ValueError("First input to Multinomial op must be scalar")
        if len(p_shape) != 1:
            raise ValueError("Second input to Multinomial op must be a 1-d vector")
        return p_shape



class Dirichlet(Op):
    """Dirichlet distribution parameterized in terms of the concentration"""

    def __init__(self):
        """
        Create a Dirichlet instance. Takes no parameters.
        """
        super().__init__(name="dirichlet", random=True)

    def _get_shape(self, concentration_shape):
        if len(concentration_shape) != 1:
            raise ValueError("Dirichlet op must have a single 1-d vector input")
        return concentration_shape
