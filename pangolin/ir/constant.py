__docformat__ = 'numpy'

"""The `Constant` class represents constant functions. You can create an *instance* of this class
with `Constant(value)` where `value` is a numpy array or something that can be cast to a numpy
array.
"""

import numpy as np
from pangolin import ir

class Constant(ir.Op):
    """
    Represents a "constant" distribution. Has no parents. Data is always stored as a
    numpy array. You can switch it to use jax's version of numpy by setting `ir.np =
    jax.numpy`.
    """

    def __init__(self, value):
        """
        Create a Constant distribution.
        Parameters
        ----------
        value
            Some constant value that is either a numpy array or something that can be casted to a
            numpy array.
        """
        self.value = np.array(value)
        """The actual stored data, stored as an immutable numpy array"""
        self.value.flags.writeable = False
        super().__init__(name="constant", random=False)

    def _get_shape(self,*parents_shapes):
        """"""
        if len(parents_shapes) != 0:
            raise ValueError(f"Constant got {len(parents_shapes)} arguments but expected 0.")
        return self.value.shape

    def __eq__(self, other):
        if isinstance(other, Constant):
            if self.value.shape == other.value.shape and np.all(
                self.value == other.value
            ) and self.value.dtype == other.value.dtype:
                assert hash(self) == hash(other), "hashes don't match for equal Constant"
                return True
        return False

    def __hash__(self):
        return hash(self.value.tobytes())

    def __repr__(self):
        # assure regular old numpy in case jax being used
        if self.value.ndim > 0 and np.max(self.value.shape) > 5:
            ret = "Constant("
            with np.printoptions(threshold=5, linewidth=50, edgeitems=2):
                ret += np.array2string(self.value)
            ret += ')'
            return ret

        numpy_value = np.array(self.value)
        array_str = repr(numpy_value)  # get base string
        array_str = array_str[6:-1]  # cut off "array(" and ")"
        array_str = array_str.replace("\n", "")  # remove newlines
        array_str = array_str.replace(" ", "")  # remove specs
        return "Constant(" + array_str + ")"

    def __str__(self):
        # return str(self.value).replace("\n", "").replace("  ", " ")
        # assure regular old numpy in case jax being used
        numpy_value = np.array(self.value)
        with np.printoptions(threshold=5, linewidth=50, edgeitems=2):
            return np.array2string(numpy_value, precision=3).replace("\n", "").replace("  ", " ")
