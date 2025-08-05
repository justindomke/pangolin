"""
A SimpleIndex is a deterministic `Op` that takes (1) an RV to be indexed and (2) a set of (
integer-valued) indexing RVs and returns the result of indexing the first RV with the second.

Conceptually, say the user does something like this:

```python
x = constant([1,2,3])
i = categorical([.1,.2,.7])
y = x[i]
```

Then we would like this to lead to an IR something like:

```python
x = RV(Constant([1,2,3]))
p = RV(Constant([.1,.2,.7])
y = RV(Index(),x,i)
```

Seems simple, right? But there are two major subtleties:

**First subtlety.** Given the above description, you might think that the Index Op itself would
need no parameters—there aren't "different kinds" of indexing, after all. And we want to allow
indices to be random variables.

But there is a problem: We also want to allow indexing with *slices* and if *slices* are defined
with random variables, then the *size* of the output would also be random, which would break our
abstraction that all RVs have fixed shapes.

To deal with this, there are different instances of `Index` Ops, depending on what dimensions
will be sliced. All slices must be baked into the Index `Op` with *fixed* integer values. Then,
all the non-sliced arguments are still free to be `RV`s. So an `Index` Op is created by givin a
list of slices, each can either be a fixed slice, or `None`, indicating that the dimension is
unsliced and will come from a `RV`.

So, for example, if you do this:

```python
x = constant([[1,2,3],[4,5,6])
i = categorical([.1,.2,.7])
y = x[:,i]
```

then under the hood you will get a representation like

```python
x = RV(Constant([1,2,3],[4,5,6]])
p = RV(Constant([.1,.2,.7])
i = RV(Categorical,p)
y = RV(Index(slice(None),None),x,i)
```

**Second subtlety.** Numpy's indexing features are way too complicated. For example, consider this code:

```python
A = np.ones([2,3,4,5])
i = [0,1,0,1,0,1]
A[i,i,:,:].shape # (6,4,5) # ok
A[:,:,i,i].shape # (2,3,6) # ok
A[:,i,i,:].shape # (2,6,5) # fine
A[i,:,:,i].shape # (6,3,4) # ok, I guess
A[:,i,:,i].shape # (6,2,4) # what!?
```

Yes, that is really what happens, try it! What's happening here is that when you have multiple
"advanced indices" (like `i`) above, numpy has very complicated [advanced indexing rules](
https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing).

We don't want to impose a heavy burden on users of the IR. Thus, the rules in Pangolin are:

1. Every dimension must be indexed (either with a slice or RV)
2. All indexing is orthogonal.

This means that the output shape is the shape of all the indices, in order.

**Note** As illustrated above, non-sliced indices are free to be random variables. Although,
currently, JAGS is the only backend that can actually do inference in these settings.
"""

from pangolin.ir import Op, RV, Constant
import numpy as np
from pangolin import util


def _slice_length(size, slice):
    return len(np.ones(size)[slice])

class SimpleIndex(Op):
    """
    Represents an `Op` to index into a `RV`. Slices for all sliced dimensions
    must be baked into the Index op when created. Non sliced dimensions are not
    part of the `Op` (they can be random variables).
    """

    def __init__(self, *slices: slice | None):
        """
        Create an Index op given a set of slices.

        Parameters
        ----------
        slices
            A list with length equal to the number of dimensions of the RV that
            is to be indexed. Each element must either be (a) a slice with fixed
            integer values or (b) None, indicating that the dimension will be
            indexed with a RV instead.
        """
        self.slices = slices
        super().__init__(random=False)

    def _get_shape(self, *shapes):
        var_shape, *indices_shapes = shapes

        if len(self.slices) != len(var_shape):
            raise ValueError(f"number of slots {len(self.slices)} doesn't match number of dims of var {len(var_shape)}")

        num_sliced = len([s for s in self.slices if s is not None])
        num_indexed = len(indices_shapes)
        num_dims = len(var_shape)
        if num_sliced + num_indexed != num_dims:
            raise ValueError(f"Indexed RV with {num_dims} dims with {num_sliced} slices and "
                             f"{num_indexed} indices, but {num_sliced} + {num_indexed} "
                             f"!= {num_dims}.")

        #num_non_scalar = len([idx_shape for idx_shape in indices_shapes if idx_shape is not ()])
        #if num_non_scalar > 1:
        #    raise ValueError(f"Only one RV index can be non-scalar, got {num_non_scalar}")

        output_shape = ()
        m = 0
        for n, my_slice in enumerate(self.slices):
            if my_slice:
                output_shape += (_slice_length(var_shape[n], my_slice),)
            else:
                output_shape += indices_shapes[m]
                m += 1
        assert m == len(indices_shapes)
        return output_shape

    def __repr__(self):
        return "Index(slices=" + repr(self.slices) + ")"

    def __str__(self):
        def slice_str(s):
            match s:
                case None:
                    return "∅"
                case slice(start=None, stop=None, step=None):
                    return ":"
                case slice(start=a, stop=b, step=c):
                    if a is None:
                        a = ""
                    if b is None:
                        b = ""
                    if c is None:
                        c = ""
                    return f"{a}:{b}:{c}"
                case _:
                    raise Exception("not a slice")

        new_slices = tuple(slice_str(s) for s in self.slices)
        return "simple_index" + util.comma_separated(new_slices)

    def __eq__(self, other):
        if isinstance(other, SimpleIndex):
            return self.slices == other.slices
        return False

    def __hash__(self):
        return hash(str(self.slices))



# function to do full orthogonal indexing on regular numpy arrays (for testing)
def index_orthogonal(array, *index_arrays):
    """
    Create orthogonal index arrays for advanced indexing.
    """
    # Calculate final shape

    assert array.ndim == len(index_arrays)

    index_arrays = [
        np.arange(array.shape[i])[arr] if isinstance(arr, slice) else np.array(arr)
        for i, arr in enumerate(index_arrays)
    ]

    index_shapes = [arr.shape for arr in index_arrays]
    total_dims = sum(len(shape) for shape in index_shapes)

    result_arrays = []
    current_dim = 0

    for arr in index_arrays:
        # Create shape with 1s everywhere except for this array's dimensions
        new_shape = [1] * total_dims

        # Place this array's dimensions in the correct position
        for j, dim_size in enumerate(arr.shape):
            new_shape[current_dim + j] = dim_size

        # Reshape and add to result
        reshaped = arr.reshape(new_shape)
        result_arrays.append(reshaped)

        current_dim += len(arr.shape)

    return array[*result_arrays]
