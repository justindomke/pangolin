Indexing
========

.. currentmodule:: pangolin.interface

You can index an `RV` with the ``[]`` operator, e.g. as ``A[B,C]``.

Note that indexing with this interface is different (and simpler) than NumPy or JAX:

First, indexing is by default fully-orthogonal. This is done to avoid the `utter insanity <https://numpy.org/doc/stable/user/basics.indexing.html>`_ that is NumPy indexing with broadcasting, basic indexing, advanced indexing, and combinations of basic and advanced indexing. In this interface, if ``A``, ``B``, and ``C`` are RVs, then ``A[B,C].shape == B.shape + C.shape``, and similarly if ``B`` or ``C`` are int / list of (list of) int / numpy array / slice.

Second, all axes must be indexed. For example, if ``A`` is a RV with 3 axes, then ``A[2]`` will trigger an exception. The idea of this is to make code more legible and self-enforcing. Instead you must write ``A[2, :, :]`` or ``A[2, ...]``.

Examples
~~~~~~~~
>>> from pangolin import interface as pi
>>> # Basic indexing
>>> A = pi.constant([9,8,7,6,5,4])
>>> B = A[2]
>>> B.op
Index()
>>> B.parents[0] == A
True
>>> B.parents[1]
RV(Constant(2))

>>> # indexing with a slice
>>> B = A[1::2]
>>> B.op
VMap(Index(), [None, 0], 3)
>>> B.parents[0] == A
True
>>> B.parents[1]
RV(Constant([1,3,5]))

>>> # indexing with a combination of constants and slices
>>> A = pi.constant([[3,4,5],[6,7,8]])
>>> B = A[[1,0],::2]
>>> B.op
VMap(VMap(Index(), [None, None, 0], 2), [None, 0, None], 2)
>>> B.parents[0] == A
True
>>> B.parents[1]
RV(Constant([1,0]))
>>> B.parents[2]
RV(Constant([0,2]))
