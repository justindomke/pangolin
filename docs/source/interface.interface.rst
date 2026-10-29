``Interface``
=============

`Interface` creates a separate interface with its own broadcasting rules. It provides the same
model-building functions as the default interface, without changing that default interface or
any other instance.

The default interface uses simple broadcasting, which rejects these different array shapes:

>>> from pangolin import interface as pi
>>> pi.normal([[0], [1]], [1, 2, 3])
Traceback (most recent call last):
    ...
ValueError: Can't broadcast shapes (2, 1) and (3,) with broadcasting=SIMPLE

Instead of importing the default interface as ``pi``, create one with NumPy broadcasting:

>>> import pangolin
>>> pi = pangolin.interface.Interface(broadcasting="numpy")
>>> x = pi.normal([[0], [1]], [1, 2, 3])
>>> x.shape
(2, 3)

Use this ``pi`` in the same way as the default interface. Here NumPy broadcasting combines
shapes ``(2, 1)`` and ``(3,)`` to produce shape ``(2, 3)``.

.. note::

   Array operations and distributions with non-scalar parameters use the IR's shape rules without
   implicit broadcasting, regardless of this interface's scalar broadcasting configuration.

Combinators work the same way as in the default interface.

Use ``pi.overload(...)`` for a temporary, context-local change. Saved functions observe the active
configuration when called:

>>> normal = pi.normal
>>> with pi.overload(broadcasting="off"):
...     y = normal(0, 1)
...     y.shape
()
>>> normal([0, 1], [[1, 1], [2, 2]]).shape
(2, 2)

See :meth:`~pangolin.interface.Interface.overload` for the full behavior of temporary overrides.

Interface
---------

.. autoclass:: pangolin.interface.Interface

.. currentmodule:: pangolin.interface

.. autosummary::
   :toctree: generated/

   Interface.log_prob
   Interface.cdf
   Interface.quantile
   Interface.truncated
   Interface.grad
   Interface.vjp
   Interface.broadcast
   Interface.get_attribute
   Interface.overload
   Interface.neg
   Interface.add
   Interface.sub
   Interface.mul
   Interface.div
   Interface.pow
   Interface.arccos
   Interface.arccosh
   Interface.arcsin
   Interface.arcsinh
   Interface.arctan
   Interface.arctanh
   Interface.cos
   Interface.cosh
   Interface.sin
   Interface.sinh
   Interface.tan
   Interface.tanh
   Interface.abs
   Interface.minimum
   Interface.maximum
   Interface.exp
   Interface.inv_logit
   Interface.log
   Interface.logit
   Interface.step
   Interface.loggamma
   Interface.normal
   Interface.normal_prec
   Interface.lognormal
   Interface.bernoulli
   Interface.bernoulli_logit
   Interface.binomial
   Interface.cauchy
   Interface.uniform
   Interface.beta
   Interface.exponential
   Interface.gamma
   Interface.poisson
   Interface.beta_binomial
   Interface.student_t
   Interface.expit
   Interface.sigmoid
   Interface.sqrt
   Interface.matmul
   Interface.transpose
   Interface.diag
   Interface.diag_matrix
   Interface.inv
   Interface.cholesky
   Interface.softmax
   Interface.sum
   Interface.concatenate
   Interface.fill_tril
   Interface.extract_tril
   Interface.multi_normal
   Interface.categorical
   Interface.multinomial
   Interface.dirichlet
   Interface.wishart
   Interface.getitem
   Interface.oindex
   Interface.vindex
   Interface.loop
   Interface.seq
   Interface.vfor
   Interface.sfor
   Interface.constant
   Interface.coerce
