interface
=========

.. testsetup::

   from pangolin.interface import *

.. automodule:: pangolin.interface

.. currentmodule:: pangolin.interface

.. toctree::
   :hidden:

   api/interface.RVLike

Constants
---------

.. autosummary::
   :toctree: generated/

   constant


Scalar arithmetic and mathematics
---------------------------------

.. autosummary::
   :toctree: generated/

   add
   sub
   mul
   div
   pow
   sqrt
   abs
   minimum
   maximum
   arccos
   arccosh
   arcsin
   arcsinh
   arctan
   arctanh
   cos
   cosh
   exp
   inv_logit
   expit
   sigmoid
   log
   loggamma
   logit
   sin
   sinh
   step
   tan
   tanh

Array operations and linear algebra
-----------------------------------

.. autosummary::
   :toctree: generated/

   matmul
   transpose
   inv
   cholesky
   diag
   diag_matrix
   softmax
   concatenate
   sum
   fill_tril
   extract_tril

Indexing
--------

See :doc:`interface.indexing` for indexing rules and examples.

.. toctree::
   :hidden:

   interface.indexing


.. autosummary::
   :toctree: generated/

   oindex
   vindex

Scalar distributions
--------------------

.. autosummary::
   :toctree: generated/

   normal
   normal_prec
   lognormal
   cauchy
   bernoulli
   bernoulli_logit
   binomial
   uniform
   beta
   beta_binomial
   exponential
   gamma
   poisson
   student_t

Multivariate distributions
--------------------------

.. autosummary::
   :toctree: generated/

   multi_normal
   categorical
   multinomial
   dirichlet
   wishart

Bijected distributions
-------------------------


.. list-table::

   * - :class:`Transform`
     - Construct transformed distributions using an invertible transformation.
   * - :func:`compose_transforms`
     - Compose Transforms in left-to-right application order.
   * - :doc:`tforms <interface.tforms>`
     - Standard transform library.

.. toctree::
   :hidden:

   api/interface.transform
   api/interface.compose_transforms

.. _interface-combinators:

Combinators
-----------

The combinators and Transform machinery are also available in :mod:`pangolin.functional`.
For their IR-level contracts, see :doc:`functional`.

.. list-table::

   * - :func:`broadcast`
     - Broadcast a scalar function.
   * - :func:`vmap`
     - Map a function over argument axes.
   * - :func:`loop`
     - Build a loop with vectorized and sequential axes.
   * - :func:`seq`
     - Specify a sequential loop axis.
   * - :func:`vfor`
     - Build a vectorized indexed loop.
   * - :func:`sfor`
     - Build a sequential indexed loop.
   * - :func:`log_prob`
     - Construct a log-probability graph.
   * - :func:`cdf`
     - Construct a cumulative-probability graph.
   * - :func:`quantile`
     - Construct a scalar quantile graph.
   * - :func:`truncated`
     - Restrict scalar distributions to a shared interval.
   * - :func:`grad`
     - Differentiate a scalar-valued function.
   * - :func:`vjp`
     - Construct a vector-Jacobian product.
   * - :func:`composite`
     - Build a Composite operation.
   * - :func:`scan`
     - Build a recurrence.

.. toctree::
   :hidden:

   api/interface.broadcast
   api/interface.vmap
   api/interface.loop
   api/interface.seq
   api/interface.vfor
   api/interface.sfor
   api/interface.log_prob
   api/interface.cdf
   api/interface.quantile
   api/interface.truncated
   api/interface.grad
   api/interface.vjp
   api/interface.composite
   api/interface.scan

Graph inspection
----------------

.. list-table::

   * - :func:`upstream_nodes`
     - Find upstream graph nodes.
   * - :func:`print_upstream`
     - Print upstream RVs as a table.
   * - :func:`graph_upstream`
     - Print upstream RVs as a Unicode DAG.
   * - :func:`ir_code`
     - Inspect a graph-building function as IR code.
   * - :func:`signature`
     - Describe the inputs and output of a function call.

.. toctree::
   :hidden:

   api/interface.upstream_nodes
   api/interface.print_upstream
   api/interface.graph_upstream
   api/interface.ir_code
   api/interface.signature


Configuration
-------------

The default interface uses simple broadcasting. Use ``with pi.overload(broadcasting="rank")``
for a temporary change. To create a separate configured interface, see :doc:`interface.interface`.

.. list-table::

   * - :doc:`Interface <interface.interface>`
     - Create a separately configured interface.
   * - :func:`overload`
     - Temporarily change the default interface's configuration.

.. toctree::
   :hidden:

   api/interface.overload

See :ref:`interface-broadcasting` for the rules and examples for each mode.

.. toctree::
   :hidden:

   interface.broadcasting

.. toctree::
   :hidden:

   Interface <interface.interface>
   tforms <interface.tforms>
   interface.examples
