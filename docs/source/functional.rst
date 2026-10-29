functional
==========

.. testsetup::

   from pangolin.functional import *

.. automodule:: pangolin.functional

The functional layer provides reusable operations and combinators for manipulating pure IR graphs
with ordinary Python functions. It does not provide interface coercion, infix
operators, or indexing syntax.

IR graph scans
--------------

.. autofunction:: pangolin.functional.graph_scan

Function inspection
-------------------

.. autofunction:: pangolin.functional.ir_code

.. autofunction:: pangolin.functional.signature

Interface binding
-----------------

.. autofunction:: pangolin.functional.bind_interface

Indexing
--------

.. autofunction:: pangolin.functional.index_orthogonal

Graph combinators
-----------------

.. autofunction:: pangolin.functional.composite

.. autofunction:: pangolin.functional.log_prob

.. autofunction:: pangolin.functional.cdf

.. autofunction:: pangolin.functional.quantile

.. autofunction:: pangolin.functional.truncated

.. autofunction:: pangolin.functional.grad

.. autofunction:: pangolin.functional.vjp

.. autofunction:: pangolin.functional.vmap

.. autofunction:: pangolin.functional.scan

.. autofunction:: pangolin.functional.vfor

.. autofunction:: pangolin.functional.sfor

Broadcasting combinators
------------------------

.. autofunction:: pangolin.functional.broadcast_off

.. autofunction:: pangolin.functional.broadcast_simple

.. autofunction:: pangolin.functional.broadcast_rank

.. autofunction:: pangolin.functional.broadcast_numpy

Bijected distributions
-------------------------

.. autoclass:: pangolin.functional.Transform
   :members:
   :special-members: __call__

.. autofunction:: pangolin.functional.compose_transforms

The standard transform combinators are documented in
:doc:`pangolin.functional.tforms <functional.tforms>`.

Backend-independent unconstraining
----------------------------------

.. autofunction:: pangolin.functional.unconstrain

.. autofunction:: pangolin.functional.transform_for

.. toctree::
   :hidden:

   functional.tforms
