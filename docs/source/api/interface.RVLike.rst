pangolin.interface.RVLike
=========================

.. currentmodule:: pangolin.interface

.. py:data:: RVLike

   An RV or a numeric array-like value. Graph-building functions accept these values and
   turn non-RV arguments into constant RVs.

   ``RVLike`` is a static type alias, not a runtime-checkable class. Use ``ir.RV`` to
   check whether a result is an RV:

   >>> from pangolin import interface as pi, ir
   >>> isinstance(pi.normal(0, 1), ir.RV)
   True
   >>> isinstance(pi.normal([0, 1], 1), ir.RV)
   True
   >>> x = pi.normal(0, 1)
   >>> isinstance(pi.normal(x, 1), ir.RV)
   True
