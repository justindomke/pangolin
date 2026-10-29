pangolin.interface.composite
============================

.. currentmodule:: pangolin.interface

.. py:function:: composite(fun, *, simplify=True)

   Transform ``fun`` into a function that returns a single RV.

   Each call invokes ``fun`` normally, then compresses the newly created graph contributing to its
   output. Every contributing `RV` created during the call becomes an operation inside the
   `Composite`. RVs that existed before the call become its parents. Any RVs that are created but
   do not contribute to the output are ignored.

   Every contributing RV created during the call other than the output must have a deterministic
   Op.

   `pangolin.interface.composite` itself does not coerce non-RV inputs into constant RVs.

   RVs supplied within argument PyTrees come first among the resulting RV's parents in flattened
   order. Contributing RVs captured through closures follow in first-encounter order.

   :param fun: A function that creates and returns exactly one new `RV`. Its arguments may have any
               Python or PyTree structure accepted by ``fun`` itself.
   :type fun: Callable[P, RV]
   :param simplify: If true, replace a Composite containing a single Op with that Op when they are
                    equivalent (the default). Otherwise, retain the Composite.
   :type simplify: bool

   :returns: A function with the same call signature as ``fun`` that returns an `RV` containing a
             `Composite` Op unless simplified.
   :rtype: Callable[P, RV]

   ``P`` represents the argument signature of ``fun``, which the returned function preserves.

   :raises TypeError: When ``fun`` does not return exactly one `RV`.
   :raises ValueError: When ``fun`` returns an RV that existed before the call, or a contributing intermediate RV
       has a random Op.

   .. rubric:: Examples

   Arguments may be PyTrees. Existing RVs within them become the Composite's parents:

   >>> from pangolin import interface as pi
   >>> def fun(stuff):
   ...     x = stuff["cat"]
   ...     y = stuff["dog"]
   ...     a = x + y
   ...     return pi.normal(a, y)
   >>> x = pi.constant(1)
   >>> y = pi.constant(2)
   >>> z = pi.composite(fun)({"cat": x, "dog": y})
   >>> z.op
   Composite(2, [Add(), Normal()], [[0, 1], [2, 1]])
   >>> z.parents == (x, y)
   True

   RVs captured through a closure follow contributing argument RVs at the boundary:

   >>> def fun(stuff):
   ...     y = stuff["dog"]
   ...     a = x + y
   ...     return pi.normal(a, y)
   >>> z = pi.composite(fun)({"dog": y})
   >>> z.op
   Composite(2, [Add(), Normal()], [[1, 0], [2, 0]])
   >>> z.parents == (y, x)
   True
