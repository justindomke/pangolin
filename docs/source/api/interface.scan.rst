pangolin.interface.scan
=======================

.. currentmodule:: pangolin.interface

.. py:function:: scan(fun, length, in_axes=0)

   Transform a step function into a function that constructs a Scan RV.

   The step function has the conceptual signature ``fun(carry, *step_args) -> next_carry``.
   ``carry`` is an RV. Mapped argument leaves must be RVs; unmapped leaves
   (``in_axes=None``) may be RVs or other Python values.
   ``fun`` must return exactly one newly created `RV` that depends on ``carry`` and has the
   same shape and interface as the carry. It cannot return ``carry``, another input, or an RV captured through a closure.

   The function returned by `pangolin.interface.scan` has the signature
   ``scan_fun(init, *args)``. ``init`` must already be an RV; use ``pi.constant(start)`` when
   starting from a numeric value. Every mapped input leaf (one with an integer ``in_axes``) must
   also be an RV. `scan` does not itself coerce non-RV arguments; any coercion performed by ``fun``
   is preserved.

   For example, calling ``scan(fun, length=5, in_axes=(0, None))(start, A, B)`` is semantically
   equivalent to:

   .. code-block:: python

       carry = start
       values = []
       for i in range(5):
           carry = fun(carry, A[i], B)
           values.append(carry)
       result = stack(values)

   Pangolin calls ``fun`` once with placeholder RVs to construct a single RV containing a `Scan`
   Op; it does not execute the Python loop. The resulting RV represents all ``length`` successive
   carry values, not merely the final carry. Its shape is ``(length,)`` followed by the carry
   shape.

   Contributing intermediate RVs created by ``fun`` must have deterministic Ops. The output may
   have either a deterministic or random Op. ``fun`` may use RVs captured through a closure; they
   become unmapped parents of the Scan RV after contributing explicit arguments.

   :param fun: Step function with the conceptual signature ``fun(carry, *step_args) -> next_carry``.
   :type fun: Callable[..., RV]
   :param length: Positive number of scan steps.
   :type length: int
   :param in_axes: An integer, ``None``, or a PyTree of integers and ``None``. The PyTree may match all
                   additional arguments or be a prefix of their structure. Any sequence (list or tuple) is
                   accepted when there are multiple arguments. An integer selects the scanned axis for every
                   RV in the corresponding argument subtree; ``None`` leaves every RV in that subtree
                   unscanned. The carry is not included in ``in_axes``. The default, ``0``, scans the first
                   axis of every additional input RV.
   :type in_axes: PyTree[int | None]

   :returns: **scan_fun** -- A function accepting an initial carry followed by the additional argument PyTrees and
             returning an `RV` containing a `Scan` Op.
   :rtype: Callable[..., RV]

   :raises TypeError: If the initial carry or a mapped input leaf is not an RV, or ``fun`` does not return
       exactly one RV.
   :raises ValueError: If ``length`` is not positive, a mapped axis has the wrong length, the output does not
       depend on the carry, the output existed before ``fun`` was called, the output shape differs
       from the carry shape or interface, or a contributing intermediate RV has a random Op.

   .. rubric:: Notes

   Static type checking preserves the argument types accepted by the scanned function.
   Runtime validation enforces which leaves must be RVs.

   .. rubric:: Examples

   Construct a distribution where ``z[i] ~ normal(exp(z[i-1]), 1)``:

   >>> from pangolin import interface as pi
   >>> x = pi.constant(3.3)
   >>> def fun(carry):
   ...     return pi.normal(pi.exp(carry), 1)
   >>> z = pi.scan(fun, 5)(x)
   >>> z.op
   Scan(Composite(1, [Exp(), Constant(1), Normal()], [[0], [], [1, 2]]), 5, [])
   >>> z.parents == (x,)
   True

   Additional arguments can be scanned along selected axes. This constructs
   ``z[i] ~ exponential(z[i-1] * y[i])``:

   >>> y = pi.constant([1, 2, 3])
   >>> def fun(carry, yi):
   ...     return pi.exponential(carry * yi)
   >>> z = pi.scan(fun, 3)(x, y)
   >>> z.op
   Scan(Composite(2, [Mul(), Exponential()], [[0, 1], [2]]), 3, [0])
   >>> z.parents == (x, y)
   True

   Create a constant RV explicitly when starting from a numeric initial carry:

   >>> z = pi.scan(fun, 3)(pi.constant(3.3), y)
   >>> z.parents
   (RV(Constant(3.3)), RV(Constant([1,2,3])))

   .. seealso:: :py:obj:`pangolin.ir.Scan`

