pangolin.interface.vmap
=======================

.. currentmodule:: pangolin.interface

.. py:function:: vmap(fun, in_axes=0, axis_size=None, *, simplify=True)

   Create a function that maps ``fun`` over argument axes.

   This function follows the core interface of `jax.vmap
   <https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html>`_, although it does not provide all
   of the optional arguments supported by ``jax.vmap``.

   The function returned by `pangolin.interface.vmap` requires every mapped input leaf (one
   with an integer ``in_axes``) to be an RV. Unmapped leaves (``in_axes=None``) may be RVs or
   other Python values. `vmap` does not itself coerce non-RV inputs into constant RVs; any coercion
   performed by ``fun`` is preserved. To construct the mapped graph, `vmap` calls ``fun`` once with
   placeholder RVs for mapped inputs. ``fun`` must return a PyTree of RV objects created by that
   call. It cannot return any of its inputs or any RVs captured through a closure.

   :param fun: The function to map. Its arguments may have any type, subject to the mapped-input
               restriction above; its output must be a PyTree of RVs.
   :type fun: Callable[..., PyTree[RV]]
   :param in_axes: An integer, ``None``, or a PyTree of integers and ``None``. The PyTree may match all
                   arguments to ``fun`` or be a prefix of their structure. Any sequence (list or tuple) is
                   accepted when there are multiple arguments. An integer selects the mapped axis for every
                   RV in the corresponding argument subtree; ``None`` leaves every RV in that subtree
                   unmapped. The default, ``0``, maps the first axis of every input RV.
   :type in_axes: PyTree[int | None]
   :param axis_size: Size of the mapped axis. This is inferred from mapped inputs when omitted and is required
                     when all entries in ``in_axes`` are ``None``.
   :type axis_size: int | None
   :param simplify: Whether to simplify indexing and permutation patterns while constructing the mapped graph.
                    Disabling simplification preserves a more direct representation of the transformation.
   :type simplify: bool

   :returns: **vec_fun** -- A vectorized version of ``fun`` with the same argument and output PyTree structures. Input
             axes appear at the positions selected by ``in_axes``; the mapped axis is added at position
             zero of every output.
   :rtype: Callable[..., PyTree[RV]]

   :raises TypeError: If a mapped input leaf or an output leaf is not an RV.
   :raises ValueError: If mapped axes have inconsistent sizes, ``axis_size`` cannot be inferred, or an output RV
       was not created by ``fun`` during the call.

   .. rubric:: Notes

   Static type checking preserves the argument types and output PyTree structure of the mapped
   function. Runtime validation enforces which leaves must be RVs.

   .. rubric:: Examples

   Here is the simplest possible example:

   >>> from pangolin import interface as pi
   >>> def fun(a):
   ...     return pi.exp(a)
   >>> A = pi.constant([0, 1, 2])
   >>> pi.vmap(fun)(A)
   RV(VMap(Exp(), [0], 3), RV(Constant([0,1,2])))

   Multiple inputs are allowed:

   >>> def fun(a, b):
   ...     return a * b
   >>> A = pi.constant([1, 2, 3])
   >>> B = pi.constant([4, 5, 6])
   >>> pi.vmap(fun)(A, B)
   RV(VMap(Mul(), [0, 0], 3), RV(Constant([1,2,3])), RV(Constant([4,5,6])))

   Inputs may be left unmapped:

   >>> pi.vmap(fun, [0, None])(A, pi.constant(7))
   RV(VMap(Mul(), [0, None], 3), RV(Constant([1,2,3])), RV(Constant(7)))

   Multiple outputs are allowed:

   >>> def fun(a):
   ...     return [pi.exp(a), pi.log(a)]
   >>> out1, out2 = pi.vmap(fun)(A)
   >>> out1
   RV(VMap(Exp(), [0], 3), RV(Constant([1,2,3])))
   >>> out2
   RV(VMap(Log(), [0], 3), RV(Constant([1,2,3])))

   PyTree inputs and PyTree ``in_axes`` are allowed:

   >>> def fun(x):
   ...     return x["cat"] * x["dog"]
   >>> x = {"cat": A, "dog": pi.constant(3)}
   >>> in_axes = {"cat": 0, "dog": None}
   >>> pi.vmap(fun, in_axes)(x)
   RV(VMap(Mul(), [0, None], 3), RV(Constant([1,2,3])), RV(Constant(3)))

   PyTree outputs are allowed:

   >>> def fun(a, b):
   ...     return {"add": a + b, "mul": a * b}
   >>> pi.vmap(fun)(A, B)
   {'add': RV(VMap(Add(), [0, 0], 3), RV(Constant([1,2,3])), RV(Constant([4,5,6]))), 'mul': RV(VMap(Mul(), [0, 0], 3), RV(Constant([1,2,3])), RV(Constant([4,5,6])))}

   A leaf of ``in_axes`` may apply to an entire subtree:

   >>> def fun(x):
   ...     a, (b, c) = x
   ...     return (a * b) + c
   >>> x = [A, (pi.constant(7), pi.constant(8))]
   >>> in_axes1 = [0, (None, None)]
   >>> in_axes2 = [0, None]
   >>> out1 = pi.vmap(fun, in_axes1)(x)
   >>> out2 = pi.vmap(fun, in_axes2)(x)
   >>> out1.op == out2.op
   True
   >>> pi.print_upstream(out1)
   shape | statement
   ----- | ---------
   (3,)  | a = [1 2 3]
   ()    | b = 7
   (3,)  | c = vmap(mul, [0, None], 3)(a, b)
   ()    | d = 8
   (3,)  | e = vmap(add, [0, None], 3)(c, d)

   .. seealso:: :py:obj:`pangolin.ir.VMap`

