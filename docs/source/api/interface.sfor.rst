pangolin.interface.sfor
=======================

.. currentmodule:: pangolin.interface

.. py:function:: sfor(fun=None, /, *, init, **sizes)

   Build a state sequence with named sequential and parallel coordinates.

   A scalar call of the form::

       x = sfor(fun, init=init, t=T)

   has these semantics::

       previous = init
       states = []
       for t in range(T):
           next_state = fun(previous, t)
           states.append(next_state)
           previous = next_state
       x = stack(states)

   For example, if::

       fun = lambda previous, t: normal(previous, 1)

   then::

       x[0] ~ normal(init, 1)
       x[t] ~ normal(x[t-1], 1), t > 0

   A call with parallel coordinates::

       x = sfor(fun, init=init, t=T, i=I, j=J)

   has these semantics::

       previous = init
       states = []
       for t in range(T):
           next_state = empty((I, J))
           for i in range(I):
               for j in range(J):
                   next_state[i, j] = fun(previous, t, i, j)
           states.append(next_state)
           previous = next_state
       x = stack(states)

   The loops are represented by one `Scan` containing nested `VMap` operations; they are not
   executed in Python. Function-parameter order determines the nesting, while keyword order has
   no effect.

   For example, if::

       fun = lambda previous, t, i, j: normal(a[i, j] * previous[i, j] + b[t, i, j], c[i, j])

   then::

       x[0, i, j] ~ normal(a[i, j] * init[i, j] + b[0, i, j], c[i, j])
       x[t, i, j] ~ normal(a[i, j] * x[t-1, i, j] + b[t, i, j], c[i, j]), t > 0

   Finally, decorator form::

       @sfor(init=init, t=T)
       def x(previous, t):
           return normal(previous, 1)

   is equivalent to::

       def fun(previous, t):
           return normal(previous, 1)

       x = sfor(fun, init=init, t=T)

   In both forms, ``init`` must be supplied as a keyword argument.

   Each call to ``fun`` must return exactly one `RV` created during that call. The result
   must depend on ``previous``. Any other RVs created during the call that contribute to the result must have
   deterministic Ops.

   The shape of ``init`` must match one complete state assembled from ``fun``. For
   ``fun(previous, t, i, j)``, let ``S`` be the shape returned by one call to ``fun``. With
   ``t=T``, ``i=I``, and ``j=J``, ``init`` must have shape ``(I, J, *S)``, and the result has shape
   ``(T, I, J, *S)``. Equivalently, ``result.shape == (T,) + init.shape``. Note that the output
   axes appear in function-parameter order; keyword order is ignored.

   Each parallel call receives the complete previous state; ``sfor`` does not slice it along the
   parallel axes.

   :param fun: Step function, or ``None`` in decorator form. It must have a fixed positional signature
               with at least two parameters and no defaults, variadic parameters, or keyword-only
               parameters. The first parameter receives the complete previous state; the second receives
               the scalar sequential coordinate; any later parameters receive scalar parallel
               coordinates. No parameter may be named ``init`` or ``interface``. The function must satisfy the return,
               carry-dependence, determinism, and shape requirements above.
   :type fun: Callable[..., RV] | None
   :param init: Keyword-only complete state before the first step. It may be any `RVLike` and is coerced
                to an RV. Its shape must match the state assembled by mapping ``fun`` over every
                parallel coordinate.
   :type init: RVLike
   :param \*\*sizes: Exactly one positive integer for every coordinate parameter: the sequential parameter and
                     all parallel parameters. Names must match the function parameters. Values determine axis
                     lengths; function-parameter order, not keyword order, determines axis order.
   :type \*\*sizes: int

   :returns: **result_or_decorator** -- When ``fun`` is supplied, an RV containing every state produced by the scan, not
             including ``init``. If the sequential size is ``T``, ``result[t]`` is the state produced at
             coordinate ``t`` and ``result.shape == (T,) + init.shape``. When ``fun`` is omitted, a
             decorator that accepts it and returns such an RV.
   :rtype: RV | Callable[[Callable[..., RV]], RV]

   :raises TypeError: If ``fun`` is not callable or lacks the required signature, a size is not an integer, or
       ``fun`` does not return exactly one `RV`.
   :raises ValueError: If a parameter is named ``init`` or ``interface``, the function has no sequential coordinate, sizes are
       missing, unknown, or nonpositive, the result does not depend on ``previous``, an
       intermediate Op is random, or the mapped state does not match the shape or interface of the coerced ``init``.

   .. rubric:: Examples

   >>> from pangolin import interface as pi
   >>> walk = pi.sfor(lambda previous, t: pi.normal(previous, 1), init=0, t=3)
   >>> walk.shape
   (3,)
