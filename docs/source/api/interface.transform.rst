pangolin.interface.Transform
============================

.. currentmodule:: pangolin.interface

.. py:class:: Transform(forward, inverse, log_det_jac, *, sign=None)

   Construct transformed distributions using an invertible transformation.

   A Transform describes an invertible map ``y = T(x)``. If ``X`` has density ``p_X`` and
   ``Y = T(X)``, then

   ``p_Y(y) = p_X(T⁻¹(y)) × |det ∇T⁻¹(y)|``.

   Applying a Transform to a distribution-producing function creates another function
   representing this transformed distribution. For example, ``tforms.exp(normal)(0, 1)``
   represents the distribution of ``exp(X)`` where ``X ~ Normal(0, 1)``. The result is one random
   `RV` containing a `pangolin.ir.Bijected` Op; the intermediate ``X`` and ``exp(X)`` are
   not added to the model as separate RVs.

   A Transform is defined by three interface-level functions:

   - ``forward(x)`` computes ``y = T(x)``.
   - ``inverse(y)`` computes ``x = T⁻¹(y)``.
   - ``log_det_jac(x, y)`` computes the scalar forward log absolute Jacobian determinant
     ``log |det ∇T(x)|``. Both ``x`` and ``y`` are supplied so the implementation may use
     whichever is more convenient.

   Each function must return exactly one `RV`, and every operation it creates must be
   deterministic. The forward and inverse functions must capture the same external RVs. These
   captured RVs become parameters of the compiled `pangolin.ir.Bijector` in first-encounter order;
   ``log_det_jac`` may use any subset of them.

   Calling ``transform(fun)`` returns a function with the same signature as ``fun``. When called,
   that function invokes ``fun`` normally, compiles the three transformation graphs into a
   `pangolin.ir.Bijector`, and returns the corresponding transformed RV.

   :param forward: Unary function mapping a base RV to its transformed value.
   :type forward: Callable[[RV], RV]
   :param inverse: Unary function mapping the transformed value back to the base value.
   :type inverse: Callable[[RV], RV]
   :param log_det_jac: Function computing the scalar forward log absolute Jacobian determinant
                      from the base and transformed values.
   :type log_det_jac: Callable[[RV, RV], RV]
   :param sign: Optional zero-argument function returning a deterministic scalar RV:
                ``+1`` for an increasing scalar bijection or ``-1`` for a decreasing one.
   :type sign: Callable[[], RV] | None

   The optional sign callback may capture any subset of the forward parameters, but must
   not depend on the transformed value or introduce additional captured RVs. By supplying it,
   the author guarantees a continuous, strictly monotone scalar bijection for valid parameters.

   .. rubric:: Examples

   ``pi.tforms.exp`` is a built-in Transform object representing the exponential bijection:

   >>> from pangolin import interface as pi
   >>> pi.tforms.exp  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
   Transform(<function exp_forward at 0x...>, <function exp_inverse at 0x...>,
             <function exp_log_det_jac at 0x...>, sign=<function positive_sign at 0x...>)

   Applying this Transform to ``pi.normal`` creates an ordinary Python function that is
   mathematically equivalent to a lognormal distribution. It is not a special wrapper type.
   Pangolin gives it a derived name indicating that the original function was transformed:

   >>> bespoke_lognormal = pi.tforms.exp(pi.normal)
   >>> bespoke_lognormal  # doctest: +ELLIPSIS
   <function transformed_normal at 0x...>

   Calling that function constructs a transformed RV:

   >>> bespoke_lognormal(0, 1)  # doctest: +NORMALIZE_WHITESPACE
   RV(Bijected(Normal(),
       Bijector(Exp(),
       Log(),
       Composite(2, [Log()], [[1]]),
       sign=Constant(1))),
       RV(Constant(0)), RV(Constant(1)))

   A custom Transform is built directly from its three defining functions:

   >>> manual_exp = Transform(
   ...     lambda x: pi.exp(x),
   ...     lambda y: pi.log(y),
   ...     lambda x, y: pi.log(y),
   ... )

   Its representation displays the three Python functions stored by the Transform:

   >>> manual_exp  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
   Transform(<function <lambda> at 0x...>, <function <lambda> at 0x...>,
             <function <lambda> at 0x...>)

   Applying this manually constructed Transform produces the same distribution as the built-in
   ``pi.tforms.exp``, but without declaring its sign. The result contains the base distribution and the compiled
   forward, inverse, and Jacobian operations:

   >>> manual_exp(pi.normal)(0, 1)  # doctest: +NORMALIZE_WHITESPACE
   RV(Bijected(Normal(),
       Bijector(Exp(),
       Log(),
       Composite(2, [Log()], [[1]]))),
       RV(Constant(0)), RV(Constant(1)))

   If a Transform is parameterized by other RVs, its forward, inverse, and ``log_det_jac``
   functions capture those RVs in closures. The captured RVs become parents of the transformed
   RV:

   >>> factor = pi.constant(2)
   >>> scaled = Transform(
   ...     lambda x: factor * x,
   ...     lambda y: y / factor,
   ...     lambda x, y: pi.log(pi.abs(factor)),
   ... )
   >>> y = scaled(pi.normal)(0, 1)
   >>> y.parents[0] is factor
   True


   .. py:method:: __call__(fun, /)

      Wrap a function that creates one continuous random RV.

      The returned function accepts the same arguments as ``fun`` and passes them to ``fun``
      unchanged. `Transform` does not coerce arguments; ``fun`` determines which inputs it
      accepts and whether to coerce them.

      :param fun: A function that produces one continuous random RV.
      :type fun: Callable[P, RV]
      :returns: A function with the same argument signature as ``fun``.
      :rtype: Callable[P, RV]

      ``P`` represents the argument signature of ``fun``.

      .. rubric:: Examples

      Existing RV arguments remain the parents of the base distribution:

      >>> from pangolin import interface as pi
      >>> bespoke_lognormal = pi.tforms.exp(pi.normal)
      >>> mean = pi.constant(0)
      >>> scale = pi.constant(1)
      >>> y = bespoke_lognormal(mean, scale)
      >>> y.parents == (mean, scale)
      True

      Scalars work too because ``pi.normal`` itself coerces them into constant RVs:

      >>> y = bespoke_lognormal(0, 1)
      >>> [parent.op for parent in y.parents]
      [Constant(0), Constant(1)]


   .. py:method:: apply_and_invert(fun, /)

      Wrap ``fun`` to return the transformed RV and its reconstructed base value.

      Calling the returned function produces ``(y, x)``, where ``y`` is the random transformed
      RV and ``x`` is reconstructed by applying this Transform's inverse operation to ``y``.
      Thus, ``x`` is a deterministic RV whose parent is ``y``, rather than a separate draw from
      the base distribution.

      :param fun: A function that produces one continuous random `RV`.
      :type fun: Callable[P, RV]

      :returns: A function with the same signature as ``fun`` that returns ``(y, x)``.
      :rtype: Callable[P, tuple[RV, RV]]

      ``P`` represents the argument signature of ``fun``.

      .. rubric:: Examples

      Reconstruct a normal value from its exponentially transformed value:

      >>> from pangolin import interface as pi
      >>> transformed_and_base = pi.tforms.exp.apply_and_invert(pi.normal)
      >>> y, x = transformed_and_base(0, 1)
      >>> x.parents == (y,)
      True
      >>> x.op
      Log()


   .. py:property:: reverse

      Return the Transform with its forward and inverse directions exchanged.
