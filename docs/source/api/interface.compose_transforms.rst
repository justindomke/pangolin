pangolin.interface.compose_transforms
=====================================

.. currentmodule:: pangolin.interface

.. py:function:: compose_transforms(transforms, log_det_direction="forward")

   Compose zero or more Transforms in left-to-right application order.

   ``compose_transforms([a, b])`` applies ``a`` first and then ``b``. The resulting Transform
   captures the union of their external RVs. ``log_det_direction`` selects whether its
   log-Jacobian graph reconstructs the intermediate values in forward or inverse order; both
   choices represent the same mathematical quantity.

   An empty sequence produces the identity Transform.

   :param transforms: Transforms to apply in sequence. ``transforms[0]`` is applied first.
   :type transforms: Sequence[Transform]
   :param log_det_direction: Whether the combined log-Jacobian reconstructs intermediate values by following the
                             transforms forward from ``x`` or backward from ``y``.
   :type log_det_direction: Literal["forward", "inverse"]

   :returns: A single `Transform` equivalent to applying every input Transform in order.
   :rtype: Transform

   :raises TypeError: If any item in ``transforms`` is not a `Transform`.
   :raises ValueError: If ``log_det_direction`` is neither ``"forward"`` nor ``"inverse"``.

   .. rubric:: Examples

   >>> from pangolin import interface as pi
   >>> transform = compose_transforms([pi.tforms.scale_by(2), pi.tforms.shift_by(3)])
   >>> y = transform(pi.normal)(0, 1)
   >>> y.shape
   ()
