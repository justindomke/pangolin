Standard transforms
===================

Provide the standard library of :class:`~pangolin.functional.Transform` objects and factories.

The ready-made Transforms are:

- :data:`~pangolin.functional.tforms.exp` and :data:`~pangolin.functional.tforms.log` for positive values;
- :data:`~pangolin.functional.tforms.logit` and :data:`~pangolin.functional.tforms.inv_logit` for the unit interval;
- :data:`~pangolin.functional.tforms.stick_breaking`, :data:`~pangolin.functional.tforms.softmax_centered`, and :data:`~pangolin.functional.tforms.softmax_centered_ilr` for simplex coordinates;
- :data:`~pangolin.functional.tforms.cholesky` for positive-definite matrices;
- :data:`~pangolin.functional.tforms.fill_tril` and :data:`~pangolin.functional.tforms.extract_tril` between vectors and lower-triangular matrices;
- :data:`~pangolin.functional.tforms.exp_diagonal` and :data:`~pangolin.functional.tforms.log_diagonal` for lower-triangular matrices; and
- :data:`~pangolin.functional.tforms.unconstrain_spd`, which composes Cholesky decomposition, diagonal logarithms, and lower-triangle
  extraction.

The parameterized factories are :meth:`~pangolin.interface.TransformLibrary.scaled_logit`, :meth:`~pangolin.interface.TransformLibrary.scale_by`, and :meth:`~pangolin.interface.TransformLibrary.shift_by`. They return Transforms
whose parameters are captured as RVs.

Examples
--------
>>> from pangolin import interface as pi
>>> positive = pi.tforms.exp(pi.normal)(0, 1)
>>> unit_interval = pi.tforms.inv_logit(pi.normal)(0, 1)
>>> positive.shape == unit_interval.shape == ()
True

.. automethod:: pangolin.interface.TransformLibrary.scale_by

.. automethod:: pangolin.interface.TransformLibrary.shift_by

.. automethod:: pangolin.interface.TransformLibrary.scaled_logit

Fixed transforms are shared with :mod:`pangolin.functional.tforms`. See
:doc:`functional.tforms` for their definitions, directions, and shape requirements.
