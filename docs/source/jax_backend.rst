jax_backend
===========

.. testsetup::

   from pangolin.ir import RV
   from pangolin.jax_backend import *

.. automodule:: pangolin.jax_backend

Graph operations
----------------

.. autofunction:: pangolin.jax_backend.ancestor_sample
.. autofunction:: pangolin.jax_backend.ancestor_sampler
.. autofunction:: pangolin.jax_backend.ancestor_log_prob
.. autofunction:: pangolin.jax_backend.fill_in

Op lowering
-----------

.. autofunction:: pangolin.jax_backend.lower
.. autoclass:: pangolin.jax_backend.LoweredDistribution
   :members: sample, log_prob

.. autofunction:: pangolin.jax_backend.lower_transformed

.. autofunction:: pangolin.jax_backend.lower_quantile

.. autofunction:: pangolin.jax_backend.lower_truncated
