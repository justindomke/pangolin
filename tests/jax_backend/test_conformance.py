from pangolin.testing.backends import BackendTests
from pangolin.testing.backends.transformation_tests import TransformationTests
from pangolin import jax_backend
from jax import numpy as jnp


class TestJax(BackendTests, TransformationTests):
    ancestor_sample_flat = staticmethod(jax_backend.ancestor_sample_flat)
    ancestor_log_prob_flat = staticmethod(jax_backend.ancestor_log_prob_flat)
    cast = staticmethod(jnp.array)
