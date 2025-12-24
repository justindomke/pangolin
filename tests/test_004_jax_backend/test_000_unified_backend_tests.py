from pangolin.testing.backends import BackendTests
import pangolin.jax_backend
from jax import numpy as jnp

class TestJax(BackendTests):
    _ancestor_sample_flat = pangolin.jax_backend.ancestor_sample_flat
    _ancestor_log_prob_flat = pangolin.jax_backend.ancestor_log_prob_flat
    _cast = jnp.array