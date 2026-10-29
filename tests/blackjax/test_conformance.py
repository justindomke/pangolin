from pangolin.testing.inference import InferenceTests
import pangolin.blackjax
from jax import numpy as jnp


class TestBlackjax(InferenceTests):
    sample_flat = staticmethod(pangolin.blackjax.sample_flat)
    cast = staticmethod(jnp.array)
