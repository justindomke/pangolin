from pangolin.testing.inference import InferenceTests
import pangolin.blackjax
from jax import numpy as jnp
from pangolin import ir


class TestBlackjax(InferenceTests):
    _sample_flat = pangolin.blackjax.sample_flat
    _cast = jnp.array
    _ops_without_sampling_support = {
        # blackjax does not support sampling discrete latents
        ir.Bernoulli,
        ir.BernoulliLogit,
        ir.Binomial,
        ir.Multinomial,
        ir.Categorical,
        ir.Poisson,
        ir.BetaBinomial,
        # blackjax also struggles with dists with hard boundaries
        ir.Uniform,
        ir.Exponential,
        ir.Dirichlet,
        ir.Wishart,
    }
