"""Truncated observations and deferred predictions through the inference graph pipeline."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr
from jax.scipy.stats import norm
from scipy import stats

from pangolin import blackjax, interface as pi


def test_truncated_observation_target_and_deferred_predictions(monkeypatch):
    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 71)
    location = pi.normal(0.0, 1.0)
    observation = pi.truncated(pi.normal, -0.5, 1.5)(location, 0.8)
    prediction = pi.truncated(pi.normal, location - 0.2, location + 0.7)(location, 0.6)
    coordinates = jnp.array([-0.4, 0.1, 0.8])
    calls = []

    def reference(loc):
        mass = ndtr((1.5 - loc) / 0.8) - ndtr((-0.5 - loc) / 0.8)
        return norm.logpdf(loc) + norm.logpdf(0.3, loc, 0.8) - jnp.log(mass)

    def driver(log_prob, key, initial_state, num_samples):
        # Inspect the actual MCMC target; do not substitute a probabilistic MCMC assertion.
        assert len(initial_state) == 1
        target = lambda loc: log_prob([loc])
        for transform in [lambda f: f, jax.grad, jax.hessian]:
            np.testing.assert_allclose(
                jax.jit(jax.vmap(transform(target)))(coordinates),
                jax.vmap(transform(reference))(coordinates),
                atol=3e-6,
                rtol=2e-5,
            )
        calls.append(True)
        return [jnp.tile(coordinates, num_samples // 3)]

    locations, observations, predictions = blackjax.sample_flat(
        [location, observation, prediction], [observation], [0.3], run_inf=driver, num_samples=3072
    )
    assert calls == [True]
    np.testing.assert_array_equal(locations, jnp.tile(coordinates, 1024))
    np.testing.assert_array_equal(observations, jnp.full(3072, 0.3))
    residuals = np.asarray(predictions - locations)
    assert np.all((residuals >= -0.2) & (residuals <= 0.7))
    probabilities = stats.truncnorm.cdf(residuals, -0.2 / 0.6, 0.7 / 0.6, scale=0.6)
    assert stats.kstest(probabilities, "uniform").statistic < 0.05
