"""Real NUTS inference with truncated likelihoods, predictions, and a bounded latent."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy import stats

from pangolin import blackjax, interface as pi


@pytest.fixture
def fixed_seed(monkeypatch):
    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: 314)


def assert_posterior_moments(samples, log_density, lo=-np.inf, hi=np.inf):
    """Compare a scalar MCMC posterior with independent SciPy quadrature."""
    samples = np.asarray(samples)
    assert samples.shape == (4096,)
    assert np.all(np.isfinite(samples))

    def density(x):
        return np.exp(log_density(x))

    mass = quad(density, lo, hi, epsabs=1e-10)[0]
    assert mass > 0
    mean = quad(lambda x: x * density(x), lo, hi, epsabs=1e-10)[0] / mass
    variance = quad(lambda x: (x - mean) ** 2 * density(x), lo, hi, epsabs=1e-10)[0] / mass
    # Allow Monte Carlo error from correlated NUTS draws, not numerical quadrature error.
    np.testing.assert_allclose(samples.mean(), mean, atol=0.08 * np.sqrt(variance), rtol=0)
    np.testing.assert_allclose(samples.var(), variance, rtol=0.15, atol=0)


def test_nuts_truncated_observations_and_posterior_prediction(fixed_seed):
    location = pi.normal(0, 1)
    observation = pi.vmap(pi.truncated(pi.normal, 0, 1.5), (None, None), axis_size=4)(location, 0.7)
    prediction = pi.truncated(pi.normal, location - 0.2, location + 0.7)(location, 0.6)
    values = np.array([0.05, 0.2, 0.35, 0.8])
    calc = blackjax.blackjax_calculate(blackjax.run_nuts, num_samples=4096)
    samples = calc.sample(
        {"location": location, "observation": observation, "prediction": prediction},
        {"data": observation},
        {"data": values},
    )

    def log_posterior(loc):
        return (
            stats.norm.logpdf(loc)
            + stats.truncnorm.logpdf(
                values, -loc / 0.7, (1.5 - loc) / 0.7, loc=loc, scale=0.7
            ).sum()
        )

    assert_posterior_moments(samples["location"], log_posterior)
    np.testing.assert_allclose(
        samples["observation"], np.broadcast_to(values, (4096, 4)), atol=1e-7
    )
    residuals = np.asarray(samples["prediction"] - samples["location"])
    assert np.all((residuals >= -0.2) & (residuals <= 0.7))
    probabilities = stats.truncnorm.cdf(residuals, -0.2 / 0.6, 0.7 / 0.6, scale=0.6)
    assert stats.kstest(probabilities, "uniform").statistic < 0.035


def test_nuts_truncated_bijected_likelihood_with_positive_latent(fixed_seed):
    excess_scale = pi.exponential(1)
    scale = 0.3 + excess_scale
    model = pi.truncated(pi.tforms.exp(pi.normal), 0.6, 2.5)
    observation = pi.vmap(model, (None, None), axis_size=3)(0, scale)
    values = np.array([0.8, 1.3, 2.0])
    calc = blackjax.blackjax_calculate(blackjax.run_nuts, num_samples=4096)
    samples = calc.sample(excess_scale, observation, values)

    def log_posterior(excess):
        s = 0.3 + excess
        mass = stats.lognorm.cdf(2.5, s) - stats.lognorm.cdf(0.6, s)
        return stats.expon.logpdf(excess) + stats.lognorm.logpdf(values, s).sum() - 3 * np.log(mass)

    assert np.all(np.asarray(samples) > 0)
    assert_posterior_moments(samples, log_posterior, lo=0)


def test_nuts_truncated_random_walk_likelihood(fixed_seed):
    coefficient = pi.normal(0.4, 0.6)

    def step(previous):
        return pi.truncated(pi.normal, previous - 0.6, previous + 0.5)(
            coefficient * previous + 0.2, 0.5
        )

    observation = pi.scan(step, 6)(pi.constant(0.1))
    values = np.array([0.2, 0.5, 0.4, 0.9, 1.2, 1.35])
    previous = np.concatenate(([0.1], values[:-1]))
    calc = blackjax.blackjax_calculate(blackjax.run_nuts, num_samples=4096)
    samples = calc.sample(coefficient, observation, values)

    def log_posterior(a):
        means = a * previous + 0.2
        return (
            stats.norm.logpdf(a, 0.4, 0.6)
            + stats.truncnorm.logpdf(
                values,
                (previous - 0.6 - means) / 0.5,
                (previous + 0.5 - means) / 0.5,
                loc=means,
                scale=0.5,
            ).sum()
        )

    assert_posterior_moments(samples, log_posterior)


@pytest.mark.parametrize("lo, hi", [(-0.7, 1.2), (-0.7, None), (None, 1.2)])
def test_nuts_truncated_latent_with_default_unconstraining(fixed_seed, lo, hi):
    location, scale = -0.1, 0.9
    latent = pi.truncated(pi.normal, lo, hi)(location, scale)
    observation = pi.normal(latent, 0.5)
    calc = blackjax.blackjax_calculate(blackjax.run_nuts, num_samples=4096)
    samples = np.asarray(calc.sample(latent, observation, 0.8))
    lo = -np.inf if lo is None else lo
    hi = np.inf if hi is None else hi
    assert np.all((samples > lo) & (samples < hi))

    # A truncated Normal prior times a Normal likelihood is another truncated Normal.
    posterior_variance = 1 / (1 / scale**2 + 1 / 0.5**2)
    posterior_mean = posterior_variance * (location / scale**2 + 0.8 / 0.5**2)
    posterior_scale = np.sqrt(posterior_variance)
    reference = stats.truncnorm(
        (lo - posterior_mean) / posterior_scale,
        (hi - posterior_mean) / posterior_scale,
        loc=posterior_mean,
        scale=posterior_scale,
    )
    assert samples.shape == (4096,)
    assert np.all(np.isfinite(samples))
    np.testing.assert_allclose(
        samples.mean(), reference.mean(), atol=0.08 * reference.std(), rtol=0
    )
    np.testing.assert_allclose(samples.var(), reference.var(), rtol=0.15, atol=0)


def test_nuts_one_sided_truncated_positive_latent(fixed_seed):
    latent = pi.truncated(pi.exponential, None, 2.0)(1.3)
    observation = pi.normal(latent, 0.4)
    calc = blackjax.blackjax_calculate(blackjax.run_nuts, num_samples=4096)
    samples = np.asarray(calc.sample(latent, observation, 0.7))
    assert np.all((samples > 0) & (samples < 2.0))
    assert_posterior_moments(
        samples,
        lambda x: stats.expon.logpdf(x, scale=1 / 1.3) + stats.norm.logpdf(0.7, x, 0.4),
        lo=0,
        hi=2.0,
    )
