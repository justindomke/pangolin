import pytest
from pangolin import inference_jags, inference_numpyro, inference_stan
import numpy as np
from pangolin.interface import makerv, vmap, plate
from pangolin.interface import normal, beta, binomial
import jax

inference_engines = [inference_jags, inference_numpyro, inference_stan]


def assert_means_close(vars, given_vars, given_vals):
    niter = 10_000
    all_means = []
    for inference in inference_engines:
        samples = inference.sample_flat(vars, given_vars, given_vals, niter=niter)
        means = [np.mean(s, axis=0) for s in samples]
        all_means.append(means)
    for means1 in all_means:
        for means2 in all_means:
            for m1, m2 in zip(means1, means2):
                assert np.max(np.abs(m1 - m2)) < 0.01


def test_double_normal():
    z = normal(1.7, 2.9)
    x = normal(z, 0.1)
    x_val = np.array(-1.2)
    assert_means_close([z], [x], [x_val])


def test_beta_binomial():
    z = beta(1.3, 2.4)
    x = binomial(10, z)
    x_val = np.array(5)
    assert_means_close([z], [x], [x_val])
