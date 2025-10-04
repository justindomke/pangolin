from pangolin import backend
from pangolin import ir
import numpyro.distributions
import numpy as np
import jax


def test_normal_handler():
    handler = backend.handlers[ir.Normal]
    assert isinstance(handler, backend.NumpyroHandler)

    parent_values = [0.5, 1.1]
    value = -0.3
    log_prob = handler.log_prob(value, parent_values)
    expected = numpyro.distributions.Normal(*parent_values).log_prob(value)
    assert np.allclose(log_prob, expected)

    key = jax.random.PRNGKey(5)
    out = handler.sample(key, parent_values)
    expected = numpyro.distributions.Normal(*parent_values).sample(key)
    assert np.allclose(out, expected)
