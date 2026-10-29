"""Provide the JAX backend for interface integration checks."""

from types import ModuleType

import pytest

from pangolin import jax_backend


@pytest.fixture
def backend() -> ModuleType:
    return jax_backend
