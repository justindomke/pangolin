"""
Pangolin's goal is to be **the world's friendliest probabilistic programming language**.

Quickstart:

- Use :mod:`pangolin.interface` to define probabilistic models in a cheerful interface.
- Use :mod:`pangolin.functional` to transform IR graphs with functional combinators.
- Use :mod:`pangolin.jax_backend` to compile probabilistic models into plain JAX functions (to be used independently of the rest of Pangolin).
- Use :mod:`pangolin.torch_backend` to compile probabilistic models into plain pytorch functions (to be used independently of the rest of Pangolin).
- Use :mod:`pangolin.blackjax` to easily call `Blackjax <https://blackjax-devs.github.io/blackjax/>`_ to do inference on probabilistic models.

Pangolin also exposes lower-level graph APIs:

- :mod:`pangolin.ir` defines the graph representation for probabilistic models.
- :mod:`pangolin.dag` provides generic directed-acyclic-graph operations.
- :mod:`pangolin.util` provides shared calculation and PyTree helpers.
"""

import warnings

with warnings.catch_warnings():
    # Only ignore DeprecationWarning
    warnings.simplefilter("ignore", category=DeprecationWarning)
    import numpyro


from pangolin import dag, util, ir, functional, interface, jax_backend, calculate, blackjax

__all__ = ["dag", "util", "ir", "functional", "interface", "jax_backend"]

try:
    from pangolin import torch_backend

    __all__.append("torch_backend")
except ImportError:
    pass

__all__.append("calculate")
__all__.append("blackjax")
