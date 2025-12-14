"""
Pangolin's goal is to be **the world's friendliest probabilistic programming language**.

Quickstart:

- Use :mod:`pangolin.interface` to define probabilistic models in a cheerful interface.
- Use :mod:`pangolin.jax_backend` to compile probabilistic models into plain JAX functions (to be used independently of the rest of Pangolin).
- Use :mod:`pangolin.torch_backend` to compile probabilistic models into plain pytorch functions (to be used independently of the rest of Pangolin).
- Use :mod:`pangolin.blackjax` to easily call `Blackjax <https://blackjax-devs.github.io/blackjax/>`_ to do inference on probabilistic models.

In addition, there are three "private" submodules that end-users would not typically interact with:

- :mod:`pangolin.util` - Various internal utility functions.
- :mod:`pangolin.dag` - Utilities for interacting with directed acyclic graphs (DAGs).
- :mod:`pangolin.ir` - The internal representation (IR) for probabilistic models in Pangolin.
"""

from pangolin import dag, util, ir, interface, jax_backend, blackjax

__all__ = ["dag", "util", "ir", "interface", "jax_backend"]

try:
    from pangolin import torch_backend

    __all__.append("torch_backend")
except ImportError:
    pass

__all__.append("blackjax")
