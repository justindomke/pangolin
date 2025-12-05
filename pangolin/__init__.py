"""
Pangolin's goal is to be **the world's friendliest probabilistic programming language**.

Quickstart:

- Use `pangolin.interface` to define probabilistic models in a cheerful interface.
- Use `pangolin.backend` to compile probabilistic models into plain JAX functions (to be used independently of the rest of Pangolin).
- Use `pangolin.torch_backend` to compile probabilistic models into plain pytorch functions (to be used independently of the rest of Pangolin).
- Use `pangolin.blackjax` to easily call [blackjax](https://blackjax-devs.github.io/blackjax/) to do inference on probabilistic models.

In addition, there are three submodules that end-users would not typically interact with:

- `pangolin.ir` - The internal representation (IR) for probabilistic models in Pangolin.
- `pangolin.util` - Various internal utility functions.
- `pangolin.dag` - Utilities for interacting with directed acyclic graphs (DAGs).
"""

from .ir import print_upstream

from pangolin import util, dag, ir, interface, backend, blackjax

__all__ = ["util", "dag", "ir", "interface", "backend", "blackjax"]

try:
    from pangolin import torch_backend

    __all__.append("torch_backend")
except ImportError:
    pass
