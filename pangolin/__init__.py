"""
Pangolin's goal is to be **the world's friendliest probabilistic programming language**.

Quickstart:

- Use `pangolin.interface` to define probabilistic models in a cheerful interface.
- Use `pangolin.backend` to compile probabilistic models into plain JAX functions (to be used independently of the rest of Pangolin).
- Use `pangolin.blackjax` to easily call [blackjax](https://blackjax-devs.github.io/blackjax/) to do inference on probabilistic models.

In addition, there are three submodules that end-users would not typically interact with:

- `pangolin.ir` - The internal representation (IR) for probabilistic models in Pangolin.
- `pangolin.util` - Various internal utility functions.
- `pangolin.dag` - Utilities for interacting with directed acyclic graphs (DAGs).
"""

from .ir import print_upstream
from pangolin import blackjax

from pangolin import util, dag, ir

# from pangolin import interface, ir, inference, simple_interface, util

# from pangolin.inference import *
# from pangolin.interface import *

# # from pangolin.interface import autoregressive, print_upstream

# # from pangolin.interface.interface import *
# from pangolin.interface.vmap import vmap
# from pangolin.interface.index import index

# # from pangolin.ir import print_upstream
# from . import base_interface

# # import test_imports
# from . import ir_test

__all__ = [
    "util",
    "dag",
    "ir",
]

# __all__ = []
# __all__ = [
#     "ir",
#     "base_interface",
#     "simple_interface",
#     "util",
#     "interface",
#     "inference",
#     "base",
#     "vmap",
#     "index",
#     "print_upstream",
#     "ir_test",
# ]
# # __all__ += base.for_api # pyright: ignore [reportUnsupportedDunderAll]
# # __all__.sort() # pyright: ignore [reportUnsupportedDunderAll]
