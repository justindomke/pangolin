"""
Main interface to pangolin. Broadly speaking, there are the following classes of functions:
- `makerv` will either cast its input to a constant RV or (if it is already an RV) leave it alone
    (TODO: This is bad)
- Functions to create new RVs with given distributions: `normal`, `exponential`,
`multi_normal`, `dirichlet`, etc.
- Functions to apply deterministic transformations to other RVs: `exp`, `sin`, `pow`, `matmul`, etc.
- Program transforms: `vmap`
"""

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
