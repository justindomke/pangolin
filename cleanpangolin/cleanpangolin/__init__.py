"""
Main interface to pangolin. Broadly speaking, there are the following classes of functions:
- `makerv` will either cast its input to a constant RV or (if it is already an RV) leave it alone
    (TODO: This is bad)
- Functions to create new RVs with given distributions: `normal`, `exponential`,
`multi_normal`, `dirichlet`, etc.
- Functions to apply deterministic transformations to other RVs: `exp`, `sin`, `pow`, `matmul`, etc.
- Program transforms: `vmap`
"""

__docformat__ = 'numpy'

from cleanpangolin import interface, ir, inference

from cleanpangolin.interface.interface import *
from cleanpangolin.interface.vmap import vmap
from cleanpangolin.interface.index import index


__all__ = ['ir','interface','vmap','index','inference']
__all__ += interface.interface.for_api
