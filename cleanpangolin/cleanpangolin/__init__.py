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

#def set_rv():
#    from cleanpangolin.interface.rv import OperatorRV
#    #import cleanpangolin.interface.rv

#import cleanpangolin.interface
#import cleanpangolin.ir
#from cleanpangolin import interface
from cleanpangolin import interface, ir

#from cleanpangolin.interface import normal, normal_prec

#normal = interface.interface.normal
#normal.__doc__ = interface.interface.normal.__doc__

from cleanpangolin.interface.interface import *
from cleanpangolin.interface.vmap import vmap
from cleanpangolin.interface.index import index

#normal.__doc__ = interface.normal.__doc__

__all__ = ['ir','interface','vmap','index']
__all__ += interface.interface.for_api

#cleanpangolin.ir.op.current_rv[-1] = cleanpangolin.interface.rv.OperatorRV

#set_rv()
