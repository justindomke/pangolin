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

from cleanpangolin.interface.interface import normal
from cleanpangolin.interface.vmap import vmap
from cleanpangolin.interface.index import index

normal.__doc__ = interface.normal.__doc__

__all__ = ['ir','interface','normal','vmap']

#cleanpangolin.ir.op.current_rv[-1] = cleanpangolin.interface.rv.OperatorRV

#set_rv()
