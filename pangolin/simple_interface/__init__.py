"""
This module defines an extremely simple interface for creating objects in the IR.
"""

from .base import *
#from . import vmap
#from . import vmap
from .vmapping import vmap

__all__ = [
    'makerv',
    'InfixRV',
    'vmap',
    'vmapping',
    'normal',
    'normal_prec',
    'lognormal',
    'cauchy',
    'bernoulli',
    'bernoulli_logit',
    'binomial',
    'uniform',
    'beta',
    'beta_binomial',
    'exponential',
    'gamma',
    'poisson',
    'student_t',
    'add',
    'sub',
    'mul',
    'div',
    'pow',
    'sqrt',
    'abs',
    'arccos',
    'arccosh',
    'arcsin',
    'arcsinh',
    'arctan',
    'arctanh',
    'cos',
    'cosh',
    'exp',
    'inv_logit',
    'expit',
    'sigmoid',
    'log',
    'log_gamma',
    'logit',
    'sin',
    'sinh',
    'step',
    'tan',
    'tanh',
    'multi_normal',
    'categorical',
    'multinomial',
    'dirichlet',
    'matmul',
    'inv',
    'softmax',
    'sum',
]


#from . import base
#from .base import *
#from .interface import normal
# from .vmap import vmap
# from .composite import composite
# from .autoregressive import autoregressive
# from .loops import Loop, slot


#__all__ = ['base','vmap','composite','autoregressive','Loop','slot','OperatorRV']
#__all__ += for_api # pyright: ignore [reportUnsupportedDunderAll]


