"""
The core IR of pangolin is essentially just `Op`s and `RV`s. Intuitively,
* An `Op` is a "marker" for a certain deterministic function or conditional distribution.
* An `RV` is a tuple of one `Op` and some number of parent `RV`s.

**Note:** `Op`s do not include parents. For example `add = Add()` represents an op that adds two variables. `Op`s only take *static* arguments. For example, `Constant` represents a constant function. This has no parents. Instead the value of the constant is passed when creating the `Op`, e.g. `three = Constant(3)`.
"""

from pangolin.ir.rv import RV
from pangolin.ir.constant import Constant
from pangolin.ir.scalar_ops import *
from pangolin.ir.multivariate_dists import *
from pangolin.ir.multivariate_funs import *
from pangolin.ir.index import Index
from pangolin.ir.vmap import VMap
from pangolin.ir.composite import Composite
from pangolin.ir.autoregressive import Autoregressive
from pangolin.ir.op import Op
from .printing import print_upstream
#from . import vmap, multivariate_dists, multivariate_funs, index, scalar_ops, op, rv, constant
import inspect

# __all__ = ['Op',
#            'RV',
#            'Constant',
#            'op',
#            'rv',
#            'constant',
#            'scalar_ops',
#            'multivariate_dists',
#            'multivariate_funs',
#            'index',
#            'VMap',
#            'vmap',
#            'Add',
#            'Index',
#            'Composite',
#            'Autoregressive',
#            'print_upstream'] 

__all__ = [
    'Op',
    'RV',
    'print_upstream',
    'Abs',
    'Add',
    'Autoregressive',
    'Composite',
    'Constant',
    'Index',
    'Arccos',
    'Arccosh',
    'Arcsin',
    'Arcsinh',
    'Arctan',
    'Arctanh',
    'Bernoulli',
    'BernoulliLogit',
    'Beta',
    'BetaBinomial',
    'Binomial',
    'Categorical',
    'Cauchy',
    'Cos',
    'Cosh',
    'Dirichlet',
    'Div',
    'Exp',
    'Exponential',
    'Gamma',
    'Inv',
    'InvLogit',
    'Log',
    'LogNormal',
    'Loggamma',
    'Logit',
    'MatMul',
    'Mul',
    'MultiNormal',
    'Multinomial',
    'Normal',
    'NormalPrec',
    'Poisson',
    'Pow',
    'Sin',
    'Sinh',
    'Softmax',
    'Step',
    'StudentT',
    'Sub',
    'Sum',
    'Tan',
    'Tanh',
    'Uniform',
    'VMap']


def test_all_ops_exported():
    excluded_op_types = [Op, ir.VecMatOp, ir.ScalarOp]

    for name in dir(ir):
        op_type = getattr(ir, name)
        if inspect.isclass(op_type):
            if issubclass(op_type, Op) and op_type not in excluded_op_types:
                name = op_type.__name__
                if name not in __all__:
                    raise Warning(f"Op {name} not exported. (Pangolin bug)")
                    
test_all_ops_exported()

# force everything to be documented
#__all__ = ['Op','RV','Constant'] + [s for s in dir() if not s.startswith('_')]


