"""
The core IR of pangolin is essentially just `Op`s and `RV`s. Intuitively,
* An `Op` is a "marker" for a certain function or conditional distribution.
* An `RV` is a tuple of one `Op` and some number of parent `RV`s.

There are different classes of `Op`s, for different types of functions. For example, `Constant`
represents constant functions, and most standard scalar functions are in
`cleanpangolin.ir.scalar_ops`.

While not explicitly listed below, all the scalar ops in `cleanpangolin.ir.scalar_ops` are also
available here.
"""

from cleanpangolin.ir.op import Op
from cleanpangolin.ir.rv import RV
from cleanpangolin.ir.constant import Constant
from cleanpangolin.ir.scalar_ops import *
from cleanpangolin.ir.multivariate_dists import *
from cleanpangolin.ir.multivariate_funs import *
from cleanpangolin.ir.index import Index
from cleanpangolin.ir.vmap import VMap


__all__ = ['Op', 'RV', 'Constant', 'op', 'rv', 'constant', 'scalar_ops', 'multivariate_dists',
           'multivariate_funs','index','VMap','vmap']

# force everything to be documented
# __all__ = ['Op','RV','Constant'] + [s for s in dir() if not s.startswith('_')]
