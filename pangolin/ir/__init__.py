"""
The core IR of pangolin is essentially just `Op`s and `RV`s. Intuitively,
* An `Op` is a "marker" for a certain function or conditional distribution.
* An `RV` is a tuple of one `Op` and some number of parent `RV`s.

There are different classes of `Op`s, for different types of functions. For example, `Constant`
represents constant functions, and most standard scalar functions are in
`pangolin.ir.scalar_ops`.

While not explicitly listed below, all the scalar ops in `pangolin.ir.scalar_ops` are also
available here.
"""

from pangolin.ir.op import Op
from pangolin.ir.rv import RV
from pangolin.ir.constant import Constant
from pangolin.ir.scalar_ops import *
from pangolin.ir.multivariate_dists import *
from pangolin.ir.multivariate_funs import *
from pangolin.ir.index import Index
from pangolin.ir.vmap import VMap
from pangolin.ir.composite import Composite
from pangolin.ir.autoregressive import Autoregressive

__all__ = ['Op', 'RV', 'Constant', 'op', 'rv', 'constant', 'scalar_ops', 'multivariate_dists',
           'multivariate_funs','index','VMap','vmap','Add','Index','Composite','Autoregressive']

# force everything to be documented
# __all__ = ['Op','RV','Constant'] + [s for s in dir() if not s.startswith('_')]
