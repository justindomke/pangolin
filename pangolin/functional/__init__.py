"""Transform Pangolin IR graphs with reusable functional operations.

`graph_scan` rebuilds IR graphs directly. `bind_interface` adapts interface-using functions to RVs
carrying any interface. The `composite`, `vmap`, and `scan` combinators accept ordinary
IR-producing functions and return new IR-producing functions. A `Transform` packages forward and
inverse RV functions; its callable interface provides combinators for distribution-producing
functions.

``loop``, `vfor`, and `sfor` build graphs over named integer ranges, with an explicit interface for
their internally created coordinates.
"""

from . import tforms
from .broadcasting import broadcast_off, broadcast_simple, broadcast_rank, broadcast_numpy
from .compositing import composite
from .indexing import index_orthogonal
from .inspecting import ir_code, signature
from .looping import loop
from .log_probabilities import log_prob
from .cumulative_probabilities import cdf
from .quantiles import quantile
from .differentiating import vjp, grad
from .scanning import scan
from .sfor_looping import sfor
from .tracing import bind_interface, graph_scan
from .transforming import Transform, compose_transforms
from .truncating import truncated
from .unconstraining import transform_for, unconstrain
from .vmapping import vmap
from .vfor_looping import vfor

__all__ = [
    "broadcast_off",
    "broadcast_simple",
    "broadcast_rank",
    "broadcast_numpy",
    "composite",
    "index_orthogonal",
    "ir_code",
    "signature",
    "log_prob",
    "cdf",
    "quantile",
    "truncated",
    "grad",
    "vjp",
    "vmap",
    "scan",
    "loop",
    "vfor",
    "sfor",
    "bind_interface",
    "graph_scan",
    "Transform",
    "compose_transforms",
    "tforms",
    "transform_for",
    "unconstrain",
]
