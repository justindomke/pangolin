"""
Program transformations
"""

from .transforms import *
from .duplicate_deterministic import duplicate_deterministic
from .normal_normal import normal_normal as normal_normal, normal_normal_ez
from .constant_op import constant_op, constant_op_ez
from .vmapped import vmap_local_transform, vmap_local_transform_ez

from .local_transforms import LocalTransform

__all__ = [
    duplicate_deterministic,
    normal_normal,
    normal_normal_ez,
    constant_op,
    constant_op_ez,
    vmap_local_transform
]

# TODO:
# normal-normal
# beta-bernoulli / beta-binomial
# vmap-normal-normal
# vmap-beta-bernoulli
# auto-vmap
# vectorized-normal-obs
# vectorized-bernoulli-obs
