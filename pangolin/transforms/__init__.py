"""
Program transformations
"""

from .transforms import *
from .duplicate_deterministic import duplicate_deterministic
from .normal_normal import normal_normal as normal_normal
from .constant_op import constant_op
from .vmapped import vmap_local_transform

from .local_transforms import LocalTransform

__all__ = [duplicate_deterministic, normal_normal, constant_op, vmap_local_transform]

# TODO:
# normal-normal
# beta-bernoulli / beta-binomial
# vmap-normal-normal
# vmap-beta-bernoulli
# auto-vmap
# vectorized-normal-obs
# vectorized-bernoulli-obs
