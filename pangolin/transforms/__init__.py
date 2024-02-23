"""
Program transformations.

Key concepts:
* `pangolin.transforms.transforms.Transform` — protocol for transforms
* `pangolin.transforms.transforms.apply_transforms` — convenience function to call a
bunch of transforms

Current transforms:
* `pangolin.transforms.duplicate_deterministic`
* `pangolin.transforms.normal_normal`
* `pangolin.transforms.constant_op`
* `pangolin.transforms.bernoulli_to_binomial`
"""


# TODO:
# normal-normal
# beta-bernoulli / beta-binomial
# vmap-normal-normal
# vmap-beta-bernoulli
# auto-vmap
# vectorized-normal-obs
# vectorized-bernoulli-obs
