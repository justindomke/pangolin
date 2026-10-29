from typing import Callable

from pangolin import ir


class InferenceTestBase:
    """Shared callback contract and defaults for inference test groups."""

    sample_flat: Callable
    cast = staticmethod(lambda x: x)
    _ops_without_sampling_support: frozenset[type[ir.Op]] = frozenset()
    "A set of pangolin random op types that this backend cannot sample from. Corresponding tests will be skipped"
    _ops_without_log_prob_support: frozenset[type[ir.Op]] = frozenset()
    "A set of pangolin random op types that this backend cannot evaluate log probabilities for. Corresponding tests will be skipped"
    _ops_without_eval_support: frozenset[type[ir.Op]] = frozenset()
    "A set of pangolin deterministic op types that this backend cannot evaluate. Corresponding tests will be skipped"
