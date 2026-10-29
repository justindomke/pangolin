from typing import Callable

from pangolin import ir


class BackendTestBase:
    """Shared callback contract and defaults for backend test groups."""

    ancestor_sample_flat: Callable
    ancestor_log_prob_flat: Callable
    cast = staticmethod(lambda x: x)
    _ops_without_sampling_support: frozenset[type[ir.Op]] = frozenset()
    _ops_without_log_prob_support: frozenset[type[ir.Op]] = frozenset()
    _ops_without_eval_support: frozenset[type[ir.Op]] = frozenset()
