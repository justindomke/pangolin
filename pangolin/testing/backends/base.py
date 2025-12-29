from typing import Callable, Protocol


class MixinBase:
    "Prevent the type checker from complaining about the mixins"

    _ancestor_sample_flat: Callable
    _ancestor_log_prob_flat: Callable
    _cast: Callable
    _ops_without_sampling_support: dict = {}
    _ops_without_log_prob_support: dict = {}

    @property
    def ancestor_sample_flat(self) -> Callable:
        return type(self)._ancestor_sample_flat

    @property
    def ancestor_log_prob_flat(self) -> Callable:
        return type(self)._ancestor_log_prob_flat

    @property
    def cast(self):
        return type(self)._cast
