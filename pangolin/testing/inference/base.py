from typing import Callable, Protocol


class MixinBase:
    "Prevent the type checker from complaining about the mixins"

    _sample_flat: Callable
    _cast: Callable
    _ops_without_sampling_support: dict = {}
    "A set of pangolin random op types that this backend cannot sample from. Corresponding tests will be skipped"
    _ops_without_log_prob_support: dict = {}
    "A set of pangolin random op types that this backend cannot evaluate log probabilities for. Corresponding tests will be skipped"
    _ops_without_eval_support: dict = {}
    "A set of pangolin deterministic op types that this backend cannot evaluate. Corresponding tests will be skipped"

    @property
    def sample_flat(self) -> Callable:
        return type(self)._sample_flat

    @property
    def cast(self):
        return type(self)._cast
