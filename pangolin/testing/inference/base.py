from typing import Callable, Protocol


class MixinBase:
    "Prevent the type checker from complaining about the mixins"

    _sample_flat: Callable
    _cast: Callable
    _ops_without_sampling_support: dict = {}

    @property
    def sample_flat(self) -> Callable:
        return type(self)._sample_flat

    @property
    def cast(self):
        return type(self)._cast
