from typing import Callable, Protocol


class HasInferenceProps(Protocol):
    "A protocol to prevent the type checker from complaining about the mixins"

    @property
    def sample_flat(self) -> Callable: ...

    @property
    def _ops_without_sampling_support(self): ...
