from .deterministic_tests import DeterministicTests
from .distribution_tests import DistributionTests
from .simple_posterior_tests import SimplePosteriorTests
from .composite_tests import CompositeTests
from .autoregressive_tests import AutoregressiveTests
from .vmap_tests import VmapTests
from .complex_tests import ComplexTests


class InferenceTests(
    DeterministicTests,
    DistributionTests,
    SimplePosteriorTests,
    CompositeTests,
    AutoregressiveTests,
    VmapTests,
    ComplexTests,
):
    """
    Module is abstract, to use should define a subclass ``_sample_flat`` etc. with a name starting with "Test"
    """

    _sample_flat = None
    _cast = None
    _ops_without_sampling_support = {}
    "A set of pangolin random ops that this backend cannot sample from. Corresponding tests will be skipped"
    _ops_without_log_prob_support = {}
    "A set of pangolin random ops that this backend cannot evaluate log_probabilities. Corresponding tests will be skipped"
    _ops_without_eval_support = {}
    "A set of pangolin deterministic ops that this backend cannot evaluate log_probabilities. Corresponding tests will be skipped"

    @property
    def sample_flat(self):
        fun = type(self)._sample_flat

        if not callable(fun):
            raise TypeError(f"'_sample_flat' must be callable, not {type(fun)}")

        return fun

    @property
    def cast(self):
        fun = type(self)._cast

        if not callable(fun):
            raise TypeError(f"'_cast' must be callable, not {type(fun)}")

        return fun
