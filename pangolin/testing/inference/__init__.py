from .deterministic_tests import DeterministicTests
from .distribution_tests import DistributionTests
from .simple_posterior_tests import SimplePosteriorTests
from .composite_tests import CompositeTests
from .autoregressive_tests import AutoregressiveTests
from .vmap_tests import VmapTests
from .complex_tests import ComplexTests
from .base import MixinBase
from typing import Callable


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
    Module is abstract, to use should define a subclass with ``_sample_flat`` and with a class name starting with "Test"
    """

    _cast = None
    _ops_without_sampling_support = {}
    "A set of pangolin random ops that this backend cannot sample from. Corresponding tests will be skipped"
    _ops_without_log_prob_support = {}
    "A set of pangolin random ops that this backend cannot evaluate log_probabilities. Corresponding tests will be skipped"
    _ops_without_eval_support = {}
    "A set of pangolin deterministic ops that this backend cannot evaluate log_probabilities. Corresponding tests will be skipped"

