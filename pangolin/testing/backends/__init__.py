from .deterministic_tests import DeterministicTests
#from .composite_tests import CompositeTests
from .distribution_tests import DistributionTests

class BackendTests(DeterministicTests, DistributionTests):
    """
    This class assumes a fixture named 'ancestor_sample_flat' will be available at runtime.
    """

    _ancestor_sample_flat = None
    _ancestor_log_prob_flat = None
    _cast = None
    _ops_without_sampling_support = {}
    "A set of pangolin random ops that this backend cannot sample from. Corresponding tests will be skipped"
    _ops_without_log_prob_support = {}
    "A set of pangolin random ops that this backend cannot evaluate log_probabilities. Corresponding tests will be skipped"
    _ops_without_eval_support = {}
    "A set of pangolin deterministic ops that this backend cannot evaluate log_probabilities. Corresponding tests will be skipped"

    @property
    def ancestor_sample_flat(self):
        fun = type(self)._ancestor_sample_flat

        if not callable(fun):
            raise TypeError(f"'_ancestor_sample_flat' must be callable, not {type(fun)}")
        
        return fun

    @property
    def ancestor_log_prob_flat(self):
        fun = type(self)._ancestor_log_prob_flat

        if not callable(fun):
            raise TypeError(f"'_ancestor_log_prob_flat' must be callable, not {type(fun)}")
        
        return fun

    @property
    def cast(self):
        fun = type(self)._cast

        if not callable(fun):
            raise TypeError(f"'_cast' must be callable, not {type(fun)}")
        
        return fun
