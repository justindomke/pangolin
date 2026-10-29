from .deterministic_tests import DeterministicTests
from .distribution_tests import DistributionTests
from .composite_tests import CompositeTests
from .transformation_tests import TransformationTests


class BackendTests(DeterministicTests, DistributionTests, CompositeTests):
    """Reusable conformance tests for backend sampling and log-probability evaluation.

    To use, define a subclass whose name starts with "Test" and supply ``ancestor_sample_flat``
    and ``ancestor_log_prob_flat`` as ``staticmethod`` callbacks. Override ``cast`` with another
    ``staticmethod`` if needed.

    Attributes
    ----------
    ancestor_sample_flat
        Callback that evaluates or samples a flat list of RVs using a random key and returns
        their values. The tests also pass ``None`` as the key.
    ancestor_log_prob_flat
        Callback taking a flat list of RVs and their values and returning their joint log
        probability, including the required ancestor factors.
    cast
        Callback for converting values to the backend's array representation. Defaults to the
        identity function.
    _ops_without_sampling_support
        Immutable set of random Op classes the backend cannot sample. Tests that explicitly
        require sampling those Ops are skipped.
    _ops_without_log_prob_support
        Immutable set of random Op classes whose log probabilities the backend cannot evaluate.
        The corresponding primitive log-probability tests are skipped.
    _ops_without_eval_support
        Immutable set of deterministic Op classes the backend cannot evaluate. Reserved for tests
        requiring that capability; currently no backend test consults it.

    Notes
    -----
    All three unsupported-Op sets default to empty. Override them on a backend's test subclass
    using classes, not Op instances, for example
    ``_ops_without_sampling_support = frozenset({ir.Bernoulli})``.
    These settings are consulted explicitly by individual tests; they do not automatically skip
    every model containing an unsupported Op. Defaults are defined once in ``BackendTestBase``
    and are also inherited by individual test groups.
    ``TransformationTests`` is a separate group that a backend can additionally inherit.
    """
