from .deterministic_tests import DeterministicTests
from .distribution_tests import DistributionTests
from .simple_posterior_tests import SimplePosteriorTests
from .composite_tests import CompositeTests
from .scan_tests import ScanTests
from .vmap_tests import VmapTests
from .complex_tests import ComplexTests
from .transformation_tests import TransformationTests


class InferenceTests(
    DeterministicTests,
    DistributionTests,
    SimplePosteriorTests,
    CompositeTests,
    ScanTests,
    VmapTests,
    ComplexTests,
    TransformationTests,
):
    """Reusable conformance tests for inference engines implementing ``sample_flat``.

    To use, define a subclass whose name starts with "Test" and supply ``sample_flat``
    as a ``staticmethod``. Override ``cast`` with another ``staticmethod`` if needed.

    Attributes
    ----------
    sample_flat
        Sampling callback accepting flat lists of requested RVs, observed RVs, and observation
        values, plus ``num_samples`` as a keyword argument. Returns samples for each requested RV
        with a leading sample dimension.
    cast
        Callback for converting values to the engine's array representation. Defaults to the
        identity function.
    _ops_without_sampling_support
        Immutable set of random Op classes that the engine cannot sample. The corresponding
        primitive sampling tests are skipped.
    _ops_without_log_prob_support
        Immutable set of random Op classes whose log probabilities the engine cannot evaluate.
        Reserved for tests requiring that capability; currently no inference test consults it.
    _ops_without_eval_support
        Immutable set of deterministic Op classes that the engine cannot evaluate. The
        corresponding primitive evaluation tests are skipped.

    Notes
    -----
    All three unsupported-Op sets default to empty. Override them on an engine's test subclass
    using classes, not Op instances, for example
    ``_ops_without_sampling_support = frozenset({ir.Bernoulli})``.
    These settings are consulted explicitly by individual tests; they do not automatically skip
    every composite or posterior model containing an unsupported Op. Defaults are defined once
    in ``InferenceTestBase`` and are also inherited by individual test groups.
    """
