import gc
import sys
from importlib.util import find_spec

collect_ignore = []

if find_spec("torch") is None:
    collect_ignore.extend(
        [
            "pangolin/torch_backend.py",
            "tests/torch_backend",
        ]
    )


def pytest_runtest_teardown(item, nextitem):
    """Prevent compiled JAX state from accumulating across test modules."""
    if nextitem is not None and item.path == nextitem.path:
        return

    if "jax" in sys.modules:
        import jax

        jax.clear_caches()
        gc.collect()
