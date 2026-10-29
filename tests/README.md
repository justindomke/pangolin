# Tests

Most directories mirror the modules whose behavior they test; `compositional/` spans layers:

| Directory | Responsibility |
|---|---|
| `dag/` | Generic traversal and graph algorithms |
| `util/` | PyTree utilities and graph rendering |
| `ir/` | Ops, shapes, RVs, identity, and syntax delegation |
| `functional/` | Tracing, combinators, broadcasting, transforms, and graph rewriting |
| `interface/` | Coercion, Python syntax, configured graph builders, and conveniences |
| `jax_backend/` | Numerical lowering, sampling, densities, and compiled transforms |
| `compositional/` | End-to-end models combining interface, functional, and JAX backend behavior |
| `torch_backend/` | Torch backend behavior, when Torch is installed |
| `blackjax/` | Inference, deferred sampling, and integration with Calculate |

Choose a directory by the behavior asserted, not every module imported. A functional
graph-structure test may use interface functions to construct inputs. A numerical test belongs
with the backend evaluating the graph. Tests deliberately exercising combinations across layers
belong in `compositional/`: one file per model, with a short module docstring and an independent
JAX, NumPy, or NumPyro reference. Put the implementations in separate named functions above the
tests, so alternatives can be read side by side without following test setup or selection branches.
Interface re-exports need no duplicate functional suite.

Use descriptive filenames without numerical prefixes. Related files can distinguish models,
arguments, and signatures: `test_scan_models.py`, `test_scan_arguments.py`, and
`test_scan_signature_typing.py`. Typing tests live beside runtime tests. Small `__init__.py`
files allow matching filenames in different directories without pytest import clashes.

Reusable backend and inference conformance suites remain in `pangolin.testing`, where backend
authors can use them independently of this repository's concrete implementations.

## Running tests

From the repository root:

```sh
# Everything, including source doctests configured in pyproject.toml.
.venv/bin/pytest

# Tests without source doctests.
.venv/bin/pytest -o addopts='' -q tests

# One module.
.venv/bin/pytest -o addopts='' -q tests/functional
```

The root `conftest.py` excludes Torch tests and source doctests when Torch is unavailable and
clears JAX compilation caches between test modules.

## Typing

Runtime execution of `assert_type` is not a static check. Run the current typing suites explicitly:

```sh
.venv/bin/pyright --pythonpath .venv/bin/python tests/dag/*typing.py tests/ir/*typing.py tests/functional/*typing.py tests/interface/*typing.py
.venv/bin/ty check tests/ir/*typing.py tests/functional/*typing.py tests/interface/*typing.py
```

Negative typing tests use `TYPE_CHECKING` and suppression comments to check rejected calls
without executing them. Pyright's `reportUnnecessaryTypeIgnoreComment` setting guards against
negative cases silently becoming accepted. The generic DAG suite runs under Pyright; ty currently
reports three inference errors there involving recursive PyTrees and a fold result. These are
not suppressed.
