"""Inspection combinators are shared by default and configured interfaces."""

from pangolin import functional
from pangolin import interface as pi


def test_inspection_exports():
    configured = pi.Interface(broadcasting="numpy")
    for name in ["signature", "ir_code"]:
        assert name in pi.__all__
        assert name in functional.__all__
        assert getattr(pi, name) is getattr(functional, name)
        assert getattr(configured, name) is getattr(functional, name)


def test_configured_signature_uses_supplied_functions_broadcasting():
    configured = pi.Interface(broadcasting="numpy")
    assert configured.signature(configured.normal)([[0.0], [1.0]], [1.0, 2.0, 3.0]) == (
        "([[0.0], [1.0]], [1.0, 2.0, 3.0]) -> RV(2, 3)"
    )
    assert pi.signature(pi.normal)([0.0, 1.0], 1.0) == "([0.0, 1.0], 1.0) -> RV(2,)"
