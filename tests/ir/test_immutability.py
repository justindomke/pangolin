import numpy as np
import pytest

from pangolin.ir import Bijector, Composite, Constant, Exp, Identity, Log, RV, Sum


def make_bijector() -> Bijector:
    log_det_jac = Composite(2, (Identity(),), ((0,),))
    return Bijector(Exp(), Log(), log_det_jac)


@pytest.mark.parametrize(
    ("value", "attribute", "replacement", "message"),
    [
        (Sum(0), "axis", 1, "Ops are immutable"),
        (make_bijector(), "forward", Identity(), "Bijectors are immutable"),
        (RV(Constant(1)), "op", Constant(2), "RVs are immutable"),
    ],
)
def test_ir_objects_reject_attribute_assignment(value, attribute, replacement, message):
    with pytest.raises(TypeError, match=message):
        setattr(value, attribute, replacement)


@pytest.mark.parametrize(
    ("value", "attribute", "message"),
    [
        (Sum(0), "axis", "Ops are immutable"),
        (make_bijector(), "forward", "Bijectors are immutable"),
        (RV(Constant(1)), "op", "RVs are immutable"),
    ],
)
def test_ir_objects_reject_attribute_deletion(value, attribute, message):
    with pytest.raises(TypeError, match=message):
        delattr(value, attribute)


def test_constant_array_is_copied_and_read_only():
    source = np.array([1, 2, 3])
    constant = Constant(source)

    source[0] = 10
    assert np.array_equal(constant.value, [1, 2, 3])

    with pytest.raises(ValueError, match="read-only"):
        constant.value[0] = 10
