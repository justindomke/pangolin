from typing import Any

import pytest

from pangolin import ir


def identity(sign: ir.Op | None = None, n: int = 0) -> ir.Bijector:
    """Build a parameterized identity bijector for testing sign."""
    value = ir.Composite(n + 1, (ir.Identity(),), ((0,),))
    ldj = ir.Composite(n + 2, (ir.Constant(0.0),), ((),))
    return ir.Bijector(value, value, ldj, n, sign=sign)


def test_sign_is_optional():
    assert identity().sign is None


@pytest.mark.parametrize("value", [-1, 1])
def test_sign_accepts_a_deterministic_scalar_op(value: int):
    sign = ir.Constant(value)
    bijector = identity(sign)
    assert bijector.sign is sign
    assert bijector.get_shape(()) == ()


@pytest.mark.parametrize("value", [-1, 1, 0, 2, -2, True, False, 1.0, -1.0, "sign", lambda: 1])
def test_sign_rejects_invalid_constants(value: Any):
    with pytest.raises(TypeError, match="sign must be an Op or None"):
        identity(value)


def test_sign_rejects_random_ops():
    with pytest.raises(ValueError, match="sign Op must be deterministic"):
        identity(ir.Normal())


def test_sign_rejects_nonscalar_output():
    with pytest.raises(ValueError, match="sign must return a scalar"):
        identity(ir.Constant([1])).get_shape(())


def test_sign_receives_parameters_not_the_value():
    assert identity(ir.Identity(), 1).get_shape((3,), ()) == (3,)
    with pytest.raises(TypeError):
        identity(ir.Identity()).get_shape(())


def test_sign_must_accept_unused_parameters():
    with pytest.raises(TypeError):
        identity(ir.Constant(1), 1).get_shape((), ())
    sign = ir.Composite(1, (ir.Constant(1),), ((),))
    assert identity(sign, 1).get_shape((), ()) == ()


def test_sign_does_not_prohibit_array_application():
    assert identity(ir.Constant(1)).get_shape((2, 3)) == (2, 3)


def test_sign_values_are_an_author_guarantee_not_structurally_evaluated():
    assert identity(ir.Constant(7)).get_shape(()) == ()


def test_sign_participates_in_structural_identity_and_printing():
    first, same = identity(ir.Constant(1)), identity(ir.Constant(1))
    assert first == same
    assert hash(first) == hash(same)
    assert len({first, same, identity(ir.Constant(-1)), identity()}) == 3
    assert repr(first).endswith("sign=Constant(1))")
    assert str(first).endswith("sign=1)")


def test_sign_is_immutable():
    bijector = identity(ir.Constant(1))
    with pytest.raises(TypeError, match="immutable"):
        setattr(bijector, "sign", ir.Constant(-1))
    with pytest.raises(TypeError, match="immutable"):
        delattr(bijector, "sign")


def test_sign_can_depend_on_multiple_parameters():
    sign = ir.Composite(2, (ir.Mul(),), ((0, 1),))
    bijector = identity(sign, 2)
    assert bijector.sign is sign
    assert bijector.get_shape((), (), ()) == ()
    with pytest.raises(ValueError, match="Number of bijection parameters"):
        bijector.get_shape((), ())


def test_sign_shape_validation_through_bijected():
    sign = ir.Composite(1, (ir.Identity(),), ((0,),))
    bijector = identity(sign, 1)
    op = ir.Bijected(ir.Normal(), bijector)
    assert op.get_shape((), (), ()) == ()
    with pytest.raises(ValueError, match="sign must return a scalar"):
        op.get_shape((2,), (), ())


def test_sign_structure_is_preserved_by_bijected():
    first = ir.Bijected(ir.Normal(), identity(ir.Constant(1)))
    same = ir.Bijected(ir.Normal(), identity(ir.Constant(1)))
    different = ir.Bijected(ir.Normal(), identity(ir.Constant(-1)))
    assert first == same
    assert hash(first) == hash(same)
    assert first != different
    assert "sign=Constant(1)" in repr(first)
