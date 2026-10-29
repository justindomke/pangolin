import pytest

from pangolin.ir import (
    Bernoulli,
    Bijected,
    Bijector,
    Composite,
    Constant,
    Exp,
    Exponential,
    Identity,
    Log,
    MultiNormal,
    Normal,
)


def exp_bijector() -> Bijector:
    return Bijector(Exp(), Log(), Composite(2, (Identity(),), [[0]]))


def test_bijected_public_name_and_representations():
    from pangolin import ir

    op = Bijected(Normal(), exp_bijector())

    assert "Bijected" in ir.__all__
    assert "Transformed" not in ir.__all__
    assert not hasattr(ir, "Transformed")
    assert repr(op) == f"Bijected({op.base_op!r}, {op.bijector!r})"
    assert str(op) == f"bijected({op.base_op}, {op.bijector})"


def alternate_exp_bijector() -> Bijector:
    return Bijector(Exp(), Log(), Composite(2, (Identity(),), [[1]]))


def parameterized_identity_bijector() -> Bijector:
    identity = Composite(2, (Identity(),), [[0]])
    zero_log_det = Composite(3, (Constant(0),), [[]])
    return Bijector(identity, identity, zero_log_det, n_biject_params=1)


def test_transformed_is_immutable():
    op = Bijected(Normal(), exp_bijector())

    with pytest.raises(TypeError, match="Ops are immutable"):
        op.base_op = Exponential()  # type: ignore

    with pytest.raises(TypeError, match="Ops are immutable"):
        op.bijector = alternate_exp_bijector()  # type: ignore


def test_transformed_is_continuous():
    op = Bijected(Normal(), exp_bijector())

    assert op.discrete is False


def test_transformed_rejects_discrete_base_op():
    with pytest.raises(ValueError, match="base_op must be continuous"):
        Bijected(Bernoulli(), exp_bijector())


def test_transformed_rejects_non_op_base_op():
    with pytest.raises(TypeError, match="base_op must be an Op"):
        Bijected("normal", exp_bijector())  # type: ignore[arg-type]


def test_transformed_rejects_deterministic_base_op():
    with pytest.raises(ValueError, match="base_op must be random"):
        Bijected(Exp(), exp_bijector())


def test_transformed_rejects_non_bijector():
    with pytest.raises(TypeError, match="bijector must be a Bijector"):
        Bijected(Normal(), Exp())  # type: ignore[arg-type]


def test_transformed_accepts_bijector_parameters_before_base_parameters():
    op = Bijected(MultiNormal(), parameterized_identity_bijector())

    assert op.get_shape((), (2,), (2, 2)) == (2,)


def test_transformed_equality_and_hash_include_base_op_and_bijector():
    op = Bijected(Normal(), exp_bijector())
    equal_op = Bijected(Normal(), exp_bijector())
    different_base_op = Bijected(Exponential(), exp_bijector())
    different_bijector = Bijected(Normal(), alternate_exp_bijector())

    assert op == equal_op
    assert hash(op) == hash(equal_op)
    assert op != different_base_op
    assert op != different_bijector


def test_transformed_can_be_used_as_a_dictionary_key():
    op = Bijected(Normal(), exp_bijector())
    equivalent_op = Bijected(Normal(), exp_bijector())

    values = {op: "value"}

    assert values[equivalent_op] == "value"
