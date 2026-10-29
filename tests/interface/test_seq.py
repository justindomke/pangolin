"""Sequential-axis shorthand works in declarations and explicit loop calls."""

import pytest

from pangolin import functional, interface as pi, ir


def test_seq_is_a_shared_tuple_helper_without_coercion():
    configured = pi.Interface()
    assert configured.seq is pi.seq
    assert "seq" in pi.__all__
    assert pi.seq(3) == (3, None)
    value = [0.0, 1.0]
    specification = configured.seq(3, value)
    assert type(specification) is tuple
    assert specification[0] == 3
    assert specification[1] is value


@pytest.mark.parametrize("initializer", [None, 0.0, [0.0, 1.0]])
def test_seq_in_both_calling_forms_matches_literal_tuple(initializer):
    configured = pi.Interface()

    def model(prev, i, t):
        if t is None:
            return i + 1.0
        return prev[i, t - 1] + 1.0

    @configured.loop
    def declared(prev, i=2, t=configured.seq(3, initializer)):
        return model(prev, i, t)

    explicit = configured.loop(model, i=2, t=configured.seq(3, initializer))
    literal = configured.loop(model, i=2, t=(3, initializer))
    assert ir.rv_equal(declared, explicit)
    assert ir.rv_equal(explicit, literal)
    assert declared.interface is configured
    assert declared.shape == ((2, 3, 2) if isinstance(initializer, list) else (2, 3))


def test_seq_preserves_shared_random_initializer_in_functional_loop():
    start = pi.normal(0, 1)

    def model(prev, i, t):
        return prev[i, t - 1] + 1.0

    actual = functional.loop(model, interface=pi.default, i=2, t=pi.seq(3, start))
    expected = functional.loop(model, interface=pi.default, i=2, t=(3, start))
    assert ir.rv_equal(actual, expected)
    assert pi.seq(3, start)[1] is start


def test_loop_still_validates_seq_sizes():
    with pytest.raises(ValueError, match="must be positive"):
        pi.loop(lambda prev, t: prev[t - 1] + 1.0, t=pi.seq(0, 0.0))
