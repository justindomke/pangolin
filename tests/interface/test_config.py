from dataclasses import FrozenInstanceError

import pytest

from pangolin.interface import Config


def test_default_broadcasting_is_simple():
    assert Config().broadcasting == "simple"


@pytest.mark.parametrize("broadcasting", ["off", "simple", "rank", "numpy"])
def test_accepts_each_broadcasting_mode(broadcasting):
    assert Config(broadcasting).broadcasting == broadcasting


def test_rejects_unknown_broadcasting_mode():
    with pytest.raises(
        ValueError,
        match=r"broadcasting must be one of \['off', 'simple', 'rank', 'numpy'\]; got 'wide'",
    ):
        Config("wide")  # type: ignore[arg-type]


def test_rejects_non_string_broadcasting_mode():
    with pytest.raises(TypeError, match="broadcasting must be a string"):
        Config(None)  # type: ignore[arg-type]


def test_config_is_immutable():
    config = Config("rank")

    with pytest.raises(FrozenInstanceError):
        config.broadcasting = "numpy"  # type: ignore[misc]


def test_configs_are_values():
    assert Config("rank") == Config("rank")
    assert Config("rank") != Config("numpy")
    assert hash(Config("rank")) == hash(Config("rank"))
