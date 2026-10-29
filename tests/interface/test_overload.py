# pyright: reportUnnecessaryTypeIgnoreComment=error

"""Check scoped interface configuration without cross-interface or cross-task mutation."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import TYPE_CHECKING, assert_type

import pytest

from pangolin import ir
from pangolin import interface as pi
from pangolin.interface import Config, Interface


@pytest.mark.parametrize("mode", ["off", "simple", "rank", "numpy"])
def test_override_and_restore_base_config(mode):
    interface = Interface(broadcasting="rank")
    base = interface._effective_config
    with interface.overload(broadcasting=mode) as result:
        assert result is None
        assert interface._effective_config == Config(mode)
    assert interface._effective_config is base


def test_nested_overrides_restore_each_level():
    interface = Interface(broadcasting="off")
    base = interface._effective_config
    with interface.overload(broadcasting="simple"):
        outer = interface._effective_config
        with interface.overload(broadcasting="numpy"):
            assert interface._effective_config.broadcasting == "numpy"
        assert interface._effective_config is outer
        with interface.overload():
            assert interface._effective_config is outer
        with interface.overload(broadcasting=None):
            assert interface._effective_config is outer
    assert interface._effective_config is base


def test_exception_restores_outer_and_base_configs():
    interface = Interface(broadcasting="off")
    base = interface._effective_config
    with pytest.raises(RuntimeError, match="outer"):
        with interface.overload(broadcasting="simple"):
            outer = interface._effective_config
            with pytest.raises(RuntimeError, match="inner"):
                with interface.overload(broadcasting="numpy"):
                    raise RuntimeError("inner")
            assert interface._effective_config is outer
            raise RuntimeError("outer")
    assert interface._effective_config is base


@pytest.mark.parametrize(
    "value,error", [("invalid", ValueError), (1, TypeError), (False, TypeError)]
)
def test_invalid_override_does_not_change_active_config(value, error):
    interface = Interface(broadcasting="off")
    with interface.overload(broadcasting="simple"):
        outer = interface._effective_config
        with pytest.raises(error):
            with interface.overload(broadcasting=value):
                pytest.fail("invalid override entered")
        assert interface._effective_config is outer
    assert interface._effective_config == Config("off")


def test_unknown_option_is_rejected():
    interface = Interface()
    base = interface._effective_config
    with pytest.raises(TypeError):
        with interface.overload(unknown="off"):  # type: ignore
            pytest.fail("unknown option accepted")
    assert interface._effective_config is base


def test_effective_config_is_read_only():
    interface = Interface()
    with pytest.raises(AttributeError):
        setattr(interface, "_effective_config", Config("off"))


def test_interfaces_are_independent():
    first, second = Interface(broadcasting="off"), Interface(broadcasting="off")
    base = second._effective_config
    with first.overload(broadcasting="simple"):
        assert second._effective_config is base
        assert first.normal([0, 1], 1).shape == (2,)
        with pytest.raises(ValueError, match="broadcasting=OFF"):
            second.normal([0, 1], 1)
        with second.overload(broadcasting="numpy"):
            assert first._effective_config == Config("simple")
            assert second._effective_config == Config("numpy")
        assert second._effective_config is base


def test_saved_function_and_new_rv_use_current_config():
    interface = Interface(broadcasting="off")
    normal = interface.normal
    with interface.overload(broadcasting="simple"):
        assert interface.normal == normal
        y = normal([0, 1], 1)
        assert y.interface is interface
        assert (y + 1).shape == (2,)
    assert y.interface is interface
    with pytest.raises(ValueError, match="broadcasting=OFF"):
        _ = y + 1
    with pytest.raises(ValueError, match="broadcasting=OFF"):
        normal([0, 1], 1)


def test_default_alias_uses_default_interface():
    normal = pi.normal
    base = pi.default._effective_config
    assert "overload" in pi.__all__
    assert pi.overload == pi.default.overload
    with pi.overload(broadcasting="off"):
        with pytest.raises(ValueError, match="broadcasting=OFF"):
            normal([0, 1], 1)
    assert pi.default._effective_config is base
    assert normal([0, 1], 1).interface is pi.default


def test_override_as_decorator():
    interface = Interface(broadcasting="off")

    @interface.overload(broadcasting="simple")
    def model():
        return interface.normal([0, 1], 1)

    for _ in range(2):
        assert model().shape == (2,)
        assert interface._effective_config == Config("off")


def test_concurrent_tasks_have_independent_overrides():
    interface = Interface(broadcasting="rank")
    base = interface._effective_config

    async def scenario():
        barrier = asyncio.Barrier(2)

        async def worker(mode):
            with interface.overload(broadcasting=mode):
                await barrier.wait()
                assert interface._effective_config == Config(mode)
                await barrier.wait()
            assert interface._effective_config is base

        await asyncio.gather(worker("off"), worker("numpy"))
        assert interface._effective_config is base

    asyncio.run(scenario())


def test_child_task_inherits_creation_context():
    interface = Interface(broadcasting="off")

    async def scenario():
        ready = asyncio.Event()

        async def child():
            await ready.wait()
            assert interface._effective_config == Config("simple")
            with interface.overload(broadcasting="numpy"):
                assert interface._effective_config == Config("numpy")
            assert interface._effective_config == Config("simple")

        with interface.overload(broadcasting="simple"):
            task = asyncio.create_task(child())
        assert interface._effective_config == Config("off")
        ready.set()
        await task
        assert interface._effective_config == Config("off")

    asyncio.run(scenario())


def test_task_cancellation_restores_config():
    interface = Interface(broadcasting="off")

    async def scenario():
        entered = asyncio.Event()
        pending = asyncio.Event()

        async def worker():
            try:
                with interface.overload(broadcasting="simple"):
                    entered.set()
                    await pending.wait()
            except asyncio.CancelledError:
                assert interface._effective_config == Config("off")
                raise

        task = asyncio.create_task(worker())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert interface._effective_config == Config("off")

    asyncio.run(scenario())


def test_threads_have_independent_overrides():
    interface = Interface(broadcasting="rank")
    barrier = Barrier(2, timeout=5)

    def worker(mode):
        with interface.overload(broadcasting=mode):
            barrier.wait()
            assert interface._effective_config == Config(mode)
            barrier.wait()
        assert interface._effective_config == Config("rank")

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(worker, ["off", "numpy"]))
    assert interface._effective_config == Config("rank")


def test_overload_typing():
    interface = Interface()
    with interface.overload(broadcasting="off"):
        assert_type(interface._effective_config, Config)
        assert_type(interface.normal(0, 1), ir.RV)


if TYPE_CHECKING:
    pi.overload("off")  # type: ignore
    pi.overload(broadcasting="invalid")  # type: ignore
    pi.overload(broadcasting=1)  # type: ignore
