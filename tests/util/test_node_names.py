"""Find identity-based names through nested and shared namespace containers."""

from collections.abc import Iterator
from collections import UserDict, namedtuple
from itertools import product
from types import MappingProxyType

import pytest

from pangolin import util


def test_direct_names():
    x, y = object(), object()
    assert util.node_names([y, x], {"x": x, "y": y}) == {x: ["x"], y: ["y"]}


def test_nested_paths():
    x, y = object(), object()
    assert util.node_names([x, y], {"params": {"location": x, "scales": (y,)}}) == {
        x: ["params['location']"],
        y: ["params['scales'][0]"],
    }


def test_all_aliases_in_depth_first_order():
    x, y = object(), object()
    assert util.node_names([x, y], {"states": [x, {"x": x}, y], "x": x}) == {
        x: ["states[0]", "states[1]['x']", "x"],
        y: ["states[2]"],
    }


def test_dictionary_insertion_order_not_sorted():
    x = object()
    assert util.node_names([x], {"z": {"z": x, "a": x}, "a": x})[x] == ["z['z']", "z['a']", "a"]


def test_missing_and_unrequested_nodes():
    x, y, z = object(), object(), object()
    assert util.node_names([y, z], {"values": [x, y], "count": 3}) == {y: ["values[1]"]}


def test_empty_inputs():
    assert util.node_names([], {"a": [1, 2]}) == {}
    assert util.node_names([object()], {}) == {}


def test_generator_and_duplicate_targets():
    x = object()
    assert util.node_names((node for node in [x, x]), {"x": x}) == {x: ["x"]}


def test_shared_container_traversed_once():
    class CountingList(list[object]):
        iterations = 0

        def __iter__(self) -> Iterator[object]:
            self.iterations += 1
            return super().__iter__()

    x = object()
    shared = CountingList([x])
    nested = [shared, shared]
    assert util.node_names([x], {"a": nested, "b": nested})[x] == [
        "a[0][0]",
        "a[1][0]",
        "b[0][0]",
        "b[1][0]",
    ]
    assert shared.iterations == 1


def test_dictionary_keys_use_repr_but_are_not_searched():
    x, key = object(), object()
    assert util.node_names([x, key], {"a": {2: x, (1, 2): x, key: None}}) == {
        x: ["a[2]", "a[(1, 2)]"]
    }


def test_arbitrary_objects_and_iterators_are_not_inspected():
    class Holder:
        def __init__(self, value: object):
            self.value = value

    x = object()
    iterator = iter([x])
    assert util.node_names([x], {"holder": Holder(x), "iterator": iterator}) == {}
    assert next(iterator) is x


def test_self_cycle_rejected():
    values: list[object] = []
    values.append(values)
    with pytest.raises(ValueError, match="acyclic"):
        util.node_names([object()], {"values": values})


def test_indirect_cycle_rejected():
    a: dict[str, object] = {}
    b = [a]
    a["b"] = b
    with pytest.raises(ValueError, match="acyclic"):
        util.node_names([object()], {"a": a})


def test_matching_uses_identity_not_equality():
    class Equal:
        def __eq__(self, other: object) -> bool:
            return isinstance(other, Equal)

        def __hash__(self) -> int:
            return 0

    x, other = Equal(), Equal()
    assert util.node_names([x], {"other": other, "x": x}) == {x: ["x"]}


def test_selected_container_is_a_leaf():
    x = object()
    container = (x,)
    assert util.node_names([container, x], {"values": container}) == {container: ["values"]}


def test_locals():
    x = object()
    states = [x, object()]
    assert util.node_names(states, locals()) == {x: ["x", "states[0]"], states[1]: ["states[1]"]}


@pytest.mark.parametrize("namespace_type", [dict, UserDict, MappingProxyType])
def test_namespace_accepts_mappings(namespace_type):
    x = object()
    namespace = namespace_type({"later": [x], "earlier": x})
    assert util.node_names([x], namespace) == {x: ["later[0]", "earlier"]}


@pytest.mark.parametrize("name", ["", "not an identifier", "α", "a.b", "a[0]"])
def test_namespace_names_are_preserved_literally(name):
    x = object()
    assert util.node_names([x], {name: [x]}) == {x: [name + "[0]"]}


@pytest.mark.parametrize("key", ["", "a'b", 'a"b', "a\\b", "a\nb", "λ", -1, None, (2, "x")])
def test_dictionary_key_paths_round_trip(key):
    x = object()
    namespace = {"values": {key: x}}
    paths = util.node_names([x], namespace)[x]
    assert paths == [f"values[{key!r}]"]
    # The expression is generated solely from these fixed test keys.
    assert eval(paths[0], {"__builtins__": {}}, namespace) is x


def test_depth_first_not_shortest_path():
    x = object()
    namespace = {"first": {"deep": [[x]]}, "short": x}
    assert util.node_names([x], namespace)[x] == ["first['deep'][0][0]", "short"]


def test_result_key_order_follows_namespace_not_targets():
    x, y, z = object(), object(), object()
    names = util.node_names([z, y, x], {"first": [y, x], "second": z})
    assert list(names) == [y, x, z]


def test_namespace_reassignment_preserves_original_position():
    x = object()
    namespace = {"first": None, "second": x}
    namespace["first"] = x
    assert util.node_names([x], namespace)[x] == ["first", "second"]


def test_identical_rendered_paths_are_not_deduplicated():
    x = object()
    assert util.node_names([x], {"a[0]": x, "a": [x]})[x] == ["a[0]", "a[0]"]


@pytest.mark.parametrize("reverse", [False, True])
def test_distinct_objects_with_same_path_rejected(reverse):
    x, y = object(), object()
    entries = [("a[0]", x), ("a", [y])]
    namespace = dict(reversed(entries) if reverse else entries)
    with pytest.raises(ValueError, match=r"Name 'a\[0\]' refers to distinct objects"):
        util.node_names([x, y], namespace)


def test_custom_keys_with_same_repr_rejected():
    class Key:
        def __repr__(self) -> str:
            return "key"

    x, y = object(), object()
    namespace = {"values": {Key(): x, Key(): y}}
    with pytest.raises(ValueError, match="refers to distinct objects"):
        util.node_names([x, y], namespace)


def test_nan_keys_with_same_repr_rejected():
    x, y = object(), object()
    values = {float("nan"): x, float("nan"): y}
    assert len(values) == 2
    with pytest.raises(ValueError, match=r"values\[nan\]"):
        util.node_names([x, y], {"values": values})


def test_nan_keys_with_same_object_allowed():
    x = object()
    values = {float("nan"): x, float("nan"): x}
    assert util.node_names([x], {"values": values})[x] == ["values[nan]", "values[nan]"]


def test_collision_with_unselected_object_ignored():
    x, y = object(), object()
    assert util.node_names([x], {"a[0]": y, "a": [x]}) == {x: ["a[0]"]}


def test_collision_in_cached_paths_rejected():
    x, y = object(), object()
    shared = [x]
    with pytest.raises(ValueError, match="refers to distinct objects"):
        util.node_names([x, y], {"a": shared, "b[0]": y, "b": shared})


def test_collision_rejected_even_when_other_aliases_exist():
    x, y = object(), object()
    with pytest.raises(ValueError, match="refers to distinct objects"):
        util.node_names([x, y], {"x": x, "y": y, "a[0]": x, "a": [y]})


def test_distinct_nodes_with_hash_collisions():
    class Colliding:
        def __hash__(self) -> int:
            return 0

    x, y = Colliding(), Colliding()
    assert util.node_names([x, y], {"both": [x, y, x]}) == {
        x: ["both[0]", "both[2]"],
        y: ["both[1]"],
    }


def test_unselected_values_are_not_hashed_or_compared():
    class Hostile:
        def __hash__(self) -> int:
            raise AssertionError("Should not hash namespace leaves")

        def __eq__(self, other: object) -> bool:
            raise AssertionError("Should not compare namespace leaves")

    x = object()
    assert util.node_names([x], {"ignored": Hostile(), "x": x}) == {x: ["x"]}


def test_tuple_and_namedtuple_use_numeric_paths():
    Pair = namedtuple("Pair", ["left", "right"])
    x, y = object(), object()
    assert util.node_names([x, y], {"pair": Pair(x, (y, x))}) == {
        x: ["pair[0]", "pair[1][1]"],
        y: ["pair[1][0]"],
    }


def test_nested_generic_mappings_and_sets_are_leaves():
    x = object()
    assert (
        util.node_names([x], {"mapping": UserDict({"x": x}), "set": {x}, "frozen": frozenset([x])})
        == {}
    )


def test_selected_noncontainer_leaves():
    marker = frozenset([1])
    assert util.node_names([None, marker], {"nothing": None, "marker": marker}) == {
        None: ["nothing"],
        marker: ["marker"],
    }


def test_shared_dictionary_traversed_once():
    class CountingDict(dict[str, object]):
        visits = 0

        def items(self):
            self.visits += 1
            return super().items()

    x = object()
    shared = CountingDict(x=x)
    assert util.node_names([x], {"a": shared, "b": [shared, shared]})[x] == [
        "a['x']",
        "b[0]['x']",
        "b[1]['x']",
    ]
    assert shared.visits == 1


def test_equal_containers_are_cached_by_identity():
    class CountingList(list[object]):
        visits = 0

        def __iter__(self) -> Iterator[object]:
            self.visits += 1
            return super().__iter__()

    x = object()
    first, second = CountingList([x]), CountingList([x])
    assert util.node_names([x], {"a": first, "b": second, "c": first})[x] == [
        "a[0]",
        "b[0]",
        "c[0]",
    ]
    assert first.visits == second.visits == 1


def test_shared_containers_without_matches_are_cached():
    class CountingList(list[object]):
        visits = 0

        def __iter__(self) -> Iterator[object]:
            self.visits += 1
            return super().__iter__()

    shared = CountingList([object()])
    assert util.node_names([object()], {"a": shared, "b": shared}) == {}
    assert shared.visits == 1


def test_deeply_shared_binary_structure_retains_all_paths():
    x = object()
    structure: object = x
    for _ in range(8):
        structure = [structure, structure]
    expected = ["root" + "".join(f"[{i}]" for i in path) for path in product(range(2), repeat=8)]
    assert util.node_names([x], {"root": structure})[x] == expected


def test_deep_mixed_container_chain():
    x = object()
    structure: object = x
    prefixes = []
    for i in range(60):
        if i % 3 == 0:
            structure = {"next": structure}
            prefixes.append("['next']")
        elif i % 3 == 1:
            structure = (structure,)
            prefixes.append("[0]")
        else:
            structure = [None, structure]
            prefixes.append("[1]")
    assert util.node_names([x], {"root": structure})[x] == ["root" + "".join(reversed(prefixes))]


def test_names_do_not_share_mutable_result_lists():
    x, y = object(), object()
    namespace = {"values": [x, y]}
    first = util.node_names([x, y], namespace)
    second = util.node_names([x, y], namespace)
    first[x].append("extra")
    assert first[y] == ["values[1]"]
    assert second == {x: ["values[0]"], y: ["values[1]"]}


def test_cache_does_not_persist_between_calls():
    x, y = object(), object()
    shared = [x]
    namespace = {"a": shared, "b": shared}
    assert util.node_names([x, y], namespace) == {x: ["a[0]", "b[0]"]}
    shared[0] = y
    assert util.node_names([x, y], namespace) == {y: ["a[0]", "b[0]"]}


def test_cycle_through_tuple_rejected():
    values: list[object] = []
    wrapper = (values,)
    values.append(wrapper)
    with pytest.raises(ValueError, match="acyclic"):
        util.node_names([object()], {"root": wrapper})


def test_cycle_after_matching_leaf_is_still_rejected():
    x = object()
    values: list[object] = [x]
    values.append(values)
    with pytest.raises(ValueError, match="acyclic"):
        util.node_names([x], {"values": values})


def test_namespace_itself_can_be_part_of_a_cycle():
    namespace: dict[str, object] = {}
    namespace["self"] = namespace
    with pytest.raises(ValueError, match="acyclic"):
        util.node_names([object()], namespace)


def test_inputs_are_not_modified():
    x, y = object(), object()
    values = [x, y, x]
    nested = {"values": values}
    namespace = {"a": nested, "b": nested}
    util.node_names([x, y], namespace)
    assert list(namespace) == ["a", "b"]
    assert namespace["a"] is namespace["b"] is nested
    assert nested["values"] is values
    assert values == [x, y, x]
