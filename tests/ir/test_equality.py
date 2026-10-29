from pangolin.ir import (
    Add,
    Bijector,
    Composite,
    Constant,
    Exp,
    Identity,
    Log,
    Normal,
    RV,
    Scan,
    StudentT,
    Sum,
    VMap,
)


def test_parameterless_ops_equal_by_concrete_type():
    assert Normal() == Normal()
    assert Normal() != StudentT()
    assert Normal() != object()


def test_parameterized_op_equality_includes_constructor_state():
    assert Sum(0) == Sum(0)
    assert Sum(0) != Sum(1)


def test_stateful_op_equality_requires_same_concrete_type():
    class SumSubclass(Sum):
        pass

    assert Sum(0) != SumSubclass(0)
    assert SumSubclass(0) != Sum(0)
    assert SumSubclass(0) == SumSubclass(0)


def test_bijector_equality_requires_same_concrete_type():
    class BijectorSubclass(Bijector):
        pass

    log_det_jac = Composite(2, (Identity(),), ((0,),))
    bijector = Bijector(Exp(), Log(), log_det_jac)
    subclass = BijectorSubclass(Exp(), Log(), log_det_jac)

    assert bijector != subclass
    assert subclass != bijector


def test_equal_parameterized_ops_have_equal_hashes():
    equal_pairs = (
        (Sum(0), Sum(0)),
        (
            Composite(2, (Add(),), ((0, 1),)),
            Composite(2, (Add(),), [[0, 1]]),
        ),
        (Scan(Add(), 3, (0,)), Scan(Add(), 3, [0])),
    )

    for first, second in equal_pairs:
        assert first == second
        assert hash(first) == hash(second)


def test_op_equality_uses_normalized_constructor_state():
    from_list = VMap(Normal(), [0, None])
    from_tuple = VMap(Normal(), (0, None))

    assert from_list == from_tuple
    assert hash(from_list) == hash(from_tuple)


def test_equal_ops_are_the_same_set_and_dictionary_key():
    first = VMap(Normal(), [0, None])
    equivalent = VMap(Normal(), (0, None))
    distinct = VMap(Normal(), (None, 0))

    assert len({first, equivalent, distinct}) == 2

    values = {first: "first", equivalent: "equivalent", distinct: "distinct"}
    assert len(values) == 2
    assert values[first] == "equivalent"
    assert values[equivalent] == "equivalent"
    assert values[distinct] == "distinct"


def test_rv_equality_is_identity_for_deterministic_and_random_rvs():
    constant_x = RV(Constant(1))
    constant_y = RV(Constant(1))
    assert constant_x == constant_x
    assert constant_x != constant_y

    mean = RV(Constant(0))
    scale = RV(Constant(1))
    normal_x = RV(Normal(), mean, scale)
    normal_y = RV(Normal(), mean, scale)
    assert normal_x == normal_x
    assert normal_x != normal_y


def test_rv_hash_is_stable_and_consistent_with_identity_equality():
    x = RV(Constant(1))
    same_x = x

    assert hash(x) == hash(x)
    assert x == same_x
    assert hash(x) == hash(same_x)


def test_structurally_equivalent_rvs_are_distinct_set_and_dictionary_keys():
    x = RV(Constant(1))
    y = RV(Constant(1))

    assert len({x, x, y}) == 2

    values = {x: "x", y: "y"}
    assert len(values) == 2
    assert values[x] == "x"
    assert values[y] == "y"
