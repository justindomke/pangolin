from pangolin import interface as pi, ir


import numpy as np
import pytest

from pangolin.interface import sfor


def test_sfor_is_exported_by_the_public_interface():
    assert pi.sfor is sfor
    assert "sfor" in pi.__all__


def test_scalar_sequence_uses_second_parameter_as_scan_axis():
    x = sfor(lambda previous, t: pi.normal(previous + t, 1), init=0, t=4)

    assert x.shape == (4,)
    assert x.op == ir.Scan(
        ir.Composite(
            2,
            (ir.Add(), ir.Constant(1), ir.Normal()),
            ((0, 1), (), (2, 3)),
        ),
        4,
        (0,),
    )
    assert x.parents[0].op == ir.Constant(0)
    assert x.parents[1].op == ir.Constant([0, 1, 2, 3])


def test_parameter_order_not_axis_name_selects_sequential_axis():
    x = sfor(
        lambda previous, i, t: pi.normal(previous[t], 1),
        init=np.zeros(3),
        t=3,
        i=2,
    )

    assert x.shape == (2, 3)
    assert x.op == ir.Scan(
        ir.Composite(
            1,
            (ir.Constant(1), ir.VMap(ir.Normal(), (0, None), 3)),
            ((), (0, 1)),
        ),
        2,
        (),
    )
    assert x.parents[0].op == ir.Constant(np.zeros(3))


def test_parallel_parameters_become_vmaps_inside_scan():
    scale = pi.constant(np.ones((3, 2)))

    x = sfor(
        lambda previous, t, i: pi.normal(previous[i], scale[t, i]),
        init=np.zeros(2),
        t=3,
        i=2,
    )

    assert x.shape == (3, 2)
    assert x.op == ir.Scan(ir.VMap(ir.Normal(), (0, 0), 2), 3, (0,))
    assert len(x.parents) == 2
    assert x.parents[0].op == ir.Constant(np.zeros(2))
    assert x.parents[1] is scale


def test_multiple_parallel_axes_have_exact_nested_vmap_order():
    x = sfor(
        lambda previous, time, row, column: pi.normal(previous[row, column], 1),
        init=np.zeros((3, 4)),
        column=4,
        time=2,
        row=3,
    )

    assert x.shape == (2, 3, 4)
    assert x.op == ir.Scan(
        ir.Composite(
            1,
            (
                ir.Constant(1),
                ir.VMap(
                    ir.VMap(ir.Normal(), (0, None), 4),
                    (0, None),
                    3,
                ),
            ),
            ((), (0, 1)),
        ),
        2,
        (),
    )


def test_body_receives_complete_previous_state():
    x = sfor(
        lambda previous, t, i: pi.normal(previous[1], 1),
        init=np.zeros(2),
        t=3,
        i=2,
    )

    assert x.shape == (3, 2)
    assert x.op == ir.Scan(
        ir.Composite(
            1,
            (
                ir.Constant(1),
                ir.Index(),
                ir.Constant(1),
                ir.VMap(ir.Normal(), (None, None), 2),
            ),
            ((), (0, 1), (), (2, 3)),
        ),
        3,
        (),
    )
    assert x.parents[0].op == ir.Constant(np.zeros(2))


def test_mapped_component_can_read_neighbor_from_complete_previous_state():
    x = sfor(
        lambda previous, t, i: pi.normal(previous[1 - i], 1),
        init=np.zeros(2),
        t=3,
        i=2,
    )

    assert x.op == ir.Scan(
        ir.Composite(
            1,
            (
                ir.Constant([0, 1]),
                ir.Constant(1),
                ir.VMap(ir.Sub(), (None, 0), 2),
                ir.VMap(ir.Index(), (None, 0), 2),
                ir.Constant(1),
                ir.VMap(ir.Normal(), (0, None), 2),
            ),
            ((), (), (2, 1), (0, 3), (), (4, 5)),
        ),
        3,
        (),
    )
    assert x.parents[0].op == ir.Constant(np.zeros(2))


def test_each_parallel_component_may_return_an_event_vector():
    x = sfor(
        lambda previous, t, i: pi.normal(previous[i, :], 1),
        init=np.zeros((2, 3)),
        t=4,
        i=2,
    )

    assert x.shape == (4, 2, 3)
    assert x.op == ir.Scan(
        ir.Composite(
            1,
            (
                ir.Constant(1),
                ir.VMap(
                    ir.VMap(ir.Normal(), (0, None), 3),
                    (0, None),
                    2,
                ),
            ),
            ((), (0, 1)),
        ),
        4,
        (),
    )


def test_decorator_and_direct_forms_build_equal_graphs():
    direct = sfor(
        lambda previous, t, i: pi.normal(previous[i], 1),
        init=np.zeros(2),
        t=3,
        i=2,
    )

    @sfor(init=np.zeros(2), i=2, t=3)
    def decorated(previous, t, i):
        return pi.normal(previous[i], 1)

    assert direct.op == decorated.op
    assert direct.parents[0].op == decorated.parents[0].op


def test_keyword_order_does_not_change_axis_order():
    ordered = sfor(
        lambda previous, t, i: pi.normal(previous[i], 1),
        init=np.zeros(2),
        t=3,
        i=2,
    )
    shuffled = sfor(
        lambda previous, t, i: pi.normal(previous[i], 1),
        init=np.zeros(2),
        i=2,
        t=3,
    )

    assert ordered.op == shuffled.op
    assert ordered.parents[0].op == shuffled.parents[0].op


def test_captured_rvs_follow_explicit_stable_parent_order():
    offset = pi.constant(5)
    scale = pi.constant(6)

    x = sfor(
        lambda previous, t: pi.normal(previous + offset, scale + t),
        init=0,
        t=3,
    )

    assert x.op == ir.Scan(
        ir.Composite(
            4,
            (ir.Add(), ir.Add(), ir.Normal()),
            ((0, 2), (3, 1), (4, 5)),
        ),
        3,
        (0, None, None),
    )
    assert x.parents[0].op == ir.Constant(0)
    assert x.parents[1].op == ir.Constant([0, 1, 2])
    assert x.parents[2] is offset
    assert x.parents[3] is scale


def test_unused_sequential_coordinate_is_eliminated_from_op_and_parents():
    x = sfor(lambda previous, t: pi.normal(previous, 1), init=0, t=100)

    assert x.op == ir.Scan(
        ir.Composite(
            1,
            (ir.Constant(1), ir.Normal()),
            ((), (0, 1)),
        ),
        100,
        (),
    )
    assert len(x.parents) == 1
    assert x.parents[0].op == ir.Constant(0)


def test_existing_initial_rv_is_preserved_by_identity():
    initial = pi.normal(0, 1)

    x = sfor(lambda previous, t: pi.normal(previous, 1), init=initial, t=3)

    assert x.parents == (initial,)
    assert x.op == ir.Scan(
        ir.Composite(
            1,
            (ir.Constant(1), ir.Normal()),
            ((), (0, 1)),
        ),
        3,
        (),
    )


def test_callback_is_traced_once_regardless_of_axis_sizes():
    calls = []

    def body(previous, t, i, j):
        calls.append((previous.shape, t.shape, i.shape, j.shape))
        return pi.normal(previous[i, j], 1)

    x = sfor(body, init=np.zeros((2, 3)), t=1000, i=2, j=3)

    assert x.shape == (1000, 2, 3)
    assert calls == [((2, 3), (), (), ())]


def test_explicit_inner_vfor_builds_the_same_op_as_parallel_sfor_axis():
    compact = sfor(
        lambda previous, t, i: pi.normal(previous[i], 1),
        init=np.zeros(2),
        t=3,
        i=2,
    )
    explicit = sfor(
        lambda previous, t: pi.vfor(
            lambda i: pi.normal(previous[i], 1),
            i=2,
        ),
        init=np.zeros(2),
        t=3,
    )

    expected_op = ir.Scan(
        ir.Composite(
            1,
            (ir.Constant(1), ir.VMap(ir.Normal(), (0, None), 2)),
            ((), (0, 1)),
        ),
        3,
        (),
    )
    assert compact.op == explicit.op == expected_op


def test_sfor_inside_vfor_builds_exact_vmap_of_scan():
    x = pi.vfor(
        lambda i: sfor(
            lambda previous, t: pi.normal(previous + i, 1),
            init=0,
            t=3,
        ),
        i=2,
    )

    expected_scan = ir.Scan(
        ir.Composite(
            2,
            (ir.Add(), ir.Constant(1), ir.Normal()),
            ((0, 1), (), (2, 3)),
        ),
        3,
        (None,),
    )
    assert x.shape == (2, 3)
    assert x.op == ir.VMap(expected_scan, (None, 0), 2)
    assert x.parents[0].op == ir.Constant(0)
    assert x.parents[1].op == ir.Constant([0, 1])


def test_numpy_integer_axis_sizes_are_accepted_and_normalized():
    x = sfor(
        lambda previous, t, i: pi.normal(previous[i], 1),
        init=np.zeros(2),
        t=np.int64(3),
        i=np.int32(2),
    )

    assert x.shape == (3, 2)
    assert isinstance(x.op, ir.Scan)
    assert x.op.length == 3
    assert isinstance(x.op.base_op, ir.Composite)
    mapped_normal = x.op.base_op.ops[-1]
    assert isinstance(mapped_normal, ir.VMap)
    assert mapped_normal.axis_size == 2


def test_positional_only_callback_parameters_are_supported():
    def body(previous, t, /, i):
        return pi.normal(previous[i], 1)

    x = sfor(body, init=np.zeros(2), t=3, i=2)

    assert x.op == ir.Scan(
        ir.Composite(
            1,
            (ir.Constant(1), ir.VMap(ir.Normal(), (0, None), 2)),
            ((), (0, 1)),
        ),
        3,
        (),
    )


def test_requires_state_and_sequential_parameters():
    with pytest.raises(
        ValueError,
        match="must accept a state and a sequential coordinate",
    ):
        sfor(lambda previous: pi.normal(previous, 1), init=0)


@pytest.mark.parametrize("kind", ["default", "variadic", "keyword_only"])
def test_rejects_callbacks_without_fixed_positional_signatures(kind):
    if kind == "default":

        def body(previous, t=3):
            return pi.normal(previous, 1)

    elif kind == "variadic":

        def body(previous, *coordinates):
            return pi.normal(previous, 1)

    else:

        def body(previous, *, t):
            return pi.normal(previous, 1)

    with pytest.raises(TypeError, match="has no fixed positional signature"):
        sfor(body, init=0, t=3)


def test_rejects_missing_axis_size():
    with pytest.raises(ValueError, match="missing size for axis 'i'"):
        sfor(
            lambda previous, t, i: pi.normal(previous[i], 1),
            init=np.zeros(2),
            t=3,
        )


def test_rejects_unknown_axis_size():
    with pytest.raises(ValueError, match="size for unknown axis 'j'"):
        sfor(lambda previous, t: pi.normal(previous, 1), init=0, t=3, j=2)


@pytest.mark.parametrize("size", [0, -1])
def test_rejects_nonpositive_axis_sizes(size):
    with pytest.raises(ValueError, match="must be positive"):
        sfor(lambda previous, t: pi.normal(previous, 1), init=0, t=size)


@pytest.mark.parametrize("size", [True, 1.5, "3", None])
def test_rejects_noninteger_axis_sizes(size):
    with pytest.raises(TypeError, match="must be an integer"):
        sfor(lambda previous, t: pi.normal(previous, 1), init=0, t=size)


def test_rejects_pytree_output():
    with pytest.raises(TypeError, match="must return exactly one ir.RV"):
        sfor(
            lambda previous, t: (pi.normal(previous, 1), pi.normal(previous, 1)),
            init=0,
            t=3,
        )


def test_rejects_output_that_does_not_depend_on_previous_state():
    with pytest.raises(ValueError, match="output must depend on its carry"):
        sfor(lambda previous, t: pi.normal(t, 1), init=0, t=3)


def test_rejects_returning_previous_state_or_an_existing_rv():
    with pytest.raises(ValueError, match="every output must be created"):
        sfor(lambda previous, t: previous, init=0, t=3)

    existing = pi.normal(0, 1)
    with pytest.raises(ValueError, match="every output must be created"):
        sfor(lambda previous, t: existing, init=0, t=3)


def test_rejects_random_intermediate_rv_inside_scan_step():
    def body(previous, t):
        intermediate = pi.normal(previous, 1)
        return pi.normal(intermediate, 1)

    with pytest.raises(ValueError, match="all but the last Op must be deterministic"):
        sfor(body, init=0, t=3)


def test_scan_rejects_output_shape_different_from_initial_state():
    with pytest.raises(ValueError, match="match carry shape"):
        sfor(
            lambda previous, t, i: pi.normal(previous[0], 1),
            init=np.zeros(3),
            t=2,
            i=2,
        )


def test_decorator_form_requires_keyword_init():
    with pytest.raises(TypeError, match="required keyword-only argument: 'init'"):
        sfor(t=3)


def test_direct_form_requires_keyword_init():
    with pytest.raises(TypeError, match="required keyword-only argument: 'init'"):
        sfor(lambda previous, t: pi.normal(previous, 1), t=3)

    with pytest.raises(TypeError, match="takes from 1 to 2 positional arguments but 3 were given"):
        sfor(lambda previous, t: pi.normal(previous, 1), 0, t=3)


@pytest.mark.parametrize(
    "body",
    [
        lambda init, t: pi.normal(init, 1),
        lambda previous, init: pi.normal(previous, 1),
    ],
)
def test_rejects_init_as_any_function_parameter_name(body):
    with pytest.raises(ValueError, match="function parameters cannot be named 'init'"):
        sfor(body, init=0, t=3)
