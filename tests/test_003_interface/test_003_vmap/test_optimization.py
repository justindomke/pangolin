from pangolin import dag, interface as pi, ir
from pangolin.interface.simplifying import permutation_view


def upstream_nodes(x):
    return dag.upstream_nodes(x)


def op_contains_index(op: ir.Op) -> bool:
    """Whether Index occurs inside an op's nested VMap structure."""

    if isinstance(op, ir.Index):
        return True

    if isinstance(op, ir.VMap):
        return op_contains_index(op.base_op)

    return False


def contains_index(x) -> bool:
    """Whether any node upstream of x contains an Index op."""

    return any(op_contains_index(node.op) for node in dag.upstream_nodes(x))


def assert_vmap_axis(x, expected_axis: int):
    assert isinstance(x.op, ir.VMap)
    assert x.op.in_axes == (expected_axis,)


# ---------------------------------------------------------------------------
# optimize=False behavior
# ---------------------------------------------------------------------------


def test_vmap_optimization_enabled_by_default():
    a = pi.constant(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    columns = pi.constant(range(3))

    default = pi.vmap(lambda j: pi.exp(a[:, j]))(columns)

    explicit = pi.vmap(
        lambda j: pi.exp(a[:, j]),
        optimize=True,
    )(columns)

    assert ir.rv_equal(default, explicit)

    assert not contains_index(default)
    assert not contains_index(explicit)

    assert default.parents == (a,)
    assert explicit.parents == (a,)

    assert_vmap_axis(default, 1)
    assert_vmap_axis(explicit, 1)


def test_vmap_optimization_can_be_disabled():
    a = pi.constant(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    columns = pi.constant(range(3))

    optimized = pi.vmap(lambda j: pi.exp(a[:, j]))(columns)

    unoptimized = pi.vmap(
        lambda j: pi.exp(a[:, j]),
        optimize=False,
    )(columns)

    assert optimized.shape == unoptimized.shape == (3, 2)

    assert not contains_index(optimized)
    assert contains_index(unoptimized)

    assert optimized.parents == (a,)
    assert_vmap_axis(optimized, 1)

    assert isinstance(unoptimized.op, ir.VMap)
    assert len(unoptimized.parents) == 1

    indexed_parent = unoptimized.parents[0]
    assert isinstance(indexed_parent.op, ir.VMap)
    assert isinstance(indexed_parent.op.base_op, ir.Index)

    assert permutation_view(indexed_parent) == (a, (1, 0))


# ---------------------------------------------------------------------------
# Eligible intermediate permutations
# ---------------------------------------------------------------------------


def test_optimize_intermediate_axis_zero():
    """a[i, :] is identity; mapped dimension remains base axis zero."""

    a = pi.constant(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    rows = pi.constant(range(2))

    x = pi.vmap(
        lambda i: pi.exp(a[i, :]),
        optimize=True,
    )(rows)

    assert x.shape == (2, 3)
    assert not contains_index(x)

    # The eliminated view was a itself, and the mapped dimension is axis 0.
    assert x.parents == (a,)
    assert_vmap_axis(x, 0)


def test_optimize_intermediate_axis_one():
    """a[:, j] is a transpose view; mapped dimension is base axis one."""

    a = pi.constant(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    columns = pi.constant(range(3))

    x = pi.vmap(
        lambda j: pi.exp(a[:, j]),
        optimize=True,
    )(columns)

    assert x.shape == (3, 2)
    assert not contains_index(x)

    # The intermediate transpose disappears. Exp maps base axis one directly.
    assert x.parents == (a,)
    assert_vmap_axis(x, 1)


def test_optimize_intermediate_axis_two():
    a = pi.constant(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ]
    )
    depths = pi.constant(range(4))

    x = pi.vmap(
        lambda k: pi.exp(a[:, :, k]),
        optimize=True,
    )(depths)

    assert x.shape == (4, 2, 3)
    assert not contains_index(x)

    assert x.parents == (a,)
    assert_vmap_axis(x, 2)


def test_optimize_intermediate_middle_axis_of_3d_array():
    a = pi.constant(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ]
    )
    middle = pi.constant(range(3))

    x = pi.vmap(
        lambda j: pi.exp(a[:, j, :]),
        optimize=True,
    )(middle)

    assert x.shape == (3, 2, 4)
    assert not contains_index(x)

    assert x.parents == (a,)
    assert_vmap_axis(x, 1)


# ---------------------------------------------------------------------------
# Output-node restrictions
# ---------------------------------------------------------------------------


def test_optimize_identity_output_returns_base():
    """An identity permutation is safe even when it is the vmap output."""

    a = pi.constant(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    rows = pi.constant(range(2))

    x = pi.vmap(
        lambda i: a[i, :],
        optimize=True,
    )(rows)

    assert x is a


def test_optimize_transpose_output_is_not_removed():
    """An output must retain its mapped dimension at axis zero."""

    a = pi.constant(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    columns = pi.constant(range(3))

    x = pi.vmap(
        lambda j: a[:, j],
        optimize=True,
    )(columns)

    assert x.shape == (3, 2)
    assert contains_index(x)
    assert permutation_view(x) == (a, (1, 0))


def test_optimize_last_axis_output_is_not_removed():
    a = pi.constant(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ]
    )
    depths = pi.constant(range(4))

    x = pi.vmap(
        lambda k: a[:, :, k],
        optimize=True,
    )(depths)

    assert x.shape == (4, 2, 3)
    assert contains_index(x)
    assert permutation_view(x) == (a, (2, 0, 1))


def test_node_that_is_also_output_must_remain_materialized():
    """One generated node is both an output and a parent of another output."""

    a = pi.constant(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    columns = pi.constant(range(3))

    def f(j):
        column = a[:, j]
        return column, pi.exp(column)

    column, exponential = pi.vmap(
        f,
        optimize=True,
    )(columns)

    assert column.shape == (3, 2)
    assert exponential.shape == (3, 2)

    # column must retain the public vmap output layout.
    assert contains_index(column)
    assert permutation_view(column) == (a, (1, 0))

    # Since the same node is also an output, downstream code must consume
    # that materialized output rather than silently replacing it with a.
    assert exponential.parents == (column,)
    assert_vmap_axis(exponential, 0)


# ---------------------------------------------------------------------------
# Propagation through downstream nodes
# ---------------------------------------------------------------------------


def test_optimized_axis_propagates_through_multiple_nodes():
    a = pi.constant(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    columns = pi.constant(range(3))

    x = pi.vmap(
        lambda j: pi.log(pi.exp(a[:, j])),
        optimize=True,
    )(columns)

    assert x.shape == (3, 2)
    assert not contains_index(x)

    # The first operation consumes axis one of a.
    exp_node = x.parents[0]
    assert exp_node.parents == (a,)
    assert_vmap_axis(exp_node, 1)

    # After Exp produces its result, its mapped dimension is conventionally
    # output axis zero again.
    assert_vmap_axis(x, 0)


def test_optimized_view_with_multiple_consumers():
    a = pi.constant(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    columns = pi.constant(range(3))

    def f(j):
        column = a[:, j]
        return pi.exp(column), pi.log(column)

    exponential, logarithm = pi.vmap(
        f,
        optimize=True,
    )(columns)

    assert exponential.shape == (3, 2)
    assert logarithm.shape == (3, 2)

    assert not contains_index((exponential, logarithm))

    assert exponential.parents == (a,)
    assert logarithm.parents == (a,)

    assert_vmap_axis(exponential, 1)
    assert_vmap_axis(logarithm, 1)


def test_two_separate_permutation_views_are_both_optimized():
    a = pi.constant(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    columns = pi.constant(range(3))

    def f(j):
        first = pi.exp(a[:, j])
        second = pi.log(a[:, j])
        return first, second

    first, second = pi.vmap(
        f,
        optimize=True,
    )(columns)

    assert not contains_index((first, second))
    assert first.parents == (a,)
    assert second.parents == (a,)
    assert_vmap_axis(first, 1)
    assert_vmap_axis(second, 1)


# ---------------------------------------------------------------------------
# Near misses that must not be optimized
# ---------------------------------------------------------------------------


def test_paired_indexing_is_not_optimized():
    """vmap(lambda i, j: a[i, j]) extracts paired entries."""

    a = pi.constant(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    i = pi.constant(range(2))
    j = pi.constant(range(2))

    x = pi.vmap(
        lambda i, j: pi.exp(a[i, j]),
        optimize=True,
    )(i, j)

    assert x.shape == (2,)
    assert contains_index(x)


def test_swapped_paired_indexing_is_not_optimized():
    a = pi.constant(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    i = pi.constant(range(2))
    j = pi.constant(range(2))

    x = pi.vmap(
        lambda i, j: pi.exp(a[j, i]),
        optimize=True,
    )(i, j)

    assert x.shape == (2,)
    assert contains_index(x)


def test_reversed_index_is_not_optimized():
    a = pi.constant(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    reverse_columns = pi.constant([2, 1, 0])

    x = pi.vmap(
        lambda j: pi.exp(a[:, j]),
        optimize=True,
    )(reverse_columns)

    assert x.shape == (3, 2)
    assert contains_index(x)


def test_repeated_index_is_not_optimized():
    a = pi.constant(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    repeated_columns = pi.constant([0, 0, 2])

    x = pi.vmap(
        lambda j: pi.exp(a[:, j]),
        optimize=True,
    )(repeated_columns)

    assert x.shape == (3, 2)
    assert contains_index(x)


def test_nonstable_permutation_is_not_optimized():
    """Permutation (1, 2, 0) moves axis one first but also reorders the rest."""

    a = pi.constant(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ]
    )
    middle = pi.constant(range(3))
    last = pi.constant(range(4))

    def f(j):
        reordered = pi.vmap(
            lambda k: a[:, j, k],
        )(last)

        return pi.exp(reordered)

    x = pi.vmap(
        f,
        optimize=True,
    )(middle)

    assert x.shape == (3, 4, 2)
    assert contains_index(x)

    permutation_nodes = [node for node in upstream_nodes(x) if permutation_view(node) is not None]

    assert any(permutation_view(node) == (a, (1, 2, 0)) for node in permutation_nodes)


# ---------------------------------------------------------------------------
# Multiple inputs
# ---------------------------------------------------------------------------


def test_optimized_view_used_with_an_unmapped_parent():
    a = pi.constant(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    offset = pi.constant([10.0, 20.0])
    columns = pi.constant(range(3))

    x = pi.vmap(
        lambda j: a[:, j] + offset,
        optimize=True,
    )(columns)

    assert x.shape == (3, 2)
    assert not contains_index(x)

    assert isinstance(x.op, ir.VMap)

    # a's logical mapped dimension is its axis one; offset is not mapped
    # by the outer vmap.
    assert x.op.in_axes == (1, None)
    assert x.parents == (a, offset)


def test_two_optimized_views_with_matching_nonzero_axes():
    a = pi.constant(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    b = pi.constant(
        [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ]
    )
    columns = pi.constant(range(3))

    x = pi.vmap(
        lambda j: a[:, j] + b[:, j],
        optimize=True,
    )(columns)

    assert x.shape == (3, 2)
    assert not contains_index(x)

    assert isinstance(x.op, ir.VMap)
    assert x.op.in_axes == (1, 1)
    assert x.parents == (a, b)
