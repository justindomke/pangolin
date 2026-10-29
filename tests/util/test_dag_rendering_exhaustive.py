from itertools import permutations, product

import pytest

from pangolin.util import dag_lines

_SUBSCRIPTS = "₀₁₂₃₄₅₆₇₈₉"
_LINE_GLYPHS = {"│", "─", "└", "├", "┐", "┴", "┼"}


def _ordered_subsets(items):
    """Yield every subset in every possible order."""
    for length in range(len(items) + 1):
        yield from permutations(items, length)


def _ordered_dags(num_nodes):
    """Yield every ordered parent list compatible with one topological order."""
    choices = [_ordered_subsets(tuple(range(node))) for node in range(num_nodes)]
    yield from product(*choices)


def _parse_drawing(lines):
    """Separate the fixed-width drawing from test-only node labels."""
    node_rows = {}
    label_width = None
    split_lines = []

    for row, line in enumerate(lines):
        if "@" in line:
            drawing, raw_node = line.split("@", maxsplit=1)
            node = int(raw_node)
            if label_width is None:
                label_width = len(drawing)
            else:
                assert len(drawing) == label_width
            node_rows[row] = node
            split_lines.append(drawing)
        else:
            split_lines.append(line)

    assert label_width is not None
    grid = [line.ljust(label_width) for line in split_lines]
    node_positions = {}
    for row, node in node_rows.items():
        assert grid[row].count("○") == 1
        node_positions[node] = (row, grid[row].index("○"))

    return grid, node_positions


def _decode_parents(grid, node_positions):
    """Trace downward/rightward routes, treating every ``┼`` as a crossing."""
    nodes_by_position = {position: node for node, position in node_positions.items()}
    decoded = {}
    vertical = {}

    for row, cells in enumerate(grid):
        horizontal = frozenset()
        next_vertical = {}

        for column, glyph in enumerate(cells):
            incoming_vertical = vertical.get(column, frozenset())

            if glyph in _SUBSCRIPTS or glyph == " ":
                assert not incoming_vertical
                assert not horizontal
            elif glyph == "│":
                assert incoming_vertical
                assert not horizontal
                next_vertical[column] = incoming_vertical
            elif glyph == "─":
                assert horizontal
                assert not incoming_vertical
            elif glyph == "└":
                assert incoming_vertical
                assert not horizontal
                horizontal = incoming_vertical
            elif glyph == "├":
                assert incoming_vertical
                assert not horizontal
                horizontal = incoming_vertical
                next_vertical[column] = incoming_vertical
            elif glyph == "┐":
                assert horizontal
                assert not incoming_vertical
                next_vertical[column] = horizontal
                horizontal = frozenset()
            elif glyph == "┴":
                assert horizontal
                assert incoming_vertical
                horizontal |= incoming_vertical
            elif glyph == "┼":
                assert horizontal
                assert incoming_vertical
                next_vertical[column] = incoming_vertical
            elif glyph == "○":
                node = nodes_by_position[row, column]
                decoded[node] = horizontal | incoming_vertical
                horizontal = frozenset()

                if row + 1 < len(grid) and grid[row + 1][column] in {
                    "│",
                    "└",
                    "├",
                    "┴",
                    "┼",
                }:
                    next_vertical[column] = frozenset({node})
            else:
                raise AssertionError(f"unexpected drawing glyph {glyph!r}")

        assert not horizontal
        vertical = next_vertical

    assert not vertical
    return decoded


def _assert_parent_numbers(grid, node_positions, parents, mode):
    node_row = {node: row for node, (row, _) in node_positions.items()}
    node_column = {node: column for node, (_, column) in node_positions.items()}

    previous_row = -1
    for node in range(len(parents)):
        row = node_row[node]
        connector_rows = grid[previous_row + 1 : row]
        actual = []
        for offset, cells in enumerate(connector_rows, start=previous_row + 1):
            for column, glyph in enumerate(cells):
                if glyph in _SUBSCRIPTS:
                    actual.append((offset, column, glyph))

        parent_columns = [node_column[parent] for parent in parents[node]]
        if mode == "on":
            numbered = bool(parents[node])
        elif mode == "auto":
            numbered = any(right < left for left, right in zip(parent_columns, parent_columns[1:]))
        else:
            numbered = False

        expected = {node_column[parent] + 1: _SUBSCRIPTS[slot] for slot, parent in enumerate(parents[node])}
        if not numbered:
            expected = {}

        assert {(column, glyph) for _, column, glyph in actual} == set(expected.items())
        assert len({row for row, _, _ in actual}) <= 1
        if actual:
            number_row = connector_rows[actual[0][0] - previous_row - 1]
            assert set(number_row) <= {" ", "│", *_SUBSCRIPTS}

        previous_row = row


def _assert_drawing(nodes, parents, mode):
    lines = dag_lines(
        nodes,
        parents,
        label_fn=lambda node: f"@{node}",
        number_parents=mode,
    )
    if not nodes:
        assert lines == []
        return

    grid, node_positions = _parse_drawing(lines)
    assert set("".join(grid)) <= {" ", "○", *_LINE_GLYPHS, *_SUBSCRIPTS}
    assert [node for _, node in sorted((row, node) for node, (row, _) in node_positions.items())] == list(nodes)

    decoded = _decode_parents(grid, node_positions)
    assert decoded == {node: frozenset(node_parents) for node, node_parents in zip(nodes, parents)}
    _assert_parent_numbers(grid, node_positions, parents, mode)


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (
            ["○ @0", "│", "○ @1"],
            {0: frozenset(), 1: frozenset({0})},
        ),
        (
            ["○   @0", "│", "├─○ @1", "│", "○   @2"],
            {0: frozenset(), 1: frozenset({0}), 2: frozenset({0})},
        ),
        (
            [
                "○     @0",
                "│",
                "│ ○   @1",
                "│ │",
                "│ │ ○ @2",
                "│ │ │",
                "└─┴─○ @3",
            ],
            {
                0: frozenset(),
                1: frozenset(),
                2: frozenset(),
                3: frozenset({0, 1, 2}),
            },
        ),
        (
            [
                "○     @0",
                "│",
                "│ ○   @1",
                "│ │",
                "├─┼─○ @2",
                "│ │",
                "○ │   @3",
                "  │",
                "  ○   @4",
            ],
            {
                0: frozenset(),
                1: frozenset(),
                2: frozenset({0}),
                3: frozenset({0}),
                4: frozenset({1}),
            },
        ),
        (
            [
                "○     @0",
                "│",
                "│ ○   @1",
                "│ ├┐",
                "└─┼┴○ @2",
                "  │",
                "  ○   @3",
            ],
            {
                0: frozenset(),
                1: frozenset(),
                2: frozenset({0, 1}),
                3: frozenset({1}),
            },
        ),
    ],
)
def test_decoder_against_handwritten_drawings(lines, expected):
    grid, node_positions = _parse_drawing(lines)
    assert _decode_parents(grid, node_positions) == expected


def test_decoder_does_not_confuse_a_crossing_with_a_join():
    lines = [
        "○     @0",
        "│",
        "│ ○   @1",
        "│ │",
        "├─┼─○ @2",
        "│ │",
        "○ │   @3",
        "  │",
        "  ○   @4",
    ]
    grid, node_positions = _parse_drawing(lines)
    grid[4] = grid[4].replace("┼", "┴")
    with pytest.raises(AssertionError):
        _decode_parents(grid, node_positions)


def test_decoder_rejects_a_broken_bend():
    lines = [
        "○     @0",
        "│",
        "│ ○   @1",
        "│ ├┐",
        "└─┼┴○ @2",
        "  │",
        "  ○   @3",
    ]
    grid, node_positions = _parse_drawing(lines)
    grid[4] = grid[4].replace("┴", "─")
    with pytest.raises(AssertionError):
        _decode_parents(grid, node_positions)


def test_number_decoder_against_a_handwritten_numbered_bend():
    lines = [
        "○     @0",
        "│",
        "│ ○   @1",
        "│₀│₁",
        "│ ├┐",
        "└─┼┴○ @2",
        "  │₀",
        "  ○   @3",
    ]
    parents = [(), (), (0, 1), (1,)]
    grid, node_positions = _parse_drawing(lines)
    _assert_parent_numbers(grid, node_positions, parents, "on")

    grid[3] = grid[3].replace("₀", "₂")
    with pytest.raises(AssertionError):
        _assert_parent_numbers(grid, node_positions, parents, "on")


@pytest.mark.parametrize("num_nodes", range(5))
def test_every_ordered_dag_through_four_nodes(num_nodes):
    nodes = tuple(range(num_nodes))
    for parents in _ordered_dags(num_nodes):
        for mode in ("off", "auto", "on"):
            _assert_drawing(nodes, parents, mode)
