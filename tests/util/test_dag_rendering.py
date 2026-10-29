import pytest

from pangolin.util import render_dag, dag_lines


def assert_numbering_modes(nodes, parents, *, off, auto, on):
    assert dag_lines(nodes, parents, number_parents="off") == off
    assert dag_lines(nodes, parents, number_parents="auto") == auto
    assert dag_lines(nodes, parents, number_parents="on") == on


def test_single_node():
    expected = ["○ A"]
    assert_numbering_modes(["A"], [[]], off=expected, auto=expected, on=expected)


def test_linear_chain():
    nodes = ["A", "B", "C"]
    parents = [[], ["A"], ["B"]]
    unnumbered = ["○ A", "│", "○ B", "│", "○ C"]
    assert_numbering_modes(
        nodes,
        parents,
        off=unnumbered,
        auto=unnumbered,
        on=["○ A", "│₀", "○ B", "│₀", "○ C"],
    )


def test_fan_out_and_fan_in():
    nodes = ["A", "B", "C"]

    parents = [[], ["A"], ["A"]]
    unnumbered = ["○   A", "│", "├─○ B", "│", "○   C"]
    assert_numbering_modes(
        nodes,
        parents,
        off=unnumbered,
        auto=unnumbered,
        on=["○   A", "│₀", "├─○ B", "│₀", "○   C"],
    )

    parents = [[], [], ["A", "B"]]
    unnumbered = ["○   A", "│", "│ ○ B", "│ │", "└─○ C"]
    assert_numbering_modes(
        nodes,
        parents,
        off=unnumbered,
        auto=unnumbered,
        on=["○   A", "│", "│ ○ B", "│₀│₁", "└─○ C"],
    )


def test_interior_parent_bends_are_distinct_from_crossings():
    nodes = ["A", "B", "C", "D"]
    parents = [[], ["A"], ["A", "B"], ["A", "B", "C"]]
    unnumbered = [
        "○     A",
        "│",
        "├─○   B",
        "│ ├┐",
        "├─┼┴○ C",
        "│ │ │",
        "└─┴─○ D",
    ]
    assert_numbering_modes(
        nodes,
        parents,
        off=unnumbered,
        auto=unnumbered,
        on=[
            "○     A",
            "│₀",
            "├─○   B",
            "│₀│₁",
            "│ ├┐",
            "├─┼┴○ C",
            "│₀│₁│₂",
            "└─┴─○ D",
        ],
    )


def test_interior_parent_bends_right_and_continues():
    nodes = ["A", "B", "C", "D"]
    parents = [[], [], ["A", "B"], ["B"]]
    unnumbered = [
        "○     A",
        "│",
        "│ ○   B",
        "│ ├┐",
        "└─┼┴○ C",
        "  │",
        "  ○   D",
    ]
    assert_numbering_modes(
        nodes,
        parents,
        off=unnumbered,
        auto=unnumbered,
        on=[
            "○     A",
            "│",
            "│ ○   B",
            "│₀│₁",
            "│ ├┐",
            "└─┼┴○ C",
            "  │₀",
            "  ○   D",
        ],
    )


def test_terminating_interior_parent_joins_without_a_bend():
    nodes = ["A", "B", "C", "D"]
    parents = [[], [], [], ["A", "B", "C"]]
    unnumbered = [
        "○     A",
        "│",
        "│ ○   B",
        "│ │",
        "│ │ ○ C",
        "│ │ │",
        "└─┴─○ D",
    ]
    assert_numbering_modes(
        nodes,
        parents,
        off=unnumbered,
        auto=unnumbered,
        on=[
            "○     A",
            "│",
            "│ ○   B",
            "│ │",
            "│ │ ○ C",
            "│₀│₁│₂",
            "└─┴─○ D",
        ],
    )


def test_pure_crossing():
    nodes = ["A", "B", "C", "D", "E"]
    parents = [[], [], ["A"], ["A"], ["B"]]
    unnumbered = [
        "○     A",
        "│",
        "│ ○   B",
        "│ │",
        "├─┼─○ C",
        "│ │",
        "○ │   D",
        "  │",
        "  ○   E",
    ]
    assert_numbering_modes(
        nodes,
        parents,
        off=unnumbered,
        auto=unnumbered,
        on=[
            "○     A",
            "│",
            "│ ○   B",
            "│₀│",
            "├─┼─○ C",
            "│₀│",
            "○ │   D",
            "  │₀",
            "  ○   E",
        ],
    )


def test_dense_fan_in_mixes_bends_crossings_and_direct_parents():
    nodes = ["A", "B", "C", "D", "E", "F"]
    parents = [[], [], [], ["A", "B"], ["A", "B", "C", "D"], ["B", "D", "E"]]
    unnumbered = [
        "○         A",
        "│",
        "│ ○       B",
        "│ │",
        "│ │ ○     C",
        "│ ├┐│",
        "├─┼┴┼─○   D",
        "│ ├┐│ ├┐",
        "└─┼┴┴─┼┴○ E",
        "  │   │ │",
        "  └───┴─○ F",
    ]
    assert_numbering_modes(
        nodes,
        parents,
        off=unnumbered,
        auto=unnumbered,
        on=[
            "○         A",
            "│",
            "│ ○       B",
            "│ │",
            "│ │ ○     C",
            "│₀│₁│",
            "│ ├┐│",
            "├─┼┴┼─○   D",
            "│₀│₁│₂│₃",
            "│ ├┐│ ├┐",
            "└─┼┴┴─┼┴○ E",
            "  │₀  │₁│₂",
            "  └───┴─○ F",
        ],
    )


def test_bends_work_on_consecutive_complicated_rows():
    nodes = ["A", "B", "C", "D", "E", "F", "G"]
    parents = [
        [],
        ["A"],
        ["A", "B"],
        ["A", "B", "C"],
        ["B", "C", "D"],
        ["C", "D", "E"],
        ["D", "E", "F"],
    ]
    unnumbered = [
        "○           A",
        "│",
        "├─○         B",
        "│ ├┐",
        "├─┼┴○       C",
        "│ ├┐├┐",
        "└─┼┴┼┴○     D",
        "  │ ├┐├┐",
        "  └─┼┴┼┴○   E",
        "    │ ├┐├┐",
        "    └─┼┴┼┴○ F",
        "      │ │ │",
        "      └─┴─○ G",
    ]
    assert_numbering_modes(
        nodes,
        parents,
        off=unnumbered,
        auto=unnumbered,
        on=[
            "○           A",
            "│₀",
            "├─○         B",
            "│₀│₁",
            "│ ├┐",
            "├─┼┴○       C",
            "│₀│₁│₂",
            "│ ├┐├┐",
            "└─┼┴┼┴○     D",
            "  │₀│₁│₂",
            "  │ ├┐├┐",
            "  └─┼┴┼┴○   E",
            "    │₀│₁│₂",
            "    │ ├┐├┐",
            "    └─┼┴┼┴○ F",
            "      │₀│₁│₂",
            "      └─┴─○ G",
        ],
    )


def test_new_routing_never_uses_mixed_weight_crossings():
    nodes = ["A", "B", "C", "D", "E", "F", "G"]
    parents = [
        [],
        ["A"],
        ["A", "B"],
        ["A", "B", "C"],
        ["B", "C", "D"],
        ["C", "D", "E"],
        ["D", "E", "F"],
    ]
    for mode in ("off", "auto", "on"):
        assert "╄" not in "\n".join(dag_lines(nodes, parents, number_parents=mode))


def test_parent_numbering_modes_without_bends():
    nodes = ["A", "B", "C", "D"]
    parents = [[], ["A"], ["A"], ["B", "C"]]
    assert_numbering_modes(
        nodes,
        parents,
        off=[
            "○   A",
            "│",
            "├─○ B",
            "│ │",
            "○ │ C",
            "│ │",
            "└─○ D",
        ],
        auto=[
            "○   A",
            "│",
            "├─○ B",
            "│ │",
            "○ │ C",
            "│₁│₀",
            "└─○ D",
        ],
        on=[
            "○   A",
            "│₀",
            "├─○ B",
            "│₀│",
            "○ │ C",
            "│₁│₀",
            "└─○ D",
        ],
    )


def test_parent_numbering_modes_with_one_bend():
    nodes = ["A", "B", "C", "D"]
    parents = [[], ["A"], ["B", "A"], ["A", "B", "C"]]
    assert_numbering_modes(
        nodes,
        parents,
        off=[
            "○     A",
            "│",
            "├─○   B",
            "│ ├┐",
            "├─┼┴○ C",
            "│ │ │",
            "└─┴─○ D",
        ],
        auto=[
            "○     A",
            "│",
            "├─○   B",
            "│₁│₀",
            "│ ├┐",
            "├─┼┴○ C",
            "│ │ │",
            "└─┴─○ D",
        ],
        on=[
            "○     A",
            "│₀",
            "├─○   B",
            "│₁│₀",
            "│ ├┐",
            "├─┼┴○ C",
            "│₀│₁│₂",
            "└─┴─○ D",
        ],
    )


def test_parent_numbering_modes_with_several_bends():
    nodes = ["A", "B", "C", "D", "E", "F"]
    parents = [[], [], [], ["C", "A", "B"], ["B"], ["C"]]
    unnumbered = [
        "○       A",
        "│",
        "│ ○     B",
        "│ │",
        "│ │ ○   C",
        "│ ├┐├┐",
        "└─┼┴┼┴○ D",
        "  │ │",
        "  ○ │   E",
        "    │",
        "    ○   F",
    ]
    assert_numbering_modes(
        nodes,
        parents,
        off=unnumbered,
        auto=[
            "○       A",
            "│",
            "│ ○     B",
            "│ │",
            "│ │ ○   C",
            "│₁│₂│₀",
            "│ ├┐├┐",
            "└─┼┴┼┴○ D",
            "  │ │",
            "  ○ │   E",
            "    │",
            "    ○   F",
        ],
        on=[
            "○       A",
            "│",
            "│ ○     B",
            "│ │",
            "│ │ ○   C",
            "│₁│₂│₀",
            "│ ├┐├┐",
            "└─┼┴┼┴○ D",
            "  │₀│",
            "  ○ │   E",
            "    │₀",
            "    ○   F",
        ],
    )


def test_labels_and_stdout(capsys):
    lines = dag_lines([1, 2, 3], [[], [1], [1]], label_fn="node {}".format)
    assert lines[-1] == "○   node 3"

    render_dag(["A", "B"], [[], ["A"]])
    assert capsys.readouterr().out == "○ A\n│\n○ B\n"
    render_dag([], [])
    assert capsys.readouterr().out == ""


def test_invalid_graphs():
    with pytest.raises(ValueError, match="number_parents must be one of"):
        dag_lines(["A"], [[]], number_parents=True)
    with pytest.raises(ValueError, match=r"len\(parents\)"):
        dag_lines(["A"], [])
    with pytest.raises(ValueError, match="duplicate node"):
        dag_lines(["A", "A"], [[], []])
    with pytest.raises(ValueError, match="unknown parent"):
        dag_lines(["A"], [["missing"]])
    with pytest.raises(ValueError, match="does not precede its child"):
        dag_lines(["A", "B"], [["B"], []])


def test_repeated_parents_use_low_asterisk_annotations():
    nodes = ["A", "B"]
    duplicate_parent = [[], ["A", "A"]]
    assert_numbering_modes(
        nodes,
        duplicate_parent,
        off=["○ A", "│", "○ B"],
        auto=["○ A", "│⁎", "○ B"],
        on=["○ A", "│⁎", "○ B"],
    )


def test_repeated_parent_annotations_mix_with_position_numbers():
    nodes = ["A", "B", "C"]
    parents = [[], [], ["A", "B", "A"]]
    assert_numbering_modes(
        nodes,
        parents,
        off=["○   A", "│", "│ ○ B", "│ │", "└─○ C"],
        auto=["○   A", "│", "│ ○ B", "│⁎│₁", "└─○ C"],
        on=["○   A", "│", "│ ○ B", "│⁎│₁", "└─○ C"],
    )

    parents = [[], [], ["B", "A", "B"]]
    assert_numbering_modes(
        nodes,
        parents,
        off=["○   A", "│", "│ ○ B", "│ │", "└─○ C"],
        auto=["○   A", "│", "│ ○ B", "│₁│⁎", "└─○ C"],
        on=["○   A", "│", "│ ○ B", "│₁│⁎", "└─○ C"],
    )


def test_repeated_parent_annotation_gets_a_separate_row_before_a_bend():
    nodes = ["A", "B", "C", "D"]
    parents = [[], [], ["A", "B", "B"], ["B"]]
    assert_numbering_modes(
        nodes,
        parents,
        off=["○     A", "│", "│ ○   B", "│ ├┐", "└─┼┴○ C", "  │", "  ○   D"],
        auto=["○     A", "│", "│ ○   B", "│₀│⁎", "│ ├┐", "└─┼┴○ C", "  │", "  ○   D"],
        on=["○     A", "│", "│ ○   B", "│₀│⁎", "│ ├┐", "└─┼┴○ C", "  │₀", "  ○   D"],
    )


def test_numbering_restrictions_apply_only_when_numbered():

    nodes = list(range(12))
    eleven_parents = [[] for _ in range(11)] + [list(range(11))]
    with pytest.raises(ValueError, match="numbering allows at most 10"):
        dag_lines(nodes, eleven_parents, number_parents="on")
    assert len(dag_lines(nodes, eleven_parents)) == 23
