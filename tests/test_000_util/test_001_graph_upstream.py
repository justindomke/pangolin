from pangolin.util import render_dag, dag_lines


def test_single_node():
    nodes = ["A"]
    parents = [[]]
    expected = ["○ A"]
    assert dag_lines(nodes, parents) == expected


# def test_linear_chain():
#     nodes = ["A", "B", "C"]
#     parents = [[], ["A"], ["B"]]
