from pangolin.util import plot_dag

def test_single_node():
    nodes = ["A"]
    parents = [[]]
    expected = "○ A"
    assert plot_dag(nodes, parents) == expected

def test_linear_chain():
    nodes = ["A", "B", "C"]
    parents = [[], ["A"], ["B"]]