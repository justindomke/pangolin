"""like test_dag.py except using RVs
"""

from pangolin.dag import (
    Node,
    upstream_nodes,
    get_children,
    has_second_path,
    upstream_with_descendent,
)
from pangolin import *


def same(l1, l2):
    assert {a for a in l1} == {a for a in l2}  # equality of frozen sets


def test_upstream_single():
    a = RV(Constant(1))
    assert upstream_nodes([a]) == [a]


def test_upstream_empty():
    a = RV(Constant(1))
    assert upstream_nodes([]) == []


def test_upstream_pair():
    a = RV(Constant(1))
    b = RV(exponential,a)
    assert upstream_nodes([a]) == [a]
    assert upstream_nodes([b]) == [a, b]

def test_upstream_pair2():
    a = RV(Constant(1))
    b = RV(exp,a)
    assert upstream_nodes([a]) == [a]
    assert upstream_nodes([b]) == [a, b]


def test_upstream_dupe():
    a = RV(Constant(1))
    b = RV(exp,a)
    assert upstream_nodes([a, b]) == [a, b]


def test_upstream_dupe2():
    a = RV(Constant(1))
    b = RV(exp, a)
    c = RV(exp, b)
    # 000
    assert upstream_nodes([]) == []
    # 001
    assert upstream_nodes([c]) == [a, b, c]
    # 010
    assert upstream_nodes([b]) == [a, b]
    # 011
    assert upstream_nodes([b, c]) == [a, b, c]
    assert upstream_nodes([c, b]) == [a, b, c]
    # 100
    assert upstream_nodes([a]) == [a]
    # 101
    assert upstream_nodes([a, c]) == [a, b, c]
    # 110
    assert upstream_nodes([a, b]) == [a, b]
    assert upstream_nodes([b, a]) == [a, b]
    # 111
    assert upstream_nodes([a, b, c]) == [a, b, c]
    assert upstream_nodes([a, c, b]) == [a, b, c]
    assert upstream_nodes([b, a, c]) == [a, b, c]
    assert upstream_nodes([b, c, a]) == [a, b, c]
    assert upstream_nodes([c, a, b]) == [a, b, c]
    assert upstream_nodes([c, b, a]) == [a, b, c]


def test_upstream_collider():
    a = RV(Constant(1))
    b = RV(exponential,a)
    c = RV(exponential,a)
    d = RV(add,b, c)
    assert upstream_nodes([d]) == [a, b, c, d]
    assert upstream_nodes([d, b]) == [a, b, c, d]
    assert upstream_nodes([b, d]) == [a, b, c, d]
    assert upstream_nodes([b, d, a]) == [a, b, c, d]
    assert upstream_nodes([b, d, a, d, b, a]) == [a, b, c, d]

def test_upstream_collider2():
    a = RV(Constant(1))
    b = RV(exp,a)
    c = RV(exp,a)
    d = RV(add,b, c)
    assert upstream_nodes([d]) == [a, b, d]
    assert upstream_nodes([d, b]) == [a, b, d]
    assert upstream_nodes([b, d]) == [a, b, d]
    assert upstream_nodes([b, d, a]) == [a, b, d]
    assert upstream_nodes([b, d, a, d, b, a]) == [a, b, d]

    assert upstream_nodes([d]) == [a, c, d]
    assert upstream_nodes([d, b]) == [a, c, d]
    assert upstream_nodes([b, d]) == [a, c, d]
    assert upstream_nodes([b, d, a]) == [a, c, d]
    assert upstream_nodes([b, d, a, d, b, a]) == [a, c, d]



def test_upstream_blockers():
    a = RV(Constant(1))
    b = RV(exponential,a)
    c = RV(exponential,b)
    d = RV(exponential,c)
    assert upstream_nodes(d, lambda n: n in [a]) == [b, c, d]
    assert upstream_nodes(d, lambda n: n in [b]) == [c, d]
    assert upstream_nodes(d, lambda n: n in [c]) == [d]
#
#
def test_collider_blockers2():
    a = RV(Constant(1))
    b = RV(exponential,a)
    c = RV(exponential,a)
    d = RV(add, b, c)
    assert upstream_nodes(d) == [a, b, c, d]
    assert upstream_nodes(d, lambda n: n in [b]) == [a, c, d]
    assert upstream_nodes(d, lambda n: n in [c]) == [a, b, d]
    assert upstream_nodes(d, lambda n: n in [b, c]) == [d]

def test_children1():
    a = RV(Constant(1))
    b = RV(exponential,a)
    c = RV(exponential,a)
    d = RV(add, b, c)
    children = get_children([d])
    assert set(children[a]) == {b, c}
    assert set(children[b]) == {d}
    assert set(children[c]) == {d}
    assert set(children[d]) == set()

def test_children2():
    a = RV(Constant(1))
    b = RV(exp,a)
    c = RV(exp,a)
    d = RV(add, b, c)
    children = get_children([d])
    assert set(children[a]) == {b, c}
    assert set(children[a]) == {b}
    assert set(children[b]) == {d}
    assert set(children[c]) == {d}
    assert set(children[d]) == set()

def test_children3():
    z = normal(0.0, 1)
    x = normal(z, 1)
    y = normal(z, 1.0)

    #all_upstream = dag.upstream_nodes()
    upstream = dag.upstream_with_descendent([y], [x])
    #upstream = dag.upstream_with_descendent_old([y], [x])
    print(f"{upstream=}")
    print(f"{len(upstream)=}")
    #upstream_observed_vars = dag.upstream_with_descendent(requested_vars, given_vars)
    #return list(x for x in upstream_observed_vars if x.cond_dist.random)


# def test_has_second_path():
#     a = Node()
#     b = Node(a)
#     assert not has_second_path(b, 0)
#
#     a = Node()
#     b = Node(a, a)
#     assert has_second_path(b, 0)
#     assert has_second_path(b, 1)
#
#     a = Node()
#     b = Node(a)
#     c = Node(a, b)
#     assert has_second_path(c, 0)
#     assert not has_second_path(c, 1)
#
#
# def test_upstream_with_descendents1():
#     a = Node()
#     b = Node(a)
#     c = Node(a)
#     d = Node(b, c)
#     assert upstream_with_descendent([a], []) == []
#     assert upstream_with_descendent([a], [d]) in [[a, c, b, d], [a, b, c, d]]
#     # assert upstream_with_descendent([b], []) == [a, b]
#     # assert upstream_with_descendent([b], [c]) == [a]
#
