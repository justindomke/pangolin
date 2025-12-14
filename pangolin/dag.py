"""
Basic code for operating on directed acyclic graphs (DAGs).
None of these functions import or use any other parts of Pangolin.
**End-users of Pangolin are not expected to use these functions directly**.
"""

from __future__ import annotations
import jax.tree_util
from typing import Sequence, Callable, Optional
from jaxtyping import PyTree


class Node:
    """
    The basic `Node` class. This is just an object that remembers a set of parents.

    Args:
        *parents: parents of this node
    """

    def __init__(self, *parents: "Node"):
        self._parents: tuple[Node] = parents

    @property
    def parents(self) -> tuple["Node"]:
        "The parents of this node"
        return self._parents


def upstream_nodes_flat(
    nodes_flat: Sequence[Node],
    node_block: Callable[[Node], bool],
    edge_block: Callable[[Node, Node], bool],
    upstream: list[Node],
):
    """
    Do a DFS starting at all the nodes in `nodes_flat`. But never visit nodes if
    `node_block(n)` and never follow an edge from `n` to `p` if `link_block(n,p)`.

    Args:
        nodes_flat: starting nodes
        node_block: should DFS be blocked from visting a node?
        edge_block: should DFS be blocked from following a link?
        upstream: list of nodes, destructively updated
    """
    # someday should have dual set / list for faster checking
    for node in nodes_flat:
        if node in upstream or node_block(node):
            continue
        for p in node.parents:
            if not edge_block(node, p) and not node_block(p):
                upstream_nodes_flat([p], node_block, edge_block, upstream)
        upstream.append(node)


def _never_block(x: Node) -> bool:
    return False


def _never_edge_block(x: Node, y: Node) -> bool:
    return False


def upstream_nodes(
    nodes: PyTree[Node],
    node_block: Callable[[Node], bool] | None = None,
    edge_block: Callable[[Node, Node], bool] | None = None,
) -> list[Node]:
    """
    Do a DFS of all ancestors starting at all the nodes in `nodes_flat`.

    Args:
        nodes
            Single node or list/pytree of node
        node_block
            Function defining which nodes to block from inclusion. If `node_block(node)`
            is True, then `node` is not visited. (Defalt: Don't block.)
        edge_block
            Function defining which edges should not be followed.  If
            `edge_block(node, parent)` is true, then don't follow edge from `node` to
            `parent`. (Defalt: Don't block)

    Returns:
        List of upstream nodes in topological order
    """

    # Args:
    #     nodes (`PyTree[Node]`): single node or list of nodes or pytree of starting nodes
    #     node_block: should DFS be blocked from visting a node? If `None`, then all nodes
    #     edge_block: should DFS be blocked from following an edge? If `None`, then all

    # Returns:
    #     List of all nodes found, with a partial order so that parents always come
    #     before children

    if node_block is None:
        node_block = _never_block

    if edge_block is None:
        edge_block = _never_edge_block

    nodes_flat, _ = jax.tree_util.tree_flatten(nodes)
    upstream = []
    upstream_nodes_flat(nodes_flat, node_block, edge_block, upstream)
    return upstream


upstream_nodes({"a": Node()})


def upstream_with_descendent_old(
    requested_nodes: list[Node], given_nodes: list[Node]
) -> list[Node]:
    """
    First, find all nodes that are upstream (inclusive) of `requested_nodes`
    Then, find all the nodes that are *downstream* (inclusive) of that set
    that have a descendant in `given_nodes`

    args:
        requested_nodes: nodes to search above
        given_nodes: must have a descendant in this set
    """

    has_descendent = upstream_nodes(given_nodes)
    children = get_children(requested_nodes + given_nodes)

    nodes = []
    processed_nodes = []
    queue = requested_nodes.copy()

    def unseen(node):
        return node not in queue and node not in processed_nodes

    while queue:
        # print(f"{queue=}")
        node = queue.pop()
        # if node.cond_dist.is_random and node in has_descendent:
        if node in has_descendent:
            nodes.append(node)
        processed_nodes.append(node)

        for p in node.parents:
            if unseen(p) and p not in given_nodes:
                queue.append(p)
        for c in children[node]:
            if unseen(c) and c in has_descendent:
                queue.append(c)
    return nodes


def upstream_with_descendent(
    requested_nodes: list[Node], given_nodes: list[Node]
) -> list[Node]:
    """
    First, find all nodes that are upstream (inclusive) of `requested_nodes`
    Then, find all the nodes that are *downstream* (inclusive) of that set
    that have a descendant in `given_nodes`
    """

    all_nodes = upstream_nodes(requested_nodes + given_nodes)
    has_descendent = upstream_nodes(given_nodes)
    return [n for n in all_nodes if n in has_descendent]


def get_children(
    nodes: PyTree[Node], node_block: Callable[[Node], bool] | None = None
) -> dict[Node, list[Node]]:
    """
    Get all children for all upstream nodes

    Args:
        nodes: Starting nodes (as in `get_upstream`)
        node_block: Block condition (as in `get_upstream`)

    Return:
        Dictionary mapping each upstream node to a list of children of that node

    """

    all_nodes = upstream_nodes(nodes, node_block)
    children = {}
    for n in all_nodes:
        children[n] = []
    for n in all_nodes:
        for p in n.parents:
            # print(f"{hash(n)=} {hash(p)=}")
            if n not in children[p]:
                children[p].append(n)
    return children


def is_in(node: Node, nodes: Sequence[Node]) -> bool:
    for p in nodes:
        # if id(node) == id(p) or node.id == p.id:
        if id(node) == id(p):
            return True
    return False


def has_second_path(node: Node, par_i: int) -> bool:
    assert par_i < len(node.parents)
    other_parents = node.parents[:par_i] + node.parents[par_i + 1 :]
    return node.parents[par_i] in upstream_nodes(other_parents)


# def downstream_nodes(roots,nodes,always_include=[]):
#     downstream = []
#     for node in nodes:
#         if is_in(node,roots):
#             continue
#         if is_in(node,always_include):
#             downstream.append(node)
#             continue
#         for p in node.parents:
#             if is_in(p,downstream+roots):
#                 downstream.append(node)
#                 break
#     return downstream
#
# def middle_nodes(roots,leafs):
#     "all nodes that are both upstream of a leaf and downstream of a root"
#     # TODO: this goes through the whole graph and then filters, could be inefficient but whatever
#     nodes = upstream_nodes(leafs)
#     middle = {}
#     for node in nodes:
#         if is_in(node,leafs):
#             continue
#         for p in node.parents:
#             if p in middle or is_in(p,roots):
#                 middle[node] = None # unordered set
#     return list(middle.keys())


def get_graph(nodes: PyTree[Node], get_content, block_condition=None):
    upstream = upstream_nodes(nodes, block_condition)
    return dict(zip(upstream, [get_content(n) for n in upstream]))
