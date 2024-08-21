"""
Basic code for operating on directed acyclic graphs (DAGs).
This is all independent of the rest of pangolin.
"""

import jax.tree_util


class Node:
    """
    The basic `Node` class. This is just an object that remembers a set of
    parents, nothing more.
    """

    def __init__(self, *parents):
        self.parents = parents
        "The parents of this node"


def upstream_nodes_flat(nodes_flat, node_block, edge_block, upstream):
    """
    Do a DFS starting at all the nodes in `nodes_flat`. But never visit nodes if
    `node_block(n)` and never follow an edge from `n` to `p` if `link_block(n,p)`.

    Parameters
    ----------
    nodes_flat: Sequence[Node]
        starting nodes
    node_block: Callable
        should DFS be blocked from visting a node?
    edge_block: Callable
        should DFS be blocked from following a link
    upstream: List
        list of nodes, destructively updated

    Returns
    -------
    upstream: list[Node]
        list of all nodes found, with a partial order so that parents always come
        before children
    """
    # someday should have dual set / list for faster checking
    for node in nodes_flat:
        if node in upstream or node_block(node):
            continue
        for p in node.parents:
            if not edge_block(node, p) and not node_block(p):
                upstream_nodes_flat([p], node_block, edge_block, upstream)
        upstream.append(node)

    return upstream


def upstream_nodes(nodes, block_condition=None, edge_block=None):
    """
    Do a DFS starting at all the nodes in `nodes_flat`. But never visit nodes if
    `node_block(n)` and never follow an edge from `n` to `p` if `link_block(n,p)`.

    Parameters
    ----------
    nodes_flat: pytree[Node]
        starting nodes
    node_block: Callable
        should DFS be blocked from visting a node? If None, then all nodes allowed.
    edge_block: Callable
        should DFS be blocked from following an edge? If None, then all edges allowed.

    Returns
    -------
    upstream: list[Node]
        list of all nodes found, with a partial order so that parents always come
        before children
    """

    nodes_flat, _ = jax.tree_util.tree_flatten(nodes)
    if block_condition is None:
        block_condition = lambda x: False
    if edge_block is None:
        edge_block = lambda x, y: False

    return upstream_nodes_flat(nodes_flat, block_condition, edge_block, [])


def upstream_with_descendent_old(requested_nodes, given_nodes):
    """
    First, find all nodes that are upstream (inclusive) of `requested_nodes`
    Then, find all the nodes that are *downstream* (inclusive) of that set
    that have a descendant in `given_nodes`
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


def upstream_with_descendent(requested_nodes, given_nodes):
    """
    First, find all nodes that are upstream (inclusive) of `requested_nodes`
    Then, find all the nodes that are *downstream* (inclusive) of that set
    that have a descendant in `given_nodes`
    """

    all_nodes = upstream_nodes(requested_nodes + given_nodes)
    has_descendent = upstream_nodes(given_nodes)
    return [n for n in all_nodes if n in has_descendent]


def get_children(nodes, block_condition=None):
    all_nodes = upstream_nodes(nodes, block_condition)
    children = {}
    for n in all_nodes:
        children[n] = []
    #print(f"{all_nodes=}")
    #print(f"{children=}")
    #print(f"{[hash(n) for n in all_nodes]=}")
    #print(f"{[hash(n) for n in children]=}")
    for n in all_nodes:
        for p in n.parents:
            #print(f"{hash(n)=} {hash(p)=}")
            if n not in children[p]:
                children[p].append(n)
    return children


def is_in(node, nodes):
    for p in nodes:
        # if id(node) == id(p) or node.id == p.id:
        if id(node) == id(p):
            return True
    return False


def has_second_path(node, par_i):
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


def get_graph(nodes, get_content, block_condition=None):
    upstream = upstream_nodes(nodes, block_condition)
    return dict(zip(upstream, [get_content(n) for n in upstream]))
