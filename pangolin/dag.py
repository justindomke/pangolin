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


# def dfs(node, upstream, block_condition):
#     # if node in upstream:
#     if is_in(node, upstream):
#         return
#
#     if (block_condition is not None) and block_condition(node):
#         return
#
#     if isinstance(node, Node):
#         for p in node.parents:
#             dfs(p, upstream,block_condition)
#
#     # upstream.add(node)
#     upstream[node] = None  # use dict as ordered set
#
#
# def upstream_nodes(nodes, block_condition=None):
#     # something wrong with this assert
#     # if any([b in nodes for b in blockers]):
#     #     raise Exception("same node can't be provided as start and blocker")
#
#     # transform into a list if needed
#     if isinstance(nodes, Node):
#         return upstream_nodes([nodes], block_condition)
#
#     #assert util.all_unique(nodes), "all inputs must be unique"
#
#     # upstream = set()
#     upstream = {}
#     for node in nodes:
#         dfs(node, upstream, block_condition)
#
#     assert len(upstream.keys()) == len(upstream) # no duplicates added
#     return upstream.keys()

# def dfs(node, upstream, block_condition):
#     if node in upstream:
#         return
#
#     if block_condition and block_condition(node):
#         return
#
#     #visited.add(node)
#     for p in node.parents:
#         dfs(p, upstream, block_condition)
#     upstream.append(node)
#
# def upstream_nodes(nodes, block_condition=None):
#     import jax.tree_util
#     nodes_flat, nodes_treedef = jax.tree_util.tree_flatten(nodes)
#
#     #visited = {} # set of all nodes visited by DFS (not necessarily in order)
#     upstream = [] # set of upstream nodes *in order*
#
#     for node in nodes_flat:
#         dfs(node, upstream, block_condition)
#
#     return upstream


def upstream_nodes(nodes, block_condition=None, link_block_condition=None):
    """
    Given some set of nodes, stored in an arbitrary pytree, get all nodes that are
    upstream (in the form of a flat list)
    """

    nodes_flat, nodes_treedef = jax.tree_util.tree_flatten(nodes)
    if block_condition is None:
        block_condition = lambda x: False
    if link_block_condition is None:
        link_block_condition = lambda x, y: False

    return upstream_nodes_flat(nodes_flat, block_condition, link_block_condition, [])


def upstream_nodes_flat(nodes_flat, node_block, link_block, upstream):
    # someday should have dual set / list for faster checking
    for node in nodes_flat:
        # print(f"{node.cond_dist=}")
        if node in upstream or node_block(node):
            continue
        for p in node.parents:
            if not link_block(node, p):
                upstream_nodes_flat([p], node_block, link_block, upstream)
        upstream.append(node)

    return upstream


def upstream_with_descendent(requested_nodes, given_nodes):
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


def get_children(nodes, block_condition=None):
    all_nodes = upstream_nodes(nodes, block_condition)
    children = {}
    for n in all_nodes:
        children[n] = []
    for n in all_nodes:
        for p in n.parents:
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
