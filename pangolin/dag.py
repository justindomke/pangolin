"""Generic nodes and operations for directed acyclic graphs.

This module is Pangolin's foundational graph API. It is intentionally usable by
interface, compiler, and inference authors as well as by Pangolin's own IR.
Functions with ``_flat`` suffixes accept flat sequences; their counterparts
accept JAX PyTrees where a structured collection of starting nodes is useful.
"""

from __future__ import annotations

from functools import reduce
from typing import Callable, Mapping, Self, Sequence

import jax.tree_util

from pangolin.util import PyTree

__all__ = [
    "Node",
    "upstream_nodes_flat",
    "upstream_nodes",
    "graph_scan",
    "graph_fold",
]


class Node:
    """A node in a directed acyclic graph.

    A node stores parents from the same node family but imposes no additional semantics on them.
    A subclass may use parents of its own class or a subclass, but not parents from a less specific
    node class.

    Parameters
    ----------
    *parents
        The parents of this node.

    Examples
    --------
    >>> root = Node()
    >>> child = Node(root)
    >>> child.parents == (root,)
    True
    """

    def __init__(self, *parents: Self):
        for parent_number, parent in enumerate(parents):
            if not isinstance(parent, type(self)):
                raise TypeError(
                    f"parent {parent_number} must be {type(self).__name__}; "
                    f"got {type(parent).__name__}"
                )
        self._parents: tuple[Self, ...] = parents

    @property
    def parents(self) -> tuple[Self, ...]:
        """The parents of this node."""
        return self._parents


def _extend_upstream_nodes[N: Node](
    nodes_flat: Sequence[N],
    node_block: Callable[[N], bool] | None,
    edge_block: Callable[[N, N], bool] | None,
    upstream: list[N],
) -> None:
    # someday should have dual set / list for faster checking
    for node in nodes_flat:
        if node in upstream or (node_block and node_block(node)):
            continue
        for p in node.parents:
            if not (edge_block and edge_block(node, p)) and not (node_block and node_block(p)):
                _extend_upstream_nodes([p], node_block, edge_block, upstream)
        upstream.append(node)


def upstream_nodes_flat[N: Node](
    nodes: Sequence[N],
    node_block: Callable[[N], bool] | None = None,
    edge_block: Callable[[N, N], bool] | None = None,
) -> list[N]:
    """Return ``nodes`` and their ancestors in topological order.

    Unlike :func:`upstream_nodes`, this function requires a flat sequence of
    starting nodes. A node for which ``node_block`` returns true is neither
    included nor traversed. An edge for which ``edge_block(child, parent)``
    returns true is not followed.

    Parameters
    ----------
    nodes
        A flat sequence of starting nodes.
    node_block
        A predicate identifying nodes to exclude and not traverse.
    edge_block
        A predicate identifying edges not to follow. It receives the child
        first and parent second.

    Examples
    --------
    >>> root = Node()
    >>> middle = Node(root)
    >>> leaf = Node(middle)
    >>> upstream_nodes_flat([leaf]) == [root, middle, leaf]
    True
    """
    upstream = []
    _extend_upstream_nodes(nodes, node_block, edge_block, upstream)
    return upstream


def upstream_nodes[N: Node](
    nodes: PyTree[N],
    node_block: Callable[[N], bool] | None = None,
    edge_block: Callable[[N, N], bool] | None = None,
) -> list[N]:
    """Return ``nodes`` and their ancestors in topological order.

    ``nodes`` may be a single node or a PyTree whose leaves are nodes. A node
    for which ``node_block`` returns true is neither included nor traversed. An
    edge for which ``edge_block(child, parent)`` returns true is not followed.

    Parameters
    ----------
    nodes
        A node or PyTree of starting nodes.
    node_block
        A predicate identifying nodes to exclude and not traverse.
    edge_block
        A predicate identifying edges not to follow. It receives the child
        first and parent second.

    Examples
    --------
    PyTree structure is ignored while finding ancestors:

    >>> root = Node()
    >>> left = Node(root)
    >>> right = Node(root)
    >>> output = Node(left, right)
    >>> starts = {"intermediate": left, "output": [output]}
    >>> upstream_nodes(starts) == [root, left, right, output]
    True

    """

    nodes_flat: list[N]
    nodes_flat, _ = jax.tree_util.tree_flatten(nodes)
    return upstream_nodes_flat(nodes_flat, node_block, edge_block)


def graph_scan[N: Node, Carry, Emission](
    nodes: PyTree[N],
    step: Callable[[N, tuple[Carry, ...]], tuple[Carry, Emission]],
    *,
    boundary: Mapping[N, Carry] | None = None,
) -> tuple[PyTree[Carry], dict[N, Emission]]:
    """Scan an upstream graph by propagating carries and collecting emissions.

    Here, a *linear scan* means a state-passing operation whose step has the form
    ``step(item, carry) -> (carry, emission)``. Items are processed from left to right: each new
    carry is passed to the next item, while emissions are collected separately. A graph has no
    unique predecessor or final position, so :func:`graph_scan` makes the following
    generalizations:

    .. list-table:: Linear and graph scans
       :header-rows: 1
       :widths: 24 36 40

       * - Concept
         - Linear scan
         - :func:`graph_scan`
       * - Work item
         - The next item in a sequence
         - The next node in a topological traversal
       * - Inputs to ``step``
         - The item and one preceding carry
         - The node and one carry per parent, in parent order
       * - External carries
         - One carry before the first item
         - An optional ``boundary`` mapping supplying carries at a graph cut
       * - Use of a new carry
         - Passed to the next item
         - Passed to every child of the node
       * - Emissions
         - One per item, collected in sequence order
         - One per visited node, collected in a node-keyed mapping
       * - Terminal carries
         - The carry after the final item
         - Carries for the requested ``nodes``, preserving their PyTree structure

    By default, traversal follows every parent edge upstream from ``nodes`` until it reaches nodes
    with no parents. ``step`` is called for those roots with an empty tuple of parent carries.

    ``boundary`` defines an optional cut through the graph. When traversal reaches a boundary
    node, it uses the supplied carry without calling ``step`` for that node or continuing through
    its parents. Boundary nodes therefore produce no emissions. A boundary can contain several
    nodes because a graph cut can expose several independently supplied carries. An ancestor
    hidden behind one boundary node is still visited if another unblocked path reaches it.

    Each non-boundary node is visited exactly once in topological order. ``step`` receives the node
    and the carries of its parents in parent order, and returns a carry for the node together with
    an emission. The carry is available to the node's children. The emission is retained in the
    returned mapping but is not propagated through the graph.

    The returned carries correspond only to the requested ``nodes`` and preserve their PyTree
    structure, just as the final carry is the primary result of a linear scan. Emissions may come
    from any visited node, including intermediate and shared nodes, so they are returned as a
    node-keyed mapping rather than forced into the structure of ``nodes``.

    Parameters
    ----------
    nodes
        A node or PyTree of nodes whose carries should be returned.
    step
        Compute a node's carry and emission from the node and its parent carries.
    boundary
        Carries defining an optional graph cut. A boundary node is not visited, does not emit, and
        prevents traversal through its parents along that path. With no boundary, the complete
        upstream graph is visited through its parentless roots.

    Returns
    -------
    output_carries
        Carries corresponding to ``nodes`` with the same runtime PyTree structure.
    emissions
        A mapping from every visited node to its emission, in topological insertion order.

    Examples
    --------
    Carries flow through edges while emissions record each visited node:

    >>> root = Node()
    >>> leaf = Node(root)
    >>> def step(node, parent_carries):
    ...     carry = 1 + sum(parent_carries)
    ...     return carry, len(node.parents)
    >>> carries, emissions = graph_scan({"leaf": leaf}, step)
    >>> carries
    {'leaf': 2}
    >>> emissions == {root: 0, leaf: 1}
    True

    A boundary supplies a carry without producing an emission:

    >>> carries, emissions = graph_scan(leaf, step, boundary={root: 10})
    >>> carries
    11
    >>> emissions == {leaf: 1}
    True
    """
    carries = dict(boundary or {})
    emissions = {}

    scanned_nodes = upstream_nodes(nodes, node_block=carries.__contains__)
    for node in scanned_nodes:
        parent_carries = tuple(carries[parent] for parent in node.parents)
        carries[node], emissions[node] = step(node, parent_carries)

    flat_nodes, treedef = jax.tree_util.tree_flatten(nodes)
    flat_carries = [carries[node] for node in flat_nodes]
    output_carries = jax.tree_util.tree_unflatten(treedef, flat_carries)
    return output_carries, emissions


def graph_fold[N: Node, Carry, Emission](
    nodes: PyTree[N],
    step: Callable[[N, tuple[Carry, ...]], tuple[Carry, Emission]],
    combine: Callable[[Emission, Emission], Emission] = lambda a, b: a + b,  # type: ignore[operator]
    *,
    boundary: Mapping[N, Carry] | None = None,
) -> Emission:
    """Scan an upstream graph and combine all emissions into one value.

    This operation is exactly a :func:`graph_scan` followed by a reduction of its emissions.
    Carries still propagate through the graph so that ``step`` can compute later nodes, but the
    requested carries and the individual emissions are discarded after the emissions are
    combined. Thus, an algorithm can use carries for local graph state while using emissions for
    the global quantity it ultimately wants, such as a joint log probability.

    In this API, *fold* therefore means folding the emissions produced by a scan. It does not mean
    that one value must serve as both the propagated carry and the collected emission.

    .. list-table:: Linear and graph folds
       :header-rows: 1
       :widths: 24 36 40

       * - Concept
         - Fold of linear-scan emissions
         - :func:`graph_fold`
       * - Values combined
         - One emission per sequence item
         - One emission per visited graph node
       * - Available order
         - The sequence's total order
         - A topological partial order; unrelated nodes have no intrinsic order
       * - Required operator laws
         - None for a specified left fold
         - Associativity and commutativity for a traversal-independent result
       * - Starting value
         - The first emission
         - The first emission
       * - Result
         - One combined emission
         - One combined emission

    Unlike a sequence, a DAG does not impose a unique order on unrelated nodes. If ``combine`` is
    associative and commutative, every valid topological traversal gives the same result.
    :func:`graph_fold` does not and cannot enforce those algebraic laws: with any other operator,
    the result may depend on the incidental ordering chosen by the upstream traversal. The first
    emission starts the reduction; because no identity is required, a graph with no emissions
    cannot be folded.

    By default, emissions are added.

    Parameters
    ----------
    nodes
        A node or PyTree of nodes defining the upstream graph to scan.
    step
        Compute a node's carry and emission from the node and its parent carries.
    combine
        Combine two emissions. The default adds them. Associativity and commutativity are not
        enforced, but are required for a result independent of upstream traversal order.
    boundary
        Carries defining an optional graph cut, with the same semantics as in
        :func:`graph_scan`. Boundary nodes and ancestors reachable only through them contribute no
        emissions.

    Returns
    -------
    folded_emission
        All visited-node emissions combined into one value.

    Raises
    ------
    ValueError
        If no nodes are visited and therefore there are no emissions to combine.

    Examples
    --------
    The default operator adds numeric emissions:

    >>> root = Node()
    >>> leaf = Node(root)
    >>> def step(node, parent_carries):
    ...     carry = 1 + sum(parent_carries)
    ...     return carry, carry
    >>> graph_fold(leaf, step)
    3

    A custom associative and commutative operator may be supplied:

    >>> graph_fold(leaf, step, lambda left, right: left * right)
    2
    """
    _, emissions = graph_scan(nodes, step, boundary=boundary)
    if not emissions:
        raise ValueError("Can't fold a graph with no emissions")
    return reduce(combine, emissions.values())


def _get_children[N: Node](
    nodes: PyTree[N], node_block: Callable[[N], bool] | None = None
) -> dict[N, list[N]]:
    """Return the children of each node upstream of ``nodes``.

    Parameters
    ----------
    nodes
        A node or PyTree of starting nodes, as in :func:`upstream_nodes`.
    node_block
        A predicate identifying nodes to exclude and not traverse.
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


def _has_second_path(node: Node, par_i: int) -> bool:
    assert par_i < len(node.parents)
    other_parents = node.parents[:par_i] + node.parents[par_i + 1 :]
    return node.parents[par_i] in upstream_nodes(other_parents)
