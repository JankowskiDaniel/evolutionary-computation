from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Node:
    index: int
    cost: int


@dataclass(frozen=True)
class Edge:
    """Edge representation. src is a source node,
    dst is a destination node.
    """
    src: Node
    dst: Node


@dataclass(frozen=True)
class Solution:
    nodes: list[Node]
    
    @property
    def edges(self) -> tuple[Edge]:
        if not self.nodes:
            return ()
        return (Edge(self.nodes[i], self.nodes[(i + 1) % len(self.nodes)]) for i in range(len(self.nodes)))
    
    def __str__(self):
        node_indices = ' -> '.join(str(node.index) for node in self.nodes)
        return f"Solution: {node_indices}"


@dataclass(frozen=True)
class RemovedEdges:
    edges: tuple[Edge]

    def __eq__(self, other):
        if isinstance(other, RemovedEdges):
            return frozenset(self.edges) == frozenset(other.edges)
        return False

    def __hash__(self):
        return hash(frozenset(self.edges))


@dataclass(frozen=True)
class AddedEdges:
    edges: tuple[Edge]

    def __eq__(self, other):
        if isinstance(other, AddedEdges):
            return frozenset(self.edges) == frozenset(other.edges)
        return False

    def __hash__(self):
        return hash(frozenset(self.edges))
    

class MoveType(Enum):
    inter_route_exchange = "inter-route-exchange"
    intra_two_edges_exchange = "intra-two-edges-exchange"
    intra_two_nodes_exchange = "intra-two-nodes-exchange"


@dataclass(frozen=True)
class Move:
    removed_edges: RemovedEdges
    added_edges: AddedEdges
    type: MoveType

    def __eq__(self, other):
        if isinstance(other, Move):
            return (self.removed_edges == other.removed_edges and 
                    self.added_edges == other.added_edges and
                    self.type == other.type)
        return False

    def __hash__(self):
        return hash((self.removed_edges, self.added_edges, self.type))
    
    def __lt__(self, other):
        # here might be whatever, it is only used to make Move comparable
        # when delta score is the same between during inserting a Move to a heap
        return self.removed_edges.edges[0].src.index < other.removed_edges.edges[0].src.index
    
    def __str__(self):
        removed_edges_str = ' , '.join(f"{edge.src.index}->{edge.dst.index}" for edge in self.removed_edges.edges)
        added_edges_str = ' , '.join(f"{edge.src.index}->{edge.dst.index}" for edge in self.added_edges.edges)

        return f"Move:\n  Type: {self.type}\n   Removed Edges: {removed_edges_str}\n  Added Edges: {added_edges_str}"
