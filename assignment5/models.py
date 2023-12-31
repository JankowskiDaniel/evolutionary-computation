from dataclasses import dataclass
from enum import Enum

"""Tutaj mamy wszystkie datamodele potrzebne do rozwiązania problemu.
    Wszystkie klasy są immutable, czyli po utworzeniu obiektu nie można go zmienić.
    Podstawowym typem jest klasa Node, później Edge, który zbudowany jest z dwóch Nodów:
    source i destination. Klasa Solution jest zbudowana z listy Node'ów, jej property edges
    od razu przy zadeklarowaniu obiektu Solution deklaruje dodatkowy parametr edges, i to jest
    list[Edges]. Klasa Move jest zbudowana z dwóch klas: RemovedEdges i AddedEdges, które są
    zbiorem Edges. Wszystkie klasy mają zaimplementowane __str__ i __repr__ metody, żeby można
    było je ładnie wyświetlać w konsoli. W klasie Move jest też zaimplementowane __lt__ żeby
    można było użyć Move jako elementu w heapq, który jest używany w algorytmie (nie jest to ważne,
    musi to być tylko po to żeby heapq nie krzyczał, że nie może porównać elementów, gdy delta jest
    sama dla obu).
"""


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
class ReversedEdges:
    edges: tuple[Edge]

    def __eq__(self, other):
        if isinstance(other, ReversedEdges):
            return frozenset(self.edges) == frozenset(other.edges)
        return False

    def __hash__(self):
        return hash(frozenset(self.edges))
    
    def invert_edge(self, edge_to_invert):
        return Edge(src=edge_to_invert.dst, dst=edge_to_invert.src)



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
    reversed_edges: ReversedEdges
    type: MoveType

    def __eq__(self, other):
        if isinstance(other, Move):
            return (self.removed_edges == other.removed_edges and 
                    self.added_edges == other.added_edges and
                    self.reversed_edges == other.reversed_edges and 
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
        reversed_edges_str = ' , '.join(f"{edge.src.index}->{edge.dst.index}" for edge in self.reversed_edges.edges)

        return f"Move:\n  Type: {self.type}\n   Removed Edges: {removed_edges_str}\n  Added Edges: {added_edges_str}\n  Reversed Edges {reversed_edges_str}"
