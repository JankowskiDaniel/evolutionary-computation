import heapq
import random
from typing import Literal
import time
from utils import *
from models import (
    Edge,
    RemovedEdges,
    AddedEdges,
    ReversedEdges,
    Move,
    Solution,
    Node,
    MoveType
)
from collections import defaultdict
class MoveTracker:
    """Podstawowy tracker ruchów (nasz LM), który dodaje ruchy do kolejki
    i zawsze trzyma je posortowane po delcie.
    """
    def __init__(self):
        self.moves_heap = []
        self.moves_set = set()

    def add_move(self, move, score):
        """Add a move with the given score to the tracker."""
        if move not in self.moves_set:
            heapq.heappush(self.moves_heap, (score, move))
            self.moves_set.add(move)

    def move_exists(self, move):
        """Check if a move exists in the tracker."""
        return move in self.moves_set



class SteepestLocalSearch():
    def __init__(self, 
                 initial_solution: Solution, 
                 distance_matrix: list[list[int]], 
                 costs: list[int]) -> None:
        self.current_solution: Solution = initial_solution
        self.current_score = objective_function(initial_solution, distance_matrix, costs)
        self.distance_matrix = distance_matrix
        self.costs = costs
        
        self.all_nodes: list[Node] = [Node(i, costs[i]) for i in range(2*len(self.current_solution.nodes))]
        self.unselected_nodes: list[Node] = [node for node in self.all_nodes if node not in self.current_solution.nodes]
        self.tracker = MoveTracker()
        
        self.moves = {
            "intra-two-edges-exchange": self.two_edges_exchange,
            "intra-two-nodes-exchange": self.two_nodes_exchange,
            "inter-route-exchange": self.inter_route_exchange
        }
        
    # Method for check applicability of intra-moves exchange
    def is_intra_move_applicable(self, solution: Solution, move: Move) -> Literal[-1, 0, 1]:
        """The method for checking applicability of intra-moves is the same
        for both moves, since we operate on only nodes that are inside the solution

        Returns:
            -1 if the move is not applicable
            0 if the move is not applicable but might be applicable later on
            1 if the move is applicable and we take it
        """
        solution_nodes = set((node.index, node.cost) for node in solution.nodes)
        
        # Check if all nodes in added_edges are present in the solution
        for edge in move.added_edges.edges:
            if (edge.src.index, edge.src.cost) not in solution_nodes or \
            (edge.dst.index, edge.dst.cost) not in solution_nodes:
                return -1

        # Convert solution edges to a set of tuples for efficient lookup
        solution_edges_set = set((edge.src.index, edge.src.cost, edge.dst.index, edge.dst.cost) for edge in solution.edges)

        all_edges_match = True
        for edge in move.removed_edges.edges:
            edge_tuple = (edge.src.index, edge.src.cost, edge.dst.index, edge.dst.cost)
            edge_reversed_tuple = (edge.dst.index, edge.dst.cost, edge.src.index, edge.src.cost)

            if edge_tuple not in solution_edges_set and edge_reversed_tuple not in solution_edges_set:
                return -1  # Edge does not exist in the solution at all
            if edge_tuple not in solution_edges_set and edge_reversed_tuple in solution_edges_set:
                all_edges_match = False  # Edge exists but is in reversed order

        return 1 if all_edges_match else 0
    

    def is_inter_move_applicable(self, solution: Solution, move: Move) -> Literal[-1, 0, 1]:
        solution_nodes = set((node.index, node.cost) for node in solution.nodes)
        
        # in case of inter move, we need to have in the solution
        # the first node from the first added edge, and the second node
        # from the second added node (those are the original nodes from the
        # old solution) look at the method for inter exchange for more
        # details
        if (move.added_edges.edges[0].src.index, move.added_edges.edges[0].src.cost) not in solution_nodes or \
            (move.added_edges.edges[1].dst.index, move.added_edges.edges[1].dst.cost) not in solution_nodes:
            # print("First condition")
            return -1
        
        # if the node that has to be inserted is not in the unselected nodes
        # it cant be applied, we remove it
        if move.added_edges.edges[1].src not in self.unselected_nodes:
            # print("Second condition")
            return -1
        
        # the rest is the same, we check removed_edges

        # Convert solution edges to a set of tuples for efficient lookup
        solution_edges_set = set((edge.src.index, edge.src.cost, edge.dst.index, edge.dst.cost) for edge in solution.edges)

        all_edges_match = True
        # print(self.current_solution)
        # print(move)
        for edge in move.removed_edges.edges:
            edge_tuple = (edge.src.index, edge.src.cost, edge.dst.index, edge.dst.cost)
            edge_reversed_tuple = (edge.dst.index, edge.dst.cost, edge.src.index, edge.src.cost)
            
            if edge_tuple not in solution_edges_set and edge_reversed_tuple not in solution_edges_set:
                # print("Third condition")
                return -1  # Edge does not exist in the solution at all
            if edge_tuple not in solution_edges_set and edge_reversed_tuple in solution_edges_set:
                all_edges_match = False  # Edge exists but is in reversed order

        return 1 if all_edges_match else 0
        
        
    """Tutaj na pewno jest jakiś bug, bo jak odpalimy z two_edges to jest deadlock i while w order_edges
    leci w nieskończoność (nie jest w stanie znaleźć ścieżki)
    """
    def two_edges_exchange(self,
                            start_index: int = 0,
                            direction: str = "right") -> None:
        """
        Generate a new solution by exchanging two edges in the current solution,
        starting from a given index and moving in the specified direction.

        :param current_solution: List of nodes in the current solution.
        :param current_distance: The score of the current solution.
        :param distance_matrix: 2D list representing the distances between nodes.
        :param start_index: The index from which to start searching for a better solution.
        :param direction: The direction in which to search for a better solution ("right" or "left").
        :return: A tuple containing the new solution and its score if it's better,
                otherwise the original solution and score.
        """
        n = len(self.current_solution.nodes)

        if direction == "right": 
            range_i = range(n - 2)
            range_j = lambda i: range(i + 2, n)
        else:  # direction == "left"
            range_i = range(n - 3, -1, -1)
            range_j = lambda i: range(n - 1, i + 1, -1)
        count = 0
        for i in range_i:
            for j in range_j(i): 
                if count >= start_index:
                    
                    # Construct a move
                    removed_edges = RemovedEdges((Edge(self.current_solution.nodes[i], self.current_solution.nodes[i + 1]),
                                                  Edge(self.current_solution.nodes[j], self.current_solution.nodes[(j + 1) % n])))
                    added_edges = AddedEdges((Edge(self.current_solution.nodes[i], self.current_solution.nodes[j]),
                                              Edge(self.current_solution.nodes[i + 1], self.current_solution.nodes[(j + 1) % n])))
                    #dla sprawdzenia czy działa
                    # print(self.current_solution)
                    # print("removed edges:", self.current_solution.nodes[i], self.current_solution.nodes[(i+1)%n])
                    # print("removed edges:", self.current_solution.nodes[j], self.current_solution.nodes[(j+1)%n])
                    # print("added edges:", self.current_solution.nodes[i], self.current_solution.nodes[j])
                    # print("added edges:", self.current_solution.nodes[(i+1)%n], self.current_solution.nodes[(j+1)%n])
                    reversed_edges = []
                    for k in range(i+1,j):
                        # print(self.current_solution.nodes[k+1],self.current_solution.nodes[k])
                        reversed_edges.append(Edge(self.current_solution.nodes[k],self.current_solution.nodes[k+1]))
                    reversed_edges= ReversedEdges(tuple(reversed_edges))
                    move = Move(removed_edges, added_edges, reversed_edges, MoveType.intra_two_edges_exchange)
                    
                    # check if a move exists, if not add to the LM
                    if self.tracker.move_exists(move):
                        continue
                    else:
                        # we don't have this move so we calculate delta and add into the heap
                        score_delta = (
                            -self.distance_matrix[self.current_solution.nodes[i].index][self.current_solution.nodes[i + 1].index]
                            -self.distance_matrix[self.current_solution.nodes[j].index][self.current_solution.nodes[(j + 1) % n].index]
                            +self.distance_matrix[self.current_solution.nodes[i].index][self.current_solution.nodes[j].index]
                            +self.distance_matrix[self.current_solution.nodes[i + 1].index][self.current_solution.nodes[(j + 1) % n].index]
                        )
                        if score_delta < 0:
                            self.tracker.add_move(move, score_delta)
                count += 1  # Increment the counter after checking the condition
    
    def two_nodes_exchange(self, start_index=0, direction='right'):
        n = len(self.current_solution.nodes)
        total_moves = n * (n - 1) // 2  # Total number of possible swaps
        reversed_edges=ReversedEdges(tuple())

        index_pairs = [(x, y) for x in range(n) for y in range(x+1, n)]
        # Adjust the indices list based on the direction
        if direction == 'left':
            index_pairs = index_pairs[::-1]
            start_index = total_moves - start_index - 1 

        for count, (i, j) in enumerate(index_pairs[start_index:], start=start_index):

            if i == 0 and j == n - 1:  # special case: first and last nodes
                # construct a move
                removed_edges = RemovedEdges((Edge(self.current_solution.nodes[j], self.current_solution.nodes[0]),
                                              Edge(self.current_solution.nodes[j-1], self.current_solution.nodes[j]),
                                              Edge(self.current_solution.nodes[0], self.current_solution.nodes[1])))
                added_edges = AddedEdges((Edge(self.current_solution.nodes[j], self.current_solution.nodes[1]),
                                            Edge(self.current_solution.nodes[j-1], self.current_solution.nodes[0]),
                                            Edge(self.current_solution.nodes[0], self.current_solution.nodes[j])))
                move = Move(removed_edges, added_edges,reversed_edges, MoveType.intra_two_nodes_exchange)
                
                if move in self.tracker.moves_set:
                    continue
                else:  
                    # we don't have this move so we calculate delta and add into the heap
                    score_delta = (
                        -self.distance_matrix[self.current_solution.nodes[j].index][self.current_solution.nodes[0].index]
                        -self.distance_matrix[self.current_solution.nodes[j-1].index][self.current_solution.nodes[j].index]
                        -self.distance_matrix[self.current_solution.nodes[0].index][self.current_solution.nodes[1].index]
                        +self.distance_matrix[self.current_solution.nodes[j].index][self.current_solution.nodes[1].index]
                        +self.distance_matrix[self.current_solution.nodes[j-1].index][self.current_solution.nodes[0].index]
                        +self.distance_matrix[self.current_solution.nodes[0].index][self.current_solution.nodes[j].index]
                    )
                    if score_delta < 0:
                        self.tracker.add_move(move, score_delta)

            elif j == i + 1:  # adjacent nodes case
                
                removed_edges = RemovedEdges((Edge(self.current_solution.nodes[i-1], self.current_solution.nodes[i]),
                                              Edge(self.current_solution.nodes[j], self.current_solution.nodes[(j+1)%n]),
                                              Edge(self.current_solution.nodes[i], self.current_solution.nodes[(i+1)%n])))
                added_edges = AddedEdges((Edge(self.current_solution.nodes[i-1], self.current_solution.nodes[j]),
                                            Edge(self.current_solution.nodes[i], self.current_solution.nodes[(j+1)%n]),
                                            Edge(self.current_solution.nodes[(i+1)%n], self.current_solution.nodes[i])))
                move = Move(removed_edges, added_edges,reversed_edges, MoveType.intra_two_nodes_exchange)
                if move in self.tracker.moves_set:
                    continue
                else:
                    # we don't have this move so we calculate delta and add into the heap
                    score_delta = (
                        -self.distance_matrix[self.current_solution.nodes[i-1].index][self.current_solution.nodes[i].index]
                        -self.distance_matrix[self.current_solution.nodes[j].index][self.current_solution.nodes[(j+1)%n].index]
                        +self.distance_matrix[self.current_solution.nodes[i-1].index][self.current_solution.nodes[j].index]
                        +self.distance_matrix[self.current_solution.nodes[i].index][self.current_solution.nodes[(j+1)%n].index]
                    )
                    if score_delta < 0:
                        self.tracker.add_move(move, score_delta)
                
                # score_delta = (
                #     -self.distance_matrix[self.current_solution[i - 1]][self.current_solution[i]]
                #     -self.distance_matrix[self.current_solution[j]][self.current_solution[(j + 1) % n]]
                #     +self.distance_matrix[self.current_solution[i - 1]][self.current_solution[j]]
                #     +self.distance_matrix[self.current_solution[i]][self.current_solution[(j + 1) % n]]
                # )
            else:  # non-adjacent nodes case
                
                removed_edges = RemovedEdges((Edge(self.current_solution.nodes[i-1], self.current_solution.nodes[i]),
                                              Edge(self.current_solution.nodes[j-1], self.current_solution.nodes[j]),
                                              Edge(self.current_solution.nodes[i], self.current_solution.nodes[(i+1)%n]),
                                              Edge(self.current_solution.nodes[j], self.current_solution.nodes[(j+1)%n])))
                added_edges = AddedEdges((Edge(self.current_solution.nodes[i-1], self.current_solution.nodes[j]),
                                            Edge(self.current_solution.nodes[j-1], self.current_solution.nodes[i]),
                                            Edge(self.current_solution.nodes[i], self.current_solution.nodes[(j+1)%n]),
                                            Edge(self.current_solution.nodes[j], self.current_solution.nodes[(i+1)%n])))
                move = Move(removed_edges, added_edges, reversed_edges, MoveType.intra_two_nodes_exchange)
                if move in self.tracker.moves_set:
                    continue
                else:
                    score_delta = (
                        -self.distance_matrix[self.current_solution.nodes[i-1].index][self.current_solution.nodes[i].index]
                        -self.distance_matrix[self.current_solution.nodes[j-1].index][self.current_solution.nodes[j].index]
                        -self.distance_matrix[self.current_solution.nodes[i].index][self.current_solution.nodes[(i+1)%n].index]
                        -self.distance_matrix[self.current_solution.nodes[j].index][self.current_solution.nodes[(j+1)%n].index]
                        +self.distance_matrix[self.current_solution.nodes[i-1].index][self.current_solution.nodes[j].index]
                        +self.distance_matrix[self.current_solution.nodes[j-1].index][self.current_solution.nodes[i].index]
                        +self.distance_matrix[self.current_solution.nodes[i].index][self.current_solution.nodes[(j+1)%n].index]
                        +self.distance_matrix[self.current_solution.nodes[j].index][self.current_solution.nodes[(i+1)%n].index]
                    )
                    if score_delta < 0:
                        self.tracker.add_move(move, score_delta)
                
                # score_delta = (
                #     -self.distance_matrix[self.current_solution[i - 1]][self.current_solution[i]]
                #     -self.distance_matrix[self.current_solution[j - 1]][self.current_solution[j]]
                #     +self.distance_matrix[self.current_solution[i - 1]][self.current_solution[j]]
                #     +self.distance_matrix[self.current_solution[j - 1]][self.current_solution[i]]
                #     -self.distance_matrix[self.current_solution[i]][self.current_solution[(i + 1) % n]]
                #     -self.distance_matrix[self.current_solution[j]][self.current_solution[(j + 1) % n]]
                #     +self.distance_matrix[self.current_solution[i]][self.current_solution[(j + 1) % n]]
                #     +self.distance_matrix[self.current_solution[j]][self.current_solution[(i + 1) % n]]
                # )

    
    
    def inter_route_exchange(self, start_index=0, direction="right"):
        n_selected = len(self.current_solution.nodes)
        n_unselected = len(self.unselected_nodes)
        
        # Create all possible combinations of selected and unselected nodes
        all_combinations = [(i, j) for i in range(n_selected) for j in range(n_unselected)]
        if direction == "left":
            all_combinations = all_combinations[::-1]
        for i, j in all_combinations[start_index:]:
            selected_node: Node = self.current_solution.nodes[i]
            new_node: Node = self.unselected_nodes[j]
            new_solution: list[Node] = self.current_solution.nodes.copy()
            new_solution[i]: Node = new_node
            prev_node_index = (i - 1) % n_selected
            next_node_index = (i + 1) % n_selected
            reversed_edges=ReversedEdges(tuple())
            
            removed_edges = RemovedEdges((Edge(self.current_solution.nodes[prev_node_index], selected_node),
                                          Edge(selected_node, self.current_solution.nodes[next_node_index])))
            added_edges = AddedEdges((Edge(self.current_solution.nodes[prev_node_index], new_node),
                                        Edge(new_node, self.current_solution.nodes[next_node_index])))
            
            move = Move(removed_edges, added_edges, reversed_edges,  MoveType.inter_route_exchange)
            
            if move in self.tracker.moves_set:
                continue
            else:
                # we don't have this move so we calculate delta and add into the heap if delta < 0
                score_delta = (
                    -self.distance_matrix[self.current_solution.nodes[prev_node_index].index][selected_node.index]
                    -self.distance_matrix[selected_node.index][self.current_solution.nodes[next_node_index].index]
                    +self.distance_matrix[self.current_solution.nodes[prev_node_index].index][new_node.index]
                    +self.distance_matrix[new_node.index][self.current_solution.nodes[next_node_index].index]
                    -selected_node.cost
                    +new_node.cost
                )
                if score_delta < 0:
                    self.tracker.add_move(move, score_delta)
                
            
            # score_delta = (
            #     -self.distance_matrix[self.current_solution[prev_node_index]][selected_node]
            #     -self.distance_matrix[selected_node][self.current_solution[next_node_index]]
            #     +self.distance_matrix[self.current_solution[prev_node_index]][new_node]
            #     +self.distance_matrix[new_node][self.current_solution[next_node_index]]
            #     -self.costs[selected_node]
            #     +self.costs[new_node]
            # )
          
    """Tutaj można się zastanowić czy nie da się lepiej jakoś układać nodów
    nowego solution. Musimy mieć na pewno przestawione, bo na jego podstawie
    robimy nowego ruchy, ale iterowanie po wszystkich edgach jest mega wolne,
    więc może jest lepszy sposób
    """  
    def order_edges(self, edges: list[Edge]) -> list[Edge]:
        if not edges:
            return []

        # Start with the first edge and initialize the ordered list of edges
        ordered_edges = [edges.pop(0)]

        # Iterate until all edges are ordered
        while edges:
            last_dst = ordered_edges[-1].dst
            for i, edge in enumerate(edges):
                if edge.src == last_dst:
                    ordered_edges.append(edges.pop(i))
                    break

        return ordered_edges
            
    def apply_move(self, move: Move, score: int) -> None:
        # Step 1: Convert Solution.edges to a list for manipulation
        # if a move is inter-route we have to update unselected nodes
        if move.type == MoveType.inter_route_exchange:
            self.unselected_nodes.append(move.removed_edges.edges[1].src)
            self.unselected_nodes.remove(move.added_edges.edges[1].src)
        
        modified_edges = list(self.current_solution.edges)


        # Step 2: Remove edges in move.removed_edges from the list
        if move.type != MoveType.intra_two_edges_exchange:
            indeces=[]
            for edge in move.removed_edges.edges:
                
                if edge in modified_edges:
                    index = modified_edges.index(edge)
                    indeces.append(index)
                    
            count=0 
            for idx,edge in enumerate(move.added_edges.edges):
                
                dst1=edge.dst.index
                for idx2,edge2 in enumerate(move.added_edges.edges):
                    src2=edge2.src.index
                    
                    if dst1==src2 and count<2:
                        count+=1
                        modified_edges[indeces[idx]]=edge
                        modified_edges[(indeces[idx]+1)%len(modified_edges)]=edge2 

        else: 
            indeces=[]
            for edge in move.removed_edges.edges:
                
                if edge in modified_edges:
                    index = modified_edges.index(edge)
                    indeces.append(index)

            for index, edge in enumerate(move.added_edges.edges):
                modified_edges[indeces[index]]=edge

            start = indeces[0]
            end = indeces[1]   
            for index,edge in enumerate(move.reversed_edges.edges):

                inverted_edge = Edge(edge.dst,edge.src)

                modified_edges[end-index-1]=inverted_edge
                


        ordered_edges = modified_edges
        nodes_in_order = [ordered_edges[0].src] + [edge.dst for edge in ordered_edges[:-1]]

            # Step 6: Create and return a new Solution object
        self.current_solution = Solution(nodes=nodes_in_order)
        self.current_score += score
            
    
    def run(self, 
            start_solution: Solution,
            moves: list[str],
            show_progress: bool = True
            ) -> tuple[list[int], int]:
        if not set(moves).issubset(self.moves.keys()):
            raise ValueError("Invalid moves list")
        if len(moves) != 2:
            raise ValueError("Only two moves are supported")
        
        if start_solution is not None: 
            self.current_solution = start_solution
            
            self.current_score = objective_function(start_solution, self.distance_matrix, self.costs)    
            
        progress = True
        epoch_counter = 0
        
        while progress:
            for move in moves:
                self.moves[move]()
            
            
            # process the LM
            temp_moves = [] # here we will store moves that need to be added to the heap again (condition #2 from the presentation)
            # we can't keep them all the time beacuse then we will never end while loop
            progress = False
            while self.tracker.moves_heap:
                score, move = heapq.heappop(self.tracker.moves_heap) # drop the best element from the heap
                if move.type == MoveType.inter_route_exchange:
                    applicability = self.is_inter_move_applicable(self.current_solution, move)
                    # print("applicability of inter route", applicability)
                else:
                    applicability = self.is_intra_move_applicable(self.current_solution, move)
                
                if applicability == -1:
                    continue # we just drop it from the heap, nothing happens
                elif applicability == 0:
                    # we have to then re-add it to the heap
                    temp_moves.append((score, move))
                else: # we apply this move
                    # print("###############")
                    # print(f"Applying move: {move.type}, Score: {score}")
                    # print(f"Old solution: {self.current_solution}")
                    # print(f"Move: {move}")

                    # if move.type == MoveType.inter_route_exchange:
                    #     print(self.unselected_nodes)
                    self.apply_move(move, score)
                    # print("New solution: ", self.current_solution)
                    progress = True
                    break


            # rebuild the heap
            for item in temp_moves:
                heapq.heappush(self.tracker.moves_heap, item)
    
            epoch_counter += 1
            if epoch_counter % 5 == 0 and show_progress:
                print(f"Epoch: {epoch_counter}, Score: {self.current_score}")
                # print(self.tracker.moves_heap)
        print(objective_function(self.current_solution,self.distance_matrix, self.costs),self.current_score)
                
            


if __name__ == "__main__":
    a = pd.read_csv("data/TSPA.csv", sep=';', header=None, names=["x", "y", "cost"])
    distance_matrix = calculate_distance_matrix(a)
    costs = a["cost"].to_numpy()
    initial_solution = generate_random_solution(100, costs)
    
    steepest = SteepestLocalSearch(initial_solution=initial_solution,
                                   distance_matrix=distance_matrix,
                                   costs=costs)
    start = time.time()
    steepest.run(start_solution=initial_solution,
                 moves=["inter-route-exchange", "intra-two-edges-exchange"],
                 show_progress=True)
    end = time.time()
    print("Runtime: ", end - start)
        
        
            