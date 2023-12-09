from typing import Literal
from utils import *
import heapq




class SteepestLocalSearch():
    def __init__(self, 
                 initial_solution: list[int], 
                 distance_matrix: list[list[int]], 
                 costs: list[int]) -> None:
        self.initial_solution = initial_solution
        self.initial_score = objective_function(initial_solution, distance_matrix, costs)
        self.distance_matrix = distance_matrix
        self.costs = costs
        
        self.lm = []
        self.n = len(initial_solution)
        self.all_nodes = list(range(200))
        self.unselected_nodes = [node for node in self.all_nodes if node not in self.initial_solution]
        
        self.current_solution: list[int] = initial_solution
        self.current_score = self.initial_score
        
        self.moves = {
            "edges": self.two_edges_exchange,
            "inter": self.inter_route_exchange
        }
        
        self.moves_applicability = {
            "edges": self.check_edges_applicability,
            "inter": self.check_inter_applicability
        }
        
        self.apply_moves = {
            "edges": self.apply_edge_move,
            "inter": self.apply_inter_move
        }
        
    def first_two_edges_exchange(self):
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

        range_i = range(self.n - 2)
        range_j = lambda i: range(i + 2, self.n)

        for i in range_i:
            for j in range_j(i):

                score_delta = (
                    -self.distance_matrix[self.current_solution[i]][self.current_solution[i + 1]]
                    -self.distance_matrix[self.current_solution[j]][self.current_solution[(j + 1) % self.n]]
                    +self.distance_matrix[self.current_solution[i]][self.current_solution[j]]
                    +self.distance_matrix[self.current_solution[i + 1]][self.current_solution[(j + 1) % self.n]]
                )
                
                if score_delta < 0:
                    added_edges = ((self.current_solution[i], self.current_solution[j]), (self.current_solution[i + 1], self.current_solution[(j + 1) % self.n]))
                    removed_edges = ((self.current_solution[i], self.current_solution[i + 1]), (self.current_solution[j], self.current_solution[(j + 1) % self.n]))
                    heapq.heappush(self.lm, (score_delta, ("edges", added_edges, removed_edges, "normal")))                    
    
    
    def first_inter_route_exchange(self):
        n_unselected = len(self.unselected_nodes)
        
        # Create all possible combinations of selected and unselected nodes
        all_combinations = [(i, j) for i in range(self.n) for j in range(n_unselected)]
        for i, j in all_combinations:
            selected_node = self.current_solution[i]
            new_node = self.unselected_nodes[j]
            new_solution = self.current_solution.copy()
            new_solution[i] = new_node
            prev_node_index = (i - 1) % self.n
            next_node_index = (i + 1) % self.n
            score_delta = (
                -self.distance_matrix[self.current_solution[prev_node_index]][selected_node]
                -self.distance_matrix[selected_node][self.current_solution[next_node_index]]
                +self.distance_matrix[self.current_solution[prev_node_index]][new_node]
                +self.distance_matrix[new_node][self.current_solution[next_node_index]]
                -self.costs[selected_node]
                +self.costs[new_node]
            )
            if score_delta < 0:
                added_edges = ((self.current_solution[prev_node_index], new_node), (new_node, self.current_solution[next_node_index]))
                removed_edges = ((self.current_solution[prev_node_index], selected_node), (selected_node, self.current_solution[next_node_index]))
                heapq.heappush(self.lm, (score_delta, ("inter", added_edges, removed_edges, "normal")))
                
    def two_edges_exchange(self, affected_nodes: tuple[int, int, int, int]) -> None:
        node1, node2, node3, node4 = affected_nodes
        
        nodes_indices = [(self.current_solution.index(node1)-1)%self.n ,self.current_solution.index(node1), self.current_solution.index(node2), self.current_solution.index(node3), self.current_solution.index(node4), (self.current_solution.index(node4)+1)%self.n]
        for i in nodes_indices:
            for j in range(self.n):
                x, y = i, j
                if y < x:
                    x, y = y, x
                if abs(x-y) >= 2:
                    # new_solution = (solution[:x + 1] 
                    #                 + solution[x + 1:y + 1][::-1] 
                    #                 + solution[y + 1:])
                    score_delta = (
                        -self.distance_matrix[self.current_solution[x]][self.current_solution[x + 1]]
                        -self.distance_matrix[self.current_solution[y]][self.current_solution[(y + 1) % self.n]]
                        +self.distance_matrix[self.current_solution[x]][self.current_solution[y]]
                        +self.distance_matrix[self.current_solution[x + 1]][self.current_solution[(y + 1) % self.n]])
                    if score_delta < 0:
                        added_edges = ((self.current_solution[x], self.current_solution[y]), (self.current_solution[x + 1], self.current_solution[(y + 1) % self.n]))
                        removed_edges = ((self.current_solution[x], self.current_solution[x + 1]), (self.current_solution[y], self.current_solution[(y + 1) % self.n]))
                        heapq.heappush(self.lm, (score_delta, ("edges", added_edges, removed_edges, "normal")))
                
    def inter_route_exchange(self, affected_nodes: tuple[int, int, int, int]) -> None:
        
        n_unselected = len(self.unselected_nodes)
        all_combinations = [(self.current_solution.index(i), j) for i in affected_nodes for j in range(n_unselected)]
        for i in affected_nodes:
            inx_1 = (self.current_solution.index(i)-1)%self.n
            inx_2 = (self.current_solution.index(i)+1)%self.n
            for j in range(n_unselected):
                all_combinations.append((inx_1, j))
                all_combinations.append((inx_2, j))
        for i, j in all_combinations:
            selected_node = self.current_solution[i]
            new_node = self.unselected_nodes[j]
            new_solution = self.current_solution.copy()
            new_solution[i] = new_node
            prev_node_index = (i - 1) % self.n
            next_node_index = (i + 1) % self.n
            score_delta = (
                -self.distance_matrix[self.current_solution[prev_node_index]][selected_node]
                -self.distance_matrix[selected_node][self.current_solution[next_node_index]]
                +self.distance_matrix[self.current_solution[prev_node_index]][new_node]
                +self.distance_matrix[new_node][self.current_solution[next_node_index]]
                -self.costs[selected_node]
                +self.costs[new_node]
            )
            if score_delta < 0:
                added_edges = ((self.current_solution[prev_node_index], new_node), (new_node, self.current_solution[next_node_index]))
                removed_edges = ((self.current_solution[prev_node_index], selected_node), (selected_node, self.current_solution[next_node_index]))
                heapq.heappush(self.lm, (score_delta, ("inter", added_edges, removed_edges, "normal")))
    
    def check_inter_applicability(
        self, 
        added_edges: tuple[tuple[int, int], tuple[int, int]],
        removed_edges: tuple[tuple[int, int], tuple[int, int]]) -> Literal[-1, 0, 1]:
        
        first_added, second_added = added_edges
        first_removed, second_removed = removed_edges
        
        if ((first_removed[0] not in self.current_solution) or
            (second_removed[1] not in self.current_solution) or
            (first_added[0] not in self.current_solution) or
            (second_added[1] not in self.current_solution) or
            (first_removed[1] not in self.current_solution) or
            (first_added[1] not in self.unselected_nodes)):
            return -1
        
        all_edges_match = True
        for edge in [first_removed, second_removed]:
            inx_first_node = self.current_solution.index(edge[0])
            if ((self.current_solution[(inx_first_node + 1) % self.n] != edge[1]) and
                (self.current_solution[inx_first_node - 1] != edge[1])):
                return -1
            
            elif ((self.current_solution[(inx_first_node + 1) % self.n] != edge[1]) and
                (self.current_solution[inx_first_node - 1] == edge[1])):
                all_edges_match = False
            # case when the edge is not present in the current solution
        return 1 if all_edges_match else 0
    
    
    def check_edges_applicability(
        self,
        added_edges: tuple[tuple[int, int], tuple[int, int]],
        removed_edges: tuple[tuple[int, int], tuple[int, int]]) -> Literal[-1, 0, 1]:
        
        for edge in added_edges:
            if (edge[0] not in self.current_solution) or (edge[1] not in self.current_solution):
                return -1
            
        all_edges_match = True
        for edge in removed_edges:
            inx_first_node = self.current_solution.index(edge[0])
            if ((self.current_solution[(inx_first_node + 1) % self.n] != edge[1]) and
                (self.current_solution[inx_first_node - 1] != edge[1])):
                return -1
            elif ((self.current_solution[(inx_first_node + 1) % self.n] != edge[1]) and
                (self.current_solution[inx_first_node - 1] == edge[1])):
                all_edges_match = False
        return 1 if all_edges_match else 0
    
    
    def apply_inter_move(self, added_edges, removed_edges, _):
        new_node = added_edges[0][1]
        
        inx_old_node = self.current_solution.index(removed_edges[0][1])
        self.current_solution[inx_old_node] = new_node
        
        self.unselected_nodes.remove(new_node)
        self.unselected_nodes.append(removed_edges[0][1])
        # print(f"Exchanged: {removed_edges[0][1]}->{new_node}")
        return added_edges[0] + added_edges[1] # return 4 integers
    
    def apply_edge_move(self, added_edges, removed_edges, action):
        x = self.current_solution.index(added_edges[0][0])
        y = self.current_solution.index(added_edges[0][1])
        if y < x:
            x, y = y, x
        self.current_solution = (self.current_solution[:x + 1] 
                                    + self.current_solution[x + 1:y + 1][::-1] 
                                    + self.current_solution[y + 1:])
        return added_edges[0] + added_edges[1]

    
    def run(self, 
            start_solution: list[int],
            moves: list[str],
            show_progress: bool = True) -> tuple[list[int], int]:
        
        self.current_solution = start_solution
        self.current_score = objective_function(start_solution, self.distance_matrix, self.costs)
        
        self.first_inter_route_exchange()
        self.first_two_edges_exchange()
        
        n_epoch = 0
        progress = True
        while(progress):
            
            progress = False
            temp_moves = []
            check_for_duplicates(self.current_solution, "current_solution")
            check_for_duplicates(self.unselected_nodes, "unselected_nodes")
            check_overlap(self.current_solution, self.unselected_nodes)
            # if len(self.unselected_nodes) != 180:
            #     raise ValueError("Error: The number of unselected nodes is not 180.")
            while(self.lm):
                delta_score, (move_type, added_edges, removed_edges, _) = heapq.heappop(self.lm)
                if self.moves_applicability[move_type](added_edges, removed_edges) == 1:
                    affected_nodes = self.apply_moves[move_type](added_edges, removed_edges, _)
                    self.current_score += delta_score
                    progress = True
                    break
                elif self.moves_applicability[move_type](added_edges, removed_edges) == 0:
                    temp_moves.append((delta_score, (move_type, added_edges, removed_edges, _)))
                else:
                    continue
                
            for item in temp_moves:
                heapq.heappush(self.lm, item)
                
            if progress:
                for move in moves:
                    self.moves[move](affected_nodes)
        
            n_epoch += 1
            if show_progress and n_epoch % 10 == 0:
                print(f"Epoch {n_epoch}: {self.current_score}, {self.current_solution}")
        return self.current_solution, self.current_score
    
def perturb(current_solution: list[int], 
                       current_distance: float, 
                       distance_matrix: list[list[int]]):
    l = len(current_solution)
    for x in range(4):
        i = random.randint(2,l-2)

        j = random.randint(0,i-1)

        k = random.randint(j+2,l)

        m = random.randint(0,k-1)

        new_solution = (current_solution[:j ] 
                    + current_solution[j :i ][::-1] 
                    + current_solution[i :])
        current_solution = new_solution
        new_solution = (current_solution[:m ] 
                    + current_solution[m :k ][::-1] 
                    + current_solution[k :])

        current_solution = new_solution




    return current_solution
def generate_greedy_weight_regret(dist_matrix, costs, current_solution, num_select, a, start_idx):
    num_nodes = dist_matrix.shape[0]
    selected_nodes = current_solution[start_idx:]+current_solution[:start_idx]
    start=selected_nodes[-1]
    unselected_nodes = set(range(num_nodes)) - set(current_solution)
    total_distance = 0
    initial_score=objective_function(selected_nodes,dist_matrix,costs)
    

    # Continue until we've selected the required number of nodes
    while len(selected_nodes) < num_select:
        score_node = None
        score_position = None
        best_score= float('-inf')
        score_best_increase = float('inf')
        # Evaluate the insertion of each unselected node
        for node in unselected_nodes:
            best_node = None
            best_position = None
            best_min_increase = float('inf')
            second_best_min_increase = float('inf')
            # Try inserting between each pair of consecutive nodes in the cycle
            for i in range(selected_nodes.index(start),len(selected_nodes)):
                # Calculate the increase in distance
                
                next_i = (i + 1) % len(selected_nodes)
                # print(selected_nodes.index(start))
                increase = (dist_matrix[selected_nodes[i], node] +
                            dist_matrix[node, selected_nodes[next_i]]+
                            costs[node]-
                            dist_matrix[selected_nodes[i], selected_nodes[next_i]])
                
                # Check if it is the best position
                if increase < second_best_min_increase: 
                    if increase < best_min_increase:
                        best_min_increase = increase
                        best_node = node
                        best_position = next_i 
                         # Insert before next_i
                    #or the second best position
                    else: 
                        second_best_min_increase = increase
            #for a given unselected node after checking all of the positions and finding two best increases, we calculate regret
            regret= second_best_min_increase - best_min_increase
            score = a * regret - (1-a)*best_min_increase
            # for keeping track of the best regret so far and best corresponding node, position and increase
            if score > best_score:
                best_score = score
                score_node =  best_node
                score_position = best_position 
                score_best_increase = best_min_increase
        
                    

        # Insert the best node into the cycle
        if score_position==0:
            selected_nodes.append(score_node)
            # print(selected_nodes, score_node)
        else:
            selected_nodes.insert(score_position, score_node)
        # print(selected_nodes,score_position)
        unselected_nodes.remove(score_node)
        total_distance += score_best_increase
        
    return selected_nodes



def destroy(current_solution: list[int]):
    n = len(current_solution)

    subset_length = random.randint(int(0.2 * n), int(0.3 * n))
    start_index = random.randint(0, n)
    start=current_solution[start_index-1]
    if start_index + subset_length > n:
        solution = current_solution[start_index+subset_length-n:start_index]
    else: 
        solution = current_solution[:start_index] + current_solution[start_index + subset_length:]
    # print(solution)
    return solution, start
    
def repair(current_solution: list[int],distance_matrix: list[list[int]],costs, num_nodes, start_node):
    start_idx= current_solution.index(start_node)+1
    selected_nodes = generate_greedy_weight_regret(distance_matrix,costs, current_solution, num_nodes, 0.5, start_idx)
    return selected_nodes
    
    
if __name__ == "__main__":
    import json
    import time   
    instances = {
        "A": pd.read_csv("data/TSPA.csv", sep=';', header=None, names=["x", "y", "cost"]),
        "B": pd.read_csv("data/TSPB.csv", sep=';', header=None, names=["x", "y", "cost"]),
        "C": pd.read_csv("data/TSPC.csv", sep=';', header=None, names=["x", "y", "cost"]),
        "D": pd.read_csv("data/TSPD.csv", sep=';', header=None, names=["x", "y", "cost"]),
    } 
    # with open("Results/runtimes_MSLS_means.json", "r") as f:
    #     means= json.load(f)
    # best_solutions_LSNS={}
    # for instance in instances: 
    #     solutions = []
    #     runtimes = [] 
    #     best_solution=[]
    #     distance_matrix = calculate_distance_matrix(instances[instance])
    #     costs = instances[instance]["cost"].to_numpy()
    #     epochs=[]
    #     for x in range(20):
    #         solution= generate_random_solution(100)
    #         s = SteepestLocalSearch(solution, distance_matrix, costs)
    #         solution, score = s.run(solution, ["inter","edges"], show_progress=False)
    #         best_score=score
    #         best_solution=solution
    #         start=time.time()
    #         i=0
    #         while time.time()-start<means[instance]:
    #             i+=1
                
    #             destroyed, start_idx = destroy(best_solution)
    #             repaired = repair(destroyed, distance_matrix, costs 100, start_idx)

                
    #             s = SteepestLocalSearch(repaired, distance_matrix, costs)
    #             solution, score = s.run(repaired, ["inter","edges"], show_progress=False)
    #             if score<best_score:
    #                 best_score=score
    #                 best_solution=solution

                
    #         epochs.append(i)
    #         solutions.append((best_solution,best_score))
    #         end_time = time.time()
    #         runtimes.append(end_time-start)
    #         print("Best_score",best_score, objective_function(best_solution,distance_matrix, costs))

    #     print("Solutions for instance-"+instance)
    #     print(f"Average score: {np.mean([x[1] for x in solutions])}, min score: {min([x[1] for x in solutions])}, max score: {max([x[1] for x in solutions])}")
    #     print("Runtimes for instance-"+instance)
    #     print(f"Average runtime: {np.mean(runtimes)}, min runtime: {min(runtimes)}, max runtime: {max(runtimes)}")
    #     print(f"Avg number of iterations:{np.mean(epochs)}")
    #     best_solution = min(solutions, key=lambda x: x[1])
    #     best_solutions_LSNS["LSNS-best-score-"+str(instance)] = best_solution

    #     with open("Results/best_solutions_LSNS.json", "w") as f:
    #         json.dump(best_solutions_LSNS, f, indent=4)


    # solution= generate_random_solution(20)
    # print("Pierwotne",solution)
    # distance_matrix = calculate_distance_matrix(instances["A"])
    # costs = instances["A"]["cost"].to_numpy()
    # score = objective_function(solution, distance_matrix, costs)
    # print(score)
    # destroyed, start = destroy(solution)
    # print("Zniszczone",destroyed)
    # solution = repair(destroyed, distance_matrix,costs,20, start)
    # print("Naprawione",solution)
    # score = objective_function(solution, distance_matrix, costs)
    # print(score)
    
    with open("Results/runtimes_MSLS_means.json", "r") as f:
        means= json.load(f)
    best_solutions_LSNS={}
    for instance in instances: 
        solutions = []
        runtimes = [] 
        best_solution=[]
        distance_matrix = calculate_distance_matrix(instances[instance])
        costs = instances[instance]["cost"].to_numpy()
        epochs=[]
        for x in range(20):
            solution= generate_random_solution(100)
            s = SteepestLocalSearch(solution, distance_matrix, costs)
            solution, score = s.run(solution, ["inter","edges"], show_progress=False)
            best_score=score
            best_solution=solution
            start=time.time()
            i=0
            while time.time()-start<means[instance]:
                i+=1
                destroyed,start_idx = destroy(best_solution)
                solution = repair(destroyed, distance_matrix, costs, 100, start_idx)
                score = objective_function(solution, distance_matrix, costs)
                
                
                # s = SteepestLocalSearch(repaired, distance_matrix, costs)
                # solution, score = s.run(repaired, ["inter","edges"], show_progress=False)
                if score<best_score:
                    best_score=score
                    best_solution=solution
                
                print("Best_score",best_score, score)
               
                
            epochs.append(i)
            solutions.append((best_solution,best_score))
            end_time = time.time()
            runtimes.append(end_time-start)
            print("Best_score",best_score, objective_function(best_solution,distance_matrix, costs))

        print("Solutions for instance-"+instance)
        print(f"Average score: {np.mean([x[1] for x in solutions])}, min score: {min([x[1] for x in solutions])}, max score: {max([x[1] for x in solutions])}")
        print("Runtimes for instance-"+instance)
        print(f"Average runtime: {np.mean(runtimes)}, min runtime: {min(runtimes)}, max runtime: {max(runtimes)}")
        print(f"Avg number of iterations:{np.mean(epochs)}")
        best_solution = min(solutions, key=lambda x: x[1])
        best_solutions_LSNS["LSNS-best-score-"+str(instance)] = best_solution

        with open("Results/best_solutions_LSNS_2.json", "w") as f:
            json.dump(best_solutions_LSNS, f, indent=4)

            
                


    
        

                
            
            
        
        
        
            