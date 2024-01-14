from typing import Literal

from tqdm import tqdm
from utils import *
import heapq
import asyncio
import json
import time  
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

pool_executor = ProcessPoolExecutor()


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
            # check_for_duplicates(self.current_solution, "current_solution")
            # check_for_duplicates(self.unselected_nodes, "unselected_nodes")
            # check_overlap(self.current_solution, self.unselected_nodes)
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


def generate_init_population(n: int, n_population: int) -> list[list[int]]:
    """
    Generate an initial population of random solutions.

    :param n: The number of nodes.
    :param n_population: The number of solutions to generate.
    :return: A list of solutions.
    """
    return [generate_random_solution(n) for _ in range(n_population)]

def select_random_parents(population: list[list[int]]) -> tuple[list[int], list[int]]:
    """
    Select two random parents from the population.

    :param population: A list of solutions.
    :return: A tuple containing the two selected parents.
    """
    parents =  random.sample(population, 2)
    return parents[0], parents[1]


def find_common_edges(path1: list[int], path2: list[int], n: int = 100) -> set[tuple[int, int]]:
    edges1 = []
    edges2 = []
    for i in range(n):
        edges1.append((path1[i], path1[(i+1)%n]))
        edges2.append((path2[i], path2[(i+1)%n]))
        edges2.append((path2[(i+1)%n], path2[i]))    
    e1_set = set(edges1)
    e2_set = set(edges2)
    return e1_set.intersection(e2_set)


def connect_edges(edges: set[tuple[int, int]]) -> list[tuple[int]]:
    # Create a dictionary to represent the graph
    graph = defaultdict(set)
    for u, v in edges:
        graph[u].add(v)

    visited = set()
    connected_components = []

    # Perform depth-first search to form connected components


    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)

    for node in graph:
        if node not in visited:
            component = []
            dfs(node, component)
            connected_components.append(tuple(component))

    return connected_components


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def fill_path(subpaths: list[tuple[int]], n: int = 100) -> list[int]:

    unselected_nodes = set(range(n*2)) - {node for subpath in subpaths for node in subpath}
    unselected_nodes = list(unselected_nodes)
    additional_nodes = random.sample(unselected_nodes, n - sum(len(subpath) for subpath in subpaths))
    random.shuffle(additional_nodes)
    split_additional_nodes = list(split(additional_nodes, len(subpaths)))
    path = []
    for subpath, additional_nodes in zip(subpaths, split_additional_nodes):
        path.extend(subpath)
        path.extend(additional_nodes)
    return path

def perturb(current_solution: list[int]):
    l = len(current_solution)
    for x in range(3):
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

def recombine_subpath_operator(parent1: list[int], parent2: list[int]) -> list[int]:
    offspring = []
    # parent1 = perturb(parent1)
    n = len(parent1)
    num_nodes = 20
    while len(offspring) < n:
        random_start = random.randint(0, len(parent1) - num_nodes)    
        random_subpath = parent1[random_start : random_start + num_nodes]
        
        parent1 = [el for el in parent1 if el not in random_subpath]
        parent2 = [el for el in parent2 if el not in random_subpath]
        
        offspring.extend(random_subpath)
        
        parent1, parent2 = parent2, parent1
    return offspring
    


def create_offspring_solution(parent1: list[int], parent2: list[int]) -> list[int]:
    if random.random() < 1.0:
        offspring = recombine_subpath_operator(parent1, parent2)
        return offspring
    else:
        common_edges = find_common_edges(parent1, parent2)
        if not common_edges:
            offspring = generate_random_solution(len(parent1))
            return offspring
        connected_components = connect_edges(common_edges)
        offspring = fill_path(connected_components)
        return offspring




def run_algorithm(distance_matrix: list[list[int]], costs: list[int], avg_runtime: int) -> tuple[list[int], int, float, int]:
    population = generate_init_population(100, 30)
    population = [(solution, objective_function(solution, distance_matrix, costs)) for solution in population]
    start_time = time.time()
    worst_solution = max(population, key=lambda x: x[1])
    all_scores = [x[1] for x in population]
    i = 0
    while time.time() - start_time < avg_runtime:
        i += 1
        # if i ==1200:
        #     break
        if i%2 == 0:
            parent1, parent2 = select_random_parents(population)
            offspring = create_offspring_solution(parent1[0], parent2[0])
        else:
            parent1, parent2 = select_random_parents(population)
            offspring = perturb(parent1[0])
        s = SteepestLocalSearch(offspring, distance_matrix, costs)
        offspring, offspring_score = s.run(offspring, ["inter","edges"], show_progress=False)
        if offspring_score < worst_solution[1] and offspring_score not in all_scores:
            population.remove(worst_solution)
            population.append((offspring, offspring_score))
            all_scores.remove(worst_solution[1])
            all_scores.append(offspring_score)
            worst_solution = max(population, key=lambda x: x[1])
    end_time = time.time()
    runtime = end_time-start_time
    best_solution, best_score = min(population, key=lambda x: x[1])
    return best_solution, best_score, runtime, i
    


def main():
    instances = {
        "A": pd.read_csv("data/TSPA.csv", sep=';', header=None, names=["x", "y", "cost"]),
        "B": pd.read_csv("data/TSPB.csv", sep=';', header=None, names=["x", "y", "cost"]),
        "C": pd.read_csv("data/TSPC.csv", sep=';', header=None, names=["x", "y", "cost"]),
        "D": pd.read_csv("data/TSPD.csv", sep=';', header=None, names=["x", "y", "cost"]),
    }
    with open("MSLS_v2.json", "r") as f:
        means = json.load(f)
    best_solutions_LSNS={
        "A": {},
        "B": {},
        "C": {},
        "D": {}
    }
    for instance in instances: 
        solutions = []
        runtimes = [] 
        best_solution=[]
        distance_matrix = calculate_distance_matrix(instances[instance])
        costs = instances[instance]["cost"].to_numpy()
        epochs=[]
        with ProcessPoolExecutor(max_workers=5) as executor:
            tasks = [executor.submit(run_algorithm, distance_matrix, costs, means[instance]["avg_runtime"]) for _ in range(20)]
            for task in tasks:
                best_solution, best_score, runtime, i = task.result()
                solutions.append((best_solution, best_score))
                runtimes.append(runtime)
                epochs.append(i)
                print(f" best score: {best_score}, iterations: {epochs}")

        print("Solutions for instance-"+instance)
        print(f"Average score: {np.mean([x[1] for x in solutions])}, min score: {min([x[1] for x in solutions])}, max score: {max([x[1] for x in solutions])}")
        print("Runtimes for instance-"+instance)
        print(f"Average runtime: {np.mean(runtimes)}, min runtime: {min(runtimes)}, max runtime: {max(runtimes)}")
        print(f"Avg number of iterations:{np.mean(epochs)}, min epochs: {min(epochs)}, max epochs: {max(epochs)}")
        best_solution = min(solutions, key=lambda x: x[1])
        best_solutions_LSNS[instance]["best-path"] = best_solution[0]
        best_solutions_LSNS[instance]["best-score"] = best_solution[1]
        best_solutions_LSNS[instance]["score-avg"] = np.mean([x[1] for x in solutions])
        best_solutions_LSNS[instance]["score-min"] = min([x[1] for x in solutions])
        best_solutions_LSNS[instance]["score-max"] = max([x[1] for x in solutions])
        best_solutions_LSNS[instance]["runtimes-avg"] = np.mean(runtimes)
        best_solutions_LSNS[instance]["runtimes-min"] = min(runtimes)
        best_solutions_LSNS[instance]["runtimes-max"] = max(runtimes)
        best_solutions_LSNS[instance]["iterations-avg"] = np.mean(epochs)
        best_solutions_LSNS[instance]["iterations-min"] = min(epochs)
        best_solutions_LSNS[instance]["iterations-max"] = max(epochs)

    with open("results/best_solutions_HEA_v2.json", "w") as f:
        json.dump(best_solutions_LSNS, f, indent=4)

if __name__ == "__main__":
    main()
    

            
                


    
        

                
            
            
        
        
        
            