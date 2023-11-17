import pandas as pd
import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from math import sqrt
from typing import List, Optional, Tuple, Union 
from models import Solution, Node


def calculate_distance_matrix(df: pd.DataFrame) -> NDArray[np.int32]:
    """
    Calculate the distance matrix from the dataframe.
    The dataframe contains 'x' and 'y' columns for the coordinates.
    The distances are Euclidean, rounded to the nearest integer + the cost of the destination node.
    """
    coordinates = df[['x', 'y']].to_numpy()
    dist_matrix = np.zeros(shape=(len(df), len(df)))
    for i in range(len(coordinates)):
        for j in range(len(coordinates)):
            dist_matrix[i, j] = round(sqrt((coordinates[i, 0] - coordinates[j, 0])**2 + (coordinates[i, 1] - coordinates[j, 1])**2))
    return dist_matrix


def visualize_selected_route(
    selected_nodes_indices: ArrayLike, 
    dataframe: pd.DataFrame,
    title: str) -> None:
    """
    Visualize the selected route returned by the algorithm, including the cost of each node represented by a colormap.

    Parameters:
    selected_nodes_indices (list): Indices of the selected nodes in the route.
    dataframe (DataFrame): DataFrame containing 'x', 'y', and 'cost' columns for each node.
    """
    x = dataframe["x"].to_numpy()
    y = dataframe["y"].to_numpy()
    costs = dataframe["cost"].to_numpy()

    cmap = plt.cm.get_cmap('viridis')
    norm = plt.Normalize(vmin=min(costs), vmax=max(costs))

    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(x, y, c=costs, cmap=cmap, norm=norm, s=100)
    plt.colorbar(scatter, label='Node Cost')

    for i, node in enumerate(selected_nodes_indices):
        start_node = selected_nodes_indices[i]
        end_node = selected_nodes_indices[(i + 1) % len(selected_nodes_indices)]
        plt.plot([x[start_node], x[end_node]], [y[start_node], y[end_node]], 'k-', lw=1)

    plt.title(title, fontsize=18)
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    plt.grid(True)
    plt.show()
    
    
def objective_function(solution: Solution, dist_matrix: list[list[int]], costs: list[int]) -> int:
    total_score = 0
    n = len(solution.nodes)
    for x in range(n):
        total_score += dist_matrix[solution.nodes[x - 1].index][solution.nodes[x].index]
        total_score += costs[solution.nodes[x].index]
    return total_score


# A function that generates a random solution
def generate_random_solution(n: int, costs: list[int]) -> Solution:
    """
    Generate a random solution for a given number of nodes.

    :param n: The number of nodes.
    :return: A list of nodes representing the solution.
    """
    nodes = []
    choosen_indices = random.sample(range(0, n * 2), n)
    for i in choosen_indices:
        node = Node(index=i, cost=costs[i])
        nodes.append(node)
        
    solution = Solution(nodes=nodes)
    return solution


def check_overlap(list1, list2):
    # Use set intersection to find common elements
    common_elements = set(list1) & set(list2)
    if common_elements:
        raise ValueError(f"Error: The following elements are present in both lists: {common_elements}")


def check_for_duplicates(lst, name: str):
    if len(lst) != len(set(lst)):
        raise ValueError(f"Duplicates found in the {name}. {lst}")