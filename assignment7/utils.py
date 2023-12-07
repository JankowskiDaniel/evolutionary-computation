from math import sqrt
import random
import pandas as pd
import numpy as np
from numpy import ndarray as NDArray
import matplotlib.pyplot as plt


def calculate_distance_matrix(df: pd.DataFrame):
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
    selected_nodes_indices: list[int], 
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



def objective_function(solution: list[int], dist_matrix: list[list[int]], costs: list[int]) -> int:
    total_score = 0
    n = len(solution)
    for x in range(n):
        total_score += dist_matrix[solution[x - 1]][solution[x]]
        total_score += costs[solution[x]]
    return total_score


# A function that generates a random solution
def generate_random_solution(n: int) -> list[int]:
    """
    Generate a random solution for a given number of nodes.

    :param n: The number of nodes.
    :return: A list of nodes representing the solution.
    """
    return random.sample(range(0, n * 2), n)


def check_overlap(list1, list2):
    # Use set intersection to find common elements
    common_elements = set(list1) & set(list2)
    if common_elements:
        raise ValueError(f"Error: The following elements are present in both lists: {common_elements}")
    

def check_for_duplicates(lst, name: str):
    duplicates = []
    seen = set()
    for item in lst:
        if item in seen:
            duplicates.append(item)
        else:
            seen.add(item)
    if duplicates:
        raise ValueError(f"Error: The following elements are duplicated in {name}: {duplicates}")