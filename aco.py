import aco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import time

class StandardAntColonyOptimizer:
    def __init__(self, num_nodes, distance_matrix, num_ants=20, alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100,
                 func_evals = 0):
        """
        Initializes the Standard Ant Colony Optimizer.

        Args:
            num_nodes (int): Number of nodes (cities).
            distance_matrix (np.ndarray): Distance matrix.
            num_ants (int): Number of ants per iteration.
            alpha (float): Pheromone importance.
            beta (float): Heuristic importance.
            evaporation_rate (float): Pheromone evaporation rate.
            Q (float): Pheromone deposit factor.
        """
        self.num_nodes = num_nodes
        self.distance_matrix = distance_matrix
        self.pheromone = np.ones((num_nodes, num_nodes))
        self.num_ants = num_ants
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance
        self.evaporation_rate = evaporation_rate
        self.Q = Q  # Pheromone deposit factor
        self.best_distance = float('inf')  # Initialize best distance
        self.best_solution = None          # Initialize best solution
        self.best_distances_per_iteration = []  # Store best distances per iteration for comparison
        self.func_evals = func_evals #keeps track of functione evaluations

    def construct_solution(self):
        """
        Constructs a solution (tour) for one ant.

        Returns:
            List[int]: Sequence of node indices representing the tour.
        """
        solution = []
        visited = set()
        current_node = np.random.randint(0, self.num_nodes)
        solution.append(current_node)
        visited.add(current_node)
        while len(visited) < self.num_nodes:
            pheromone = self.pheromone[current_node]
            heuristic = 1 / (self.distance_matrix[current_node] + 1e-6)
            heuristic[heuristic == np.inf] = 0

            combined = (pheromone ** self.alpha) * (heuristic ** self.beta)
            for node in visited:
                combined[node] = 0  # Exclude visited nodes
            total = np.sum(combined)
            if total == 0:
                probabilities = np.ones(self.num_nodes)
                probabilities[list(visited)] = 0
                probabilities /= np.sum(probabilities)
            else:
                probabilities = combined / total
            next_node = np.random.choice(range(self.num_nodes), p=probabilities)
            solution.append(next_node)
            visited.add(next_node)
            current_node = next_node
        return solution

    def update_pheromone(self, solutions, distances):
        """
        Updates the pheromone matrix based on the solutions and their distances.

        Args:
            solutions (List[List[int]]): List of solutions (tours).
            distances (List[float]): Corresponding distances of the solutions.
        """
        self.pheromone *= (1 - self.evaporation_rate)  # Evaporation step
        for solution, distance in zip(solutions, distances):
            pheromone_contribution = self.Q / distance  # Higher pheromone contribution for better solutions
            for i in range(len(solution) - 1):
                from_node = solution[i]
                to_node = solution[i + 1]
                self.pheromone[from_node][to_node] += pheromone_contribution
            # Complete the tour (return to the start)
            self.pheromone[solution[-1]][solution[0]] += pheromone_contribution

    def calculate_total_distance(self, solution):
        """
        Calculates the total distance of a tour.

        Args:
            solution (List[int]): Sequence of node indices representing the tour.

        Returns:
            float: Total distance of the tour.
        """
        distance = 0
        for i in range(len(solution) - 1):
            distance += self.distance_matrix[solution[i]][solution[i + 1]]
        distance += self.distance_matrix[solution[-1]][solution[0]]  # Return to start
        self.func_evals += 1
        return distance

    def generate_training_data_using_aco(self, num_iterations=50):
        """
        Generates synthetic data by running the standard ACO algorithm.
        Returns sequences and corresponding pheromone matrices.

        Args:
            num_iterations (int): Number of iterations to simulate.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Generated sequences and pheromone matrices.
        """
        sequences = []
        pheromone_matrices = []

        for iteration in range(num_iterations):
            solutions = []
            distances = []
            for _ in range(self.num_ants):
                solution = self.construct_solution()
                distance = self.calculate_total_distance(solution)
                solutions.append(solution)
                distances.append(distance)
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_solution = solution
            self.update_pheromone(solutions, distances)
            self.best_distances_per_iteration.append(self.best_distance)
            print(f"Iteration {iteration+1}/{num_iterations}, Best Distance: {self.best_distance:.4f}")
            # Collect sequences and pheromone matrices
            sequences.extend(solutions)
            pheromone_matrices.extend(self.solutions_to_pheromone_matrix(solutions))

        return np.array(sequences), np.array(pheromone_matrices)

    def solutions_to_pheromone_matrix(self, solutions):
        """
        Converts a list of solutions (tours) into corresponding pheromone matrices.
        Each pheromone matrix has pheromone levels incremented for the edges in the solution.

        Args:
            solutions (List[List[int]]): List of solutions (tours).

        Returns:
            List[np.ndarray]: List of pheromone matrices corresponding to each solution.
        """
        pheromone_matrices = []
        for solution in solutions:
            pheromone = np.zeros((self.num_nodes, self.num_nodes))
            for i in range(len(solution) - 1):
                from_node = solution[i]
                to_node = solution[i + 1]
                pheromone[from_node][to_node] += 1
            # Complete the tour by connecting last to first node
            pheromone[solution[-1]][solution[0]] += 1
            # Normalize pheromone
            if pheromone.max() > 0:
                pheromone /= pheromone.max()
            pheromone_matrices.append(pheromone)
        return pheromone_matrices
    def optimize(self, iterations=100):
        """
        Runs the optimization process for a specified number of iterations.

        Args:
            iterations (int): Number of iterations to run.

        Returns:
            Tuple[List[int], float]: Best solution found and its distance.
        """
        best_distance = float('inf')
        best_solution = None
        for iteration in range(iterations):
            solutions = []
            distances = []
            for _ in range(self.num_ants):
                solution = self.construct_solution()
                distance = self.calculate_total_distance(solution)
                solutions.append(solution)
                distances.append(distance)
                if distance < best_distance:
                    best_distance = distance
                    best_solution = solution
            self.update_pheromone(solutions, distances)
            self.best_distances_per_iteration.append(best_distance)

        return best_solution, best_distance



def compute_distance_matrix(coordinates):
    """
    Computes the Euclidean distance matrix for the given coordinates.

    Args:
        coordinates (np.ndarray): Array of shape (num_nodes, 2).

    Returns:
        np.ndarray: Distance matrix of shape (num_nodes, num_nodes).
    """
    num_nodes = coordinates.shape[0]
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dx = coordinates[i][0] - coordinates[j][0]
                dy = coordinates[i][1] - coordinates[j][1]
                distance = np.sqrt(dx**2 + dy**2)
                distance_matrix[i][j] = distance
            else:
                distance_matrix[i][j] = np.inf  # To avoid self-loop in path
    return distance_matrix

def aco(matrix):
    func_evals = 0
    # Parameters
    num_nodes = 20
    num_ants = 20
    num_iterations = 50


    standard_aco = StandardAntColonyOptimizer(
        num_nodes=num_nodes,
        distance_matrix=matrix,
        num_ants=num_ants,
        alpha=1.0,
        beta=2.0,
        evaporation_rate=0.5,
        Q=100,
        func_evals = 0
    )

    params = {'num_nodes': num_nodes, 'num_ants': num_ants,
              'num_iterations': num_iterations, 'alpha': standard_aco.alpha, 'beta': standard_aco.beta,
              'evaporation_rate': standard_aco.evaporation_rate, 'Q': standard_aco.Q}

    best_solution_standard, best_distance_standard = standard_aco.optimize(iterations=num_iterations)
    print("\nRunning Standard ACO...")
    best_solution_standard, best_distance_standard = standard_aco.optimize(iterations=num_iterations)
    print(f"\nStandard ACO - Best Distance: {best_distance_standard:.4f}")
    print(f"Best solution: {best_solution_standard}")

    return {'func_evals': standard_aco.func_evals, 'sequence': best_solution_standard, 'parameters':params}


