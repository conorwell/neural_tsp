import numpy as np
import torch
import random
import time
import tsp_data_util as tsp_data_util
import tsplib95
import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import gc
import os

class ImprovedAntColonyOptimizer:
    """
    Enhanced ACO algorithm where the initial pheromone matrix is predicted by the Transformer model.
    """
    def __init__(self, num_nodes, distance_matrix, initial_pheromone=None, num_ants=10, alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100,
                func_evals = 0 ):
        self.num_nodes = num_nodes
        self.distance_matrix = distance_matrix
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        if initial_pheromone is not None:
            self.pheromone = initial_pheromone  # Use the provided initial pheromone matrix
        else:
            self.pheromone = torch.ones((num_nodes, num_nodes))  # Initialize pheromone levels uniformly
        self.best_distance = float('inf')
        self.best_solution = None
        self.best_distances_per_iteration = []  # Store best distances per iteration for comparison
        self.func_evals = func_evals  # keeps track of functione evaluations

    def construct_solution(self):
        """
        Constructs a solution (tour) for one ant.

        Returns:
            List[int]: Sequence of node indices representing the tour.
        """
        solution = []
        visited = set()
        current_node = np.random.randint(0, self.num_nodes)
        print(current_node)
        solution.append(current_node)
        visited.add(current_node)
        while len(visited) < self.num_nodes:
            pheromone = self.pheromone[current_node]
            heuristic = 1 / (self.distance_matrix[current_node] + 1e-6)
            heuristic[heuristic == np.inf] = 0
            print(self.alpha, self.beta)
            print('alpha', pheromone ** self.alpha, 'beta', heuristic ** self.beta)

            combined = (pheromone ** self.alpha) * (heuristic ** self.beta)
            for node in visited:
                combined[node] = 0  # Exclude visited nodes
            total = torch.sum(combined)
            if total == 0:
                probabilities = torch.ones(self.num_nodes)
                probabilities[list(visited)] = 0
                probabilities /= torch.sum(probabilities)
            else:
                probabilities = combined / total

            nodes = torch.arange(self.num_nodes)

            # Convert probabilities to a PyTorch tensor if it isn't already
            probabilities_tensor = torch.tensor(probabilities, dtype=torch.float32)
            # Normalize probabilities if they don't sum to 1 (just in case)
            probabilities_tensor = probabilities_tensor / probabilities_tensor.sum()
            # Use torch.multinomial to sample from the nodes based on probabilities
            next_node_index = torch.multinomial(probabilities_tensor, 1)  # Sample one node
            # Get the actual node index
            next_node = nodes[next_node_index.item()].item()

            #Get the chosen values
            #chosen_values = choices[chosen_indices]

            #next_node = np.random.choice(range(self.num_nodes), p=probabilities)
            solution.append(next_node)
            visited.add(next_node)
            current_node = next_node
        return torch.tensor(solution)

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
            distance += self.distance_matrix[solution[i],solution[i + 1]]
        distance += self.distance_matrix[solution[-1], solution[0]]  # Return to start
        self.func_evals += 1
        return distance



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
            pheromone = torch.zeros((self.num_nodes, self.num_nodes))
            for i in range(len(solution) - 1):
                from_node = solution[i]
                to_node = solution[i + 1]
                pheromone[from_node, to_node] += 1
            # Complete the tour by connecting last to first node
            pheromone[solution[-1], solution[0]] += 1
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

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)




# Positional Encoding for Transformer Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerModel(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=6, dropout=0.1, learnable_pos=False):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model) if not learnable_pos else nn.Parameter(
            torch.zeros(1, d_model, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_projection = nn.Linear(d_model, d_model)
        self.final_activation = nn.ReLU()

    def forward(self, x, mask=None):
        batch_size, num_nodes, _ = x.size()  # Extract the number of nodes dynamically

        # Dynamically set the input projection layer to handle num_nodes
        self.input_projection = nn.Linear(num_nodes, self.d_model).to(x.device)
        x = self.input_projection(x) * np.sqrt(self.d_model)

        # Adjust positional encoding to the number of nodes
        x = self.pos_encoder(x)[:, :num_nodes]

        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.output_projection(x)
        pheromone_matrix = torch.matmul(x, x.transpose(1, 2))
        pheromone_matrix = self.final_activation(pheromone_matrix)
        return pheromone_matrix
def pheremone_using_nn(distance_matrix, model_path='best_transformer_model_final.pth'):
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_nodes = len(distance_matrix)
    model = TransformerModel(d_model=128, nhead=8, num_layers=6, dropout=0.1, learnable_pos=False).to(device)

    # Load state dict without loading positional encodings
    if device.type == "cpu":
        model_state = torch.load(model_path, map_location=torch.device('cpu') )
    else:
        model_state = torch.load(model_path)

    model.load_state_dict(model_state, strict=False)  # strict=False will allow skipping unmatched keys

    model.eval()

    distance_matrix = distance_matrix.unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_pheromone = model(distance_matrix.float())[:, :distance_matrix.size(1), :distance_matrix.size(2)]

    predicted_pheromone_np = predicted_pheromone.squeeze(0).cpu().numpy()
    predicted_pheromone_np -= predicted_pheromone_np.min()
    max_val = predicted_pheromone_np.max()
    predicted_pheromone_np /= max_val if max_val > 0 else 1
    predicted_pheromone_np += 1e-6
    return predicted_pheromone_np

def aco_nn(matrix):

    #get the pheremone matrix

    #give it to improvedaco

    initial_pheromone = pheremone_using_nn(matrix, model_path='best_transformer_model_final.pth')
    print('INITIAL PHEROMONE', initial_pheromone)
    # Parameters
    num_ants = 20
    num_iterations = 100

    improved_aco = ImprovedAntColonyOptimizer(
        num_nodes=matrix.size(0),
        distance_matrix=matrix,
        num_ants=num_ants,
        alpha=1.0,
        beta=2.0,
        evaporation_rate=0.5,
        Q=100,
        func_evals = 0,
        initial_pheromone = torch.tensor(initial_pheromone)
    )

    params = {'num_nodes': matrix.size(0), 'num_ants': num_ants,
              'num_iterations': num_iterations, 'alpha': improved_aco.alpha, 'beta': improved_aco.beta,
              'evaporation_rate': improved_aco.evaporation_rate, 'Q': improved_aco.Q}

    #best_solution_standard, best_distance_standard = improved_aco.optimize(iterations=num_iterations)
    print("\nRunning Improved ACO...")
    best_solution_standard, best_distance_standard = improved_aco.optimize(iterations=num_iterations)
    print(f"\nImproved ACO - Best Distance: {best_distance_standard:.4f}")
    print(f"Best solution: {best_solution_standard}")

    return {'func_evals': improved_aco.func_evals, 'sequence': best_solution_standard, 'parameters':params}
