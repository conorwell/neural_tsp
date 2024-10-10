import tsplib95 as tsp
import numpy as np
import os

def generate_edge_weight_matrix(tsp_file):
    ''' Given a tsp file with the problem stored as coordinates, matrices, etc.
    return a edge weight matrix.'''

    problem = tsp.load(tsp_file)
    num_nodes = problem.dimension
    # Create an edge weight matrix
    weight_matrix = np.zeros((num_nodes, num_nodes))

    for i, node1 in enumerate(problem.get_nodes()):
        for j, node2 in enumerate(problem.get_nodes()):
            weight_matrix[i][j] = problem.get_weight(node1, node2)

    return weight_matrix

def get_optimal_solution(tsp_file):
    '''return the optimal solution of a tsp file (IF IT EXISTS)
    otherwise return [] and print "no optimal solution file exists"'''
    split_path = tsp_file.split(".")

    #ensure that a compatible tsp file was given
    if len(split_path) != 2 or split_path[1] != "tsp":
        raise Exception("Incompatible .tsp file | can not pass .opt.tour files to this function")

    opt_solution_path = split_path[0] + ".opt.tour"
    if os.path.exists(opt_solution_path): #when there is a solution
        solution = tsp.load(opt_solution_path)
        return solution.tours[0]
    else: #when there is no solution file
        print("no solution file exists for the given tsp problem")
        return []


def get_tour_length(tsp_file, tour):
    ''' Given a tour, get the length of following the sequence
    Note: a tour needs to be in the format of [[tour]] not [tour]'''
    p = tsp.load(tsp_file)
    return p.trace_tours([tour])[0]


if __name__ == '__main__':
    '''Test code: 
        - gets an edge_weight_matrix
        - gets the optimal solution 
        - gets the tour length of optimal solution
        - The length matches the optimal length!!
    
    '''
    tsp_file = 'heidelberg_TSP_data/ch150.tsp'  # coord matrix

    weight_matrix = generate_edge_weight_matrix(tsp_file)
    print("Edge Weight Matrix:")
    print(weight_matrix)

    opt_solution_tour = get_optimal_solution(tsp_file)
    print(opt_solution_tour)

    opt_solution_length = get_tour_length(tsp_file, opt_solution_tour)
    print(opt_solution_length)
    #











''' TSPLIB95 REFERENCE FUNCTIONS
# loading in a given tsp file
problem = tsp.load('ALL_tsp/att48.tsp')

#getting the nodes of the tsp problem
problem.get_nodes()

#gets all of the edges from a tsp problem.
#returns tuple pairs of all of the edges

print(problem.get_edges())

#accessing variables of TSP file
#all variables are accessible using problem.variableName
print(problem.render())
print(problem.dimension)

# represent all elements as a dictionary
problem.as_name_dict()

#distances
print(problem.node_coords[1])
print(problem.node_coords[2])
edge = 8,9
print("Euclidean distance: ", math.dist(problem.node_coords[1],problem.node_coords[2]))
print("tsplib edgeweight value: ", problem.get_weight(1, 2))
'''
