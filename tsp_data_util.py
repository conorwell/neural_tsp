import tsplib95 as tsp
import numpy as np
import os

optimal_tour_distance = {
'a280' : 2579,
'ali535' : 202339,
'att48' : 10628,
'att532' : 27686,
'bayg29' : 1610,
'bays29' : 2020,
'berlin52' : 7542,
'bier127' : 118282,
'brazil58' : 25395,
'brd14051' : 469385,
'brg180' : 1950,
'burma14' : 3323,
'ch130' : 6110,
'ch150' : 6528,
'd198': 15780,
'd493' : 35002,
'd657' : 48912,
'd1291' : 50801,
'd1655' : 62128,
'd2103' : 80450,
'd15112' : 1573084,
'd18512' : 645238,
'dantzig42' : 699,
'dsj1000' : 18660188,
'eil51' : 426,
'eil76' : 538,
'eil101' : 629,
'fl417' : 11861,
'fl1400' : 20127,
'fl1577' : 22249,
'fl3795' : 28772,
'fnl4461' : 182566,
'fri26' : 937,
'gil262' : 2378,
'gr17' : 2085,
'gr21' : 2707,
'gr24' : 1272,
'gr48' : 5046,
'gr96' : 55209,
'gr120' : 6942,
'gr137' : 69853,
'gr202' : 40160,
'gr229' : 134602,
'gr431' : 171414,
'gr666' : 294358,
'hk48' : 11461,
'kroA100' : 21282,
'kroB100' : 22141,
'kroC100' : 20749,
'kroD100' : 21294,
'kroE100' : 22068,
'kroA150' : 26524,
'kroB150' : 26130,
'kroA200' : 29368,
'kroB200' : 29437,
'lin105' : 14379,
'lin318' : 42029,
'linhp318': 41345,
'nrw1379' : 56638,
'p654' : 34643,
'pa561' : 2763,
'pcb442' : 50778,
'pcb1173' : 56892,
'pcb3038' : 137694,
'pla7397' : 23260728,
'pla33810': 66048945,
'pla85900' : 142382641,
'pr76' : 108159,
'pr107' : 44303,
'pr124' : 59030,
'pr136' : 96772,
'pr144' : 58537,
'pr152' : 73682,
'pr226' : 80369,
'pr264' : 49135,
'pr299' : 48191,
'pr439' : 107217,
'pr1002' : 259045,
'pr2392' : 378032,
'rat99' : 1211,
'rat195' : 2323,
'rat575' : 6773,
'rat783' : 8806,
'rd100' : 7910,
'rd400' : 15281,
'rl1304' : 252948,
'rl1323' : 270199,
'rl1889' : 316536,
'rl5915' : 565530,
'rl5934' : 556045,
'rl11849' : 923288,
'si175' : 21407,
'si535' : 48450,
'si1032' : 92650,
'st70' : 675,
'swiss42' : 1273,
'ts225' : 126643,
'tsp225' : 3916,
'u159' : 42080,
'u574' : 36905,
'u724' : 41910,
'u1060' : 224094,
'u1432' : 152970,
'u1817' : 57201,
'u2152' : 64253,
'u2319' : 234256,
'ulysses16' : 6859,
'ulysses22' : 7013,
'usa13509': 19982859,
'vm1084' : 239297,
'vm1748' : 336556,
}
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

def get_optimal_tour(tsp_file):
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

def get_optimal_tour_length(tsp_file):
    '''get the optimal tour length of a given tsp file'''
    split = tsp_file.split('.')
    problem_name = split[0]
    return optimal_tour_distance[problem_name]



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

    opt_solution_tour = get_optimal_tour(tsp_file)
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
