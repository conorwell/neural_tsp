import torch
import numpy as np
import matplotlib.pyplot as plt
from pso import pso

def generateTSP(dim):
    tsp = np.random.randint(25,size=(dim,dim))
    tsptrans = tsp.transpose()
    newtsp = tsp+tsptrans
    for i in range(newtsp.shape[0]):
        newtsp[i][i] = 0
    return newtsp

def set_problems(n_problems, prob_size): 
    distance = []
    for i in range(n_problems):
        matrix = generateTSP(prob_size)
        distances = torch.Tensor(matrix)
        distance.append(distances)

    return distance

def graph_pso(n_particles, n_problems, prob_size):

    # make the tsp problems
    distance = set_problems(n_problems, prob_size)
    
    plt.subplot(3, 1, 1)
    wval = np.arange(0.1, 1.0, 0.01)
    wy = np.array([pso(distance[0], n_particles, x, 1, 1)['gbest'] for x in wval])
    wyavg = wy
    for i in range(n_problems-1):
        print('w:', i)
        wy = np.array([pso(distance[i+1], n_particles, x, 1, 1)['gbest'] for x in wval])
        wyavg+=wy
    wyavg /= n_problems
    plt.plot(wval,wyavg)
    plt.title("w vs. Avg. Fitness")
    plt.xlabel('w')
    plt.ylabel('Avg. Fitness')

    plt.subplot(3, 1, 2)
    c1val = np.arange(0.1, 5.0, 0.1)
    c1y = np.array([pso(distance[0], n_particles, 1, x, 1)['gbest'] for x in c1val])
    c1avg = c1y
    for i in range(n_problems-1):
        print('c1:', i)
        c1y = np.array([pso(distance[i+1], n_particles, 1, x, 1)['gbest'] for x in c1val])
        c1avg+=c1y
    c1avg /= n_problems
    plt.plot(c1val,c1avg)
    plt.title("c1 vs. Avg. Fitness")
    plt.xlabel('c1')
    plt.ylabel('Avg. Fitness')

    plt.subplot(3, 1, 3)
    c2val = np.arange(0.1, 5.0, 0.1)
    c2y = np.array([pso(distance[0], n_particles, 1, 1, x)['gbest'] for x in c2val])
    c2avg = c2y
    for i in range(n_problems-1):
        print('c2', i)
        c2y = np.array([pso(distance[i+1], n_particles, 1, 1, x)['gbest'] for x in c2val])
        c2avg+=c2y
    c2avg /= n_problems
    plt.plot(c2val,c2avg)
    plt.title("c2 vs. Avg. Fitness")
    plt.xlabel('c2')
    plt.ylabel('Avg. Fitness')

    plt.subplots_adjust(hspace=1)
    plt.show()

def main(): 
    # n_particles, n_problems, prob_size
    graph_pso(100, 100, 10)

if __name__ == "__main__":
    main()