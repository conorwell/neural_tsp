# neural_tsp
A class project for Prof. Scott's Block 2 2024 class, CP341: Deep Learning and Optimization


EXPERIMENT:

The goal of this project was to compare different optimization algorithms on solving the Traveling Salesperson Problem. We compiled a set of 29 traveling salesperson problems with a maxmimum size of 666 cities, a minimum size of 15 cities, and an average size of 143 cities. Most of these files were provided by the HeidelBerg TSPLIB. We tested each algorithm on all problems, analyzing the approximation ratio, runtime, distance evaluations, and memory allocation.

THE ALGORITHMS:

Tabu Search:
Tabu search is an enhancement on local neighbor search. It keeps track of a list of bad/invalid/visited solutions that it marks as "tabu" and will not check again.

Particle Swarm Optimization:



Ant Colony Optimization:
Ant colony optimisation algorithm is a swarm intelligence algorithm to optimise TSP which mimics the movement of ants towards their food. Simulated ants lay down pheromones on the paths they travel, which evaporate per an evaporation rate. These simulated ants initially move towards solutions randomly, but then follow a path with higher concentration of pheromone to a solution. A short path is marched over more frequently, and thus the pheromone density becomes higher on shorter paths than longer ones, enabling the algorithm to determine a global minimum.

Ant Colony Optimization with Transformer:
A transformer-enhanced ant colony optimization algorithm. The transformer is trained on different graphs and recognizes patterns to output an initial pheromone matrix optimal for any given graph

RESULTS:

Tabu Search:
It should be noted that this section is incomplete for Tabu search. Due to decreased scaling capabilities, Tabu Search was only tested on cities of size 48 or less. On these cities, Tabu scored the following:

Average Approximation Ratio: 1.36977
Average Memory Allocated: 2,077,634 bytes
Average Distance Checks: 34,881
Average Runtime: 76.8953 seconds

Plots:


Particle Swarm Optimization:

Average Approximation Ratio: 5.995
Average Memory Allocated: 271,454.48 bytes
Average Distance Checks: 4,020.6896
Average Runtime: 12.1956 seconds

Plots:

Ant Colony Optimization:

Average Approximation Ratio: 1.129
Average Memory Allocated: 25,721.4 bytes
Average Distance Checks: 2,000 (fixed)
Average Runtime: 132.67 seconds

Plots:

Ant Colony Optimization with Transformer:

Average Approximation Ratio: 1.12716
Average Memory Allocated: 290,193 bytes
Average Distance Checks: 2,000 (fixed)
Average Runtime: 103.899 seconds

Plots:
<img width="631" alt="Screenshot 2024-10-15 at 11 13 09 PM" src="https://github.com/user-attachments/assets/719fd07f-602f-46b6-b844-4f68c9b017a0">


Comparison Plots:



Interpretation:
From our tests, it can be observed that Ant Colony Optimization is the superior optimization method for the Traveling Salesperson Problem. ACO had the lowest approximation ratio while maintaining a reasonable runtime. Furthermore, Ant Colony Optimization with Transformers demonstrated a similar approximation ratio while reducing runtime.
















