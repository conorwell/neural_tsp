# neural_tsp
A class project for Prof. Scott's Block 2 2024 class, CP341: Deep Learning and Optimization

EXPERIMENT:

The goal of this project was to compare different optimization algorithms on solving the Traveling Salesperson Problem. We compiled a set of 29 traveling salesperson problems with a maxmimum size of 666 cities, a minimum size of 15 cities, and an average size of 143 cities. Most of these files were provided by the HeidelBerg TSPLIB. We tested each algorithm on all problems, analyzing the approximation ratio, runtime, distance evaluations, and memory allocation.

THE ALGORITHMS:

Tabu Search:
Tabu search is an enhancement on local neighbor search. It keeps track of a list of bad/invalid/visited solutions that it marks as "tabu" and will not check again.

Particle Swarm Optimization:
We implemented a discrete adaptation of particle swarm optimization (pso) to approximate answers for tsp problems. We attempted to find the optimal w, c1, and c2 parameters for our pso implementation by first using coordinate descent with golden section search. After running into convergence issues with this method, we created graphs of each parameter being tweaked across averages of tsp problems while keeping the other parameters constant. 


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
<img width="638" alt="Screenshot 2024-10-15 at 10 05 44 PM" src="https://github.com/user-attachments/assets/800af325-614d-412d-adfb-1c537997e853">

Particle Swarm Optimization:

Average Approximation Ratio: 5.995
Average Memory Allocated: 271,454.48 bytes
Average Distance Checks: 4,020.6896
Average Runtime: 12.1956 seconds

Plots:
(with outlier)
<img width="640" alt="Screenshot 2024-10-15 at 9 30 50 PM" src="https://github.com/user-attachments/assets/a6bf1a30-503d-45df-95fe-91af1db60294">

(outlier removed)
<img width="636" alt="Screenshot 2024-10-15 at 9 31 03 PM" src="https://github.com/user-attachments/assets/ec84d72f-cb20-4fff-ba0a-d8bcbeb12581">

Ant Colony Optimization:

Average Approximation Ratio: 1.129
Average Memory Allocated: 25,721.4 bytes
Average Distance Checks: 2,000 (fixed)
Average Runtime: 132.67 seconds

Plots:
<img width="651" alt="Screenshot 2024-10-15 at 9 25 33 PM" src="https://github.com/user-attachments/assets/05882878-6f9b-4509-8363-6986be039dbe">

Ant Colony Optimization with Transformer:

Average Approximation Ratio: 1.12716
Average Memory Allocated: 290,193 bytes
Average Distance Checks: 2,000 (fixed)
Average Runtime: 103.899 seconds

Plots:
<img width="631" alt="Screenshot 2024-10-15 at 11 13 09 PM" src="https://github.com/user-attachments/assets/719fd07f-602f-46b6-b844-4f68c9b017a0">


Comparison Plots:

Approximation Ratio:
![Runtimecomparisons](https://github.com/user-attachments/assets/cea161ab-9df3-4cdb-a1ae-8297abc4043e)
![Func_Evalscomparisons](https://github.com/user-attachments/assets/bfdf5bd7-6f17-40a9-abc0-69d6cbdfa49b)
![Memorycomparisons](https://github.com/user-attachments/assets/a759f879-86e5-420f-b9fa-8d560c2cf3e5)
![ARcomparisons](https://github.com/user-attachments/assets/71e77d9c-e42c-4c7f-9062-fd93291134b1)



Interpretation:
From our tests, it can be observed that Ant Colony Optimization is the superior optimization method for the Traveling Salesperson Problem. ACO had the lowest approximation ratio while maintaining a reasonable runtime. Furthermore, Ant Colony Optimization with Transformers demonstrated a similar approximation ratio while reducing runtime.
