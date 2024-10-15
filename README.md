# neural_tsp
A class project for Prof. Scott's Block 2 2024 class, CP341: Deep Learning and Optimization

## PSO for TSP
We implemented a discrete adaptation of particle swarm optimization (pso) to approximate answers for tsp problems. We attempted to find the optimal w, c1, and c2 parameters for our pso implementation by first using coordinate descent with golden section search. After running into convergence issues with this method, we created graphs of each parameter being tweaked across averages of tsp problems while keeping the other parameters constant. 
