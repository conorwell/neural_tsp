import torch
import numpy as np

def fitness(distances, pos): 
    """
    returns the total length of the tour
    """
    total_length = 0
    pos = pos_to_route(pos)

    for i in range(len(pos) - 1):
        total_length += distances[pos[i], pos[i+1]].item()

    return total_length

def pos_to_route(pos): 

    # make identity list
    identity = [x for x in range(len(pos))]

    # turn pos into a route
    pos = sorted(identity, key=lambda x: pos[x])
    pos = torch.tensor(pos)
    return pos

def update_pos(positions, velocities): 
    return positions + velocities

def update_vel(distances, velocities, positions, pbests_pos, gbest_pos, w, c1, c2): 

    r1 = torch.rand(distances.shape[0])
    r2 = torch.rand(distances.shape[0])

    # possibly loses some pbest info
    return w*velocities + (c1*r1*(pbests_pos - positions)) + (c2*r2*(gbest_pos - positions))

def pso(distances, n_particles=25, w=0.9, c1=0.5, c2=0.5, stagnation_limit=50): 
    """
    performs pso
    """

    # initialize random positions and velocities
    positions = torch.rand(n_particles, distances.shape[0]) * 2 - 1
    velocities = torch.rand(n_particles, distances.shape[0]) * 2 - 1
    
    # initialize fitnesses
    fitnesses = torch.Tensor([fitness(distances, pos) for pos in positions])
    count = positions.shape[0] # initialize fitness function call count

    # initialize pbests
    pbests = fitnesses.clone()
    pbests_pos = positions.clone()

    # initialize gbest
    gbest_idx = torch.argmin(pbests).item()
    gbest = pbests[gbest_idx].item()
    gbest_pos = pbests_pos.clone()[gbest_idx, :]

    # initialize stagnation count 
    stagnation_count = 0

    # perform pso iterations
    for _ in range(1000): 

        # update velocities and keep within a range
        velocities = update_vel(distances, positions, velocities, pbests_pos, gbest_pos, w, c1, c2)
        velocities = torch.nn.functional.softmax(velocities, dim=1)
        # velocities = torch.clamp(velocities, min=-500, max=500)

        # update positions
        positions = update_pos(positions, velocities)

        # calculate fitnesses
        fitnesses = torch.Tensor([fitness(distances, pos) for pos in positions])
        count += positions.shape[0] # increment fitness function call count

        # update pbest if needed
        pbests = torch.where(fitnesses < pbests, fitnesses, pbests)
        pbests_pos = torch.where(fitnesses.reshape(n_particles, 1) < pbests.reshape(n_particles, 1), positions, pbests_pos)

        # update gbest if needed
        min_fitnesses_idx = torch.argmin(fitnesses).item()
        min_fitness = fitnesses[min_fitnesses_idx].item()
        if min_fitness < gbest:
            gbest = min_fitness
            gbest_pos = positions.clone()[min_fitnesses_idx, :]
            stagnation_count = 0
        else: 
            stagnation_count += 1

        if stagnation_count >= stagnation_limit:
            print("Stopped for stagnation!")
            break

        # print(pos_to_route(gbest_pos), "GBEST")
        # for pos in positions: 
        #     print(pos_to_route(pos))

    return {"gbest": gbest, "sequence": pos_to_route(gbest_pos), "func_evals": count, "parameters": {"n_particles": n_particles, "w": w, "c1": c1, "c2": c2}}


def main(): 

   matrix = np.array([[0,10,5,20,6,32,6,14],[10,0,6,2,31,5,18,1],[5,6,0,10,4,21,6,37],[20,2,10,0,9,10,7,16],[6,31,4,9,0,26,5,39],[32,5,21,10,26,0,3,21],[6,18,6,7,5,3,0,47],[14,1,37,16,39,21,47,0]])
   distances = torch.Tensor(matrix)
   
   solution = pso(distances, 200, .2, 1, 3)
   print("Best found route:", solution["sequence"])
   print("Route distances:", solution["gbest"])
   print("Fitness function evals:", solution["func_evals"])

if __name__ == "__main__": 
    main()