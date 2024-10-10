import torch
import numpy as np

def fitness(distances, pos): 
    """
    returns the total length of the tour
    """
    total_length = 0

    pos = pos_to_route(pos)

    for i in range(len(pos) - 1):
        total_length += distances[pos[i], pos[i+1]]

    return total_length

def pos_to_route(pos): 

    # make identity list
    identity = [x for x in range(len(pos))]

    # turn pos into a route
    pos = sorted(identity, key=lambda x: pos[x])
    return pos

def update_pos(prev_pos, vel): 
    return prev_pos + vel

def update_vel(prev_vel, prev_pos, pbest, gbest, w, c1, c2): 
    r1 = np.random.random()
    r2 = np.random.random()

    return w*prev_vel + (c1*r1*(pbest - prev_pos)) + (c2*r2*(gbest - prev_pos))

def pso(distances, n_particles=10, w=.5, c1=1, c2=0.3): 
    """
    performs pso
    """

    # initialize random positions and velocities
    positions = torch.rand(n_particles, distances.shape[0])
    velocities = torch.rand(n_particles, distances.shape[0]) * 2 - 1
    print(velocities)
    count = 0

    # softmax all the positions
    positions = torch.nn.functional.softmax(positions, dim=1)

    # initialize pbest and gbest
    pbest = fitness(distances, positions[0])
    gbest = fitness(distances, positions[0])
    gbest_pos = positions[0]
    count += 2

    # neural net goes here?

    # perform pso iterations
    for i in range(100): 

        # loop thru each particle
        for j in range(n_particles): 

            prev_pos = positions[j]
            prev_vel = velocities[j]           

            # update velocity
            vel = update_vel(prev_vel, prev_pos, pbest, gbest, w, c1, c2)
            velocities[j, :] = vel

            # update position
            pos = update_pos(prev_pos, vel)
            pos = torch.nn.functional.softmax(pos, dim=0)
            positions[j, :] = pos

            # calculate fitness
            fpos = fitness(distances, pos)
            count += 1

            # update pbest if needed
            if fpos < pbest: 
                pbest = fpos

            # update gbest if needed
            if fpos < gbest: 
                gbest = fpos
                gbest_pos = pos

    return {"sequence": gbest_pos, "func_evals": count, "parameters": (n_particles, w, c1,c2)}


def main(): 
    
   matrix = np.array([[0,15,2,34,1],[15,0,48,2,17],[2,48,0,22,39],[34,2,22,0,3],[1,17,39,3,0]])
   distances = torch.Tensor(matrix)
   
   solution= pso(distances, 10, 0.8, 1, 1)
   print(pos_to_route(solution["sequence"]))
   print(fitness(distances, solution["sequence"]).item())

if __name__ == "__main__": 
    main()