import torch
import numpy as np

path = [1,2,3,4]
citylist = [1,2,3,4]

arr = [
    [0,2,3,6],
    [2,0,4,1],
    [3,4,0,2],
    [6,1,2,0]
    ]

arr = np.array(arr)
arr = torch.Tensor(arr)

# finds fitness of path
def fitness(mat, path):
    # sum = 0
    # newpath = path.clone()
    # last_node = torch.tensor([path[0]])
    # newpath = torch.cat((newpath, last_node), 0)
    newpath = torch.cat((path, path[:1]), 0) #full circle path
    start_nodes = newpath[:-1]-1
    end_nodes = newpath[1:]-1
    weights = mat[start_nodes][end_nodes]

    if torch.any(weights==0): #if any of the paths are 0 (impossible)
        return 0 
    return weights.sum().item()
    # for i in range(len(newpath)-1):
    #     start = newpath[i]-1
    #     end = newpath[i+1]-1
    #     if mat[start][end] == 0:
    #         return 0
    #     else:
    #         sum += mat[start][end]
    # return sum

# creates list of paths 1 swap away from current path
def alter(path):
    alteredlist = []
    for i in range(len(path)-1):
        for j in range(i+1, len(path)):
            alteredpath = path.clone()
            temp = int(alteredpath[i])
            alteredpath[i] = alteredpath[j]
            alteredpath[j] = temp
            alteredlist.append(alteredpath)
    return alteredlist

def sortedTechnique(e_matrix):
    res = torch.tensor(range(1,e_matrix.shape[0]+1))
    return res

def tabu_search(mat, max_iters=100, worsening_thresh=1.01):
    num_cities = mat.size(dim = 0)
    func_evals = 0
    tabu_list_size = num_cities * (num_cities-1) / 2
    initial = sortedTechnique(mat)
    bestpath = initial
    perm_list = []
    bestfit = fitness(mat, bestpath)
    tabu_list = []
    for i in range(max_iters):
        altered = alter(bestpath)
        for newpath in altered:
            #check if path is in tabulist
            #if newpath in tabu_list or newpath in perm_list:
            if any([(newpath == short).all() for short in tabu_list]) or any([(newpath == long).all() for long in perm_list]):
                continue
            else:
                newfit = fitness(mat, newpath)
                func_evals +=1
                # bad path is found
                if newfit == 0:
                    perm_list.append(newpath)
                    continue
                # need to tune accept condition
                elif newfit <= bestfit * worsening_thresh:
                    bestpath = newpath
                    bestfit = newfit
                # worse path is found
                else:
                    if len(tabu_list) >= tabu_list_size:
                        tabu_list.pop(0)
                    tabu_list.append(newpath)
                    continue
    last_node = torch.tensor([bestpath[0]])
    bestpath = torch.cat((bestpath, last_node), 0)
    params = {'tabu_list_size': tabu_list_size, 'max_iterations': max_iters, 'worsening_threshold': worsening_thresh}
    return {'func_evals': func_evals, 'sequence': bestpath, 'parameters':params}

print(tabu_search(arr))