# path = [city1, city3, city4, city2, city1]
# citylist = [city1, city2, city3, city4]
import torch
import numpy as np

path = ['city1', 'city2', 'city3', 'city4']
citylist = ['city1', 'city2', 'city3', 'city4']

arr = [
    [0,2,3,6],
    [2,0,4,1],
    [3,4,0,2],
    [6,1,2,0]
    ]

np.array(arr)
torch.Tensor(arr)

# finds fitness of path
def fitness(mat, path):
    sum = 0
    newpath = path.copy()
    newpath.append(path[0])
    for i in range(len(newpath)-1):
        start = citylist.index(newpath[i])
        end = citylist.index(newpath[i+1])
        if mat[start][end] == 0:
            return 0
        else:
            sum += mat[start][end]
    return sum

# creates list of paths 1 swap away from current path
def alter(path):
    alteredlist = []
    # print("original path: ",path)
    for i in range(len(path)-1):
        for j in range(i+1, len(path)):
            alteredpath = path.copy()
            alteredpath[i], alteredpath[j] = alteredpath[j],alteredpath[i]
            alteredlist.append(alteredpath)
    return alteredlist

def tabu_search(mat, max_iters, tabu_list_size, initial):
    bestpath = initial
    perm_list = []
    bestfit = fitness(mat, bestpath)
    tabu_list = []
    for i in range(max_iters):
        altered = alter(bestpath)
        for newpath in altered:
            newfit = fitness(mat, newpath)
            #check if path is in tabulist
            if newpath in tabu_list or newpath in perm_list:
                continue
            else:
                # bad path is found
                if newfit == 0:
                    perm_list.append(newpath)
                    continue
                # need to tune accept condition
                elif newfit <= bestfit * 1.01:
                    bestpath = newpath
                    bestfit = newfit
                # worse path is found
                else:
                    if len(tabu_list) >= tabu_list_size:
                        tabu_list.pop(0)
                    tabu_list.append(newpath)
                    continue
    bestpath.append(bestpath[0])
    return bestpath

tabu_search(arr, 1, 6, path)