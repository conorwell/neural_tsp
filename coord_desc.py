import torch
import numpy as np

from pso import pso

def generateTSP(dim):
    tsp = np.random.randint(25,size=(dim,dim))
    tsptrans = tsp.transpose()
    newtsp = tsp+tsptrans
    for i in range(newtsp.shape[0]):
        newtsp[i][i] = 0
    return newtsp

phi = (-1 + 5**.5)/2

def gss(f, x0, x3, eps=1e-6):
    x1 = (1-phi)*(x3 - x0) + x0
    x2 = (phi)*(x3 - x0) + x0

    f0, f1, f2, f3 = [f(x) for x in [x0, x1, x2, x3]]

    if x3 - x0 < 2*eps:
        min_idx = sorted([0,1,2,3], key= lambda x: [f0,f1,f2,f3][x])[0]
        min_x = [x0,x1,x2,x3][min_idx]
        # min_f = [f0,f1,f2,f3][min_idx]
        return min_x

    if f1 < f2:
        return gss(f, x0, x2)
    else:
        return gss(f, x1, x3)
    

def pso_avg(n_particles, w, c1, c2): 

    total = 0
    n_problems = 5
    for i in range(n_problems): 
        distances = generateTSP(5)
        result = pso(distances, n_particles, w, c1, c2)
        total += result['gbest']

    return total / n_problems
 
def coord_desc(f, x0, eps=1e-6):

    x_old = np.inf*x0.copy()
    n_particles = 100

    while np.linalg.norm(x0 - x_old) > eps:
        print("new while")
        x_old = x0.copy()
        w, c1, c2 = x0
        for i in range(3):
            print("i:", i)
            if i == 0:
                # optimize x, holding y constant
                func = lambda var: f(n_particles, var, c1, c2)
                result = gss(func, .75, 1.0)
            elif i == 1:
                func = lambda var: f(n_particles, w, var, c2)
                result = gss(func, .2, 5.0)
            else:
                func = lambda var: f(n_particles, w, c1, var)
                result = gss(func, .2, 5.0)
            
            x0[i] = result
            print("result:",x0)

    return x0

def main(): 
    print(coord_desc(pso_avg, np.array([1.0, 1.0, 1.0])))

if __name__ == '__main__': 
    main()