import os
import sys
import time
import argparse

import numpy as np
import pandas as pd

from docplex.mp.model import Model

### LOAD UNSEEN CLASSES MATRIX 
def load_classes(dataset_folder, method, reduced, num_sub='all'):
    with open(f"{dataset_folder}/nome_classi.txt") as file:
        lines = file.readlines()
        all_classes = [line.rstrip() for line in lines]

    if method == 'trainval':
        if reduced == 'no':
            with open(f"{dataset_folder}/unseen_classi.txt") as file:
                lines = file.readlines()
                unseen_classes = [line.rstrip() for line in lines]
            
            pick_classes = [c for c in all_classes if c not in unseen_classes]
        elif reduced == 'yes':
            with open(f"{dataset_folder}/subset_{num_sub}.txt") as file:
                lines = file.readlines()
                pick_classes = [int(line.rstrip()) for line in lines]
    
    elif method == 'val':
        with open(f"{dataset_folder}/seen_classi.txt") as file:
            lines = file.readlines()
            pick_classes = [line.rstrip() for line in lines]
    
    elif method == 'test':
        with open(f"{dataset_folder}/unseen_classi.txt") as file:
            lines = file.readlines()
            pick_classes = [line.rstrip() for line in lines]

    return all_classes, pick_classes

### LOAD MATRIX
def load_class_attribute_matrix(all_classes, unseen_classes, dataset_folder, method=None):
    if method == 'test':
        what = [all_classes.index(u) for u in unseen_classes]
    elif method == 'trainval':
        what = [all_classes.index(u) for u in unseen_classes]
    elif method == 'val':
        what = [int(i) for i in unseen_classes]
    
    M = pd.read_pickle(f"{dataset_folder}/class_attribute_matrix.pickle")
    M = np.array(M)
    M = M[what,:]
    return M

def solve_adversarial_perfect_detectors(M):

    m = Model(name="Q-function")
    
    k, n = M.shape

    K = range(k)
    V = range(2**n)
    #M = np.array([[0.8,0.5],[0.3,0.4]])#np.random.rand(k,n)# #np.random.rand(k,n)

    P = m.continuous_var_matrix(k,2**n,lb=0, name='P')
    delta = m.continuous_var_list(2**n, name='Delta')

    s =  time.time()

    m.add_constraint( m.sum(P) == 1)
    print(f'Constraint 0: {time.time()-s}')
    s =  time.time()  

    m.add_constraints((m.sum(P[j,v] for v in  V) == 1/k) for j in range(k))
    print(f'Constraint 1: {time.time()-s}')

    s =  time.time()

    in_v_list_j = [[v for v in V if( (v & (2**i)) > 0)] for i in range(n)]
    indices = [(j,i) for j in range(k) for i in range(n)]

    m.add_constraints((m.sum(P[j,l] for l in in_v_list_j[i]) == M[j,i]*1/k ) for j,i in indices)
    print(f'Constraint 2: {time.time()-s}')

    s =  time.time()

    for j in range(k):
        m.add_constraints( (delta[v] - P[j,v] >= 0) for v in range(2**n))
    print(f'Constraint 3: {time.time() - s}')

    m.minimize(sum(delta[i] for i in V))
    m.solve()

    print(m.print_information())

    print(f'Solution: {m.solution.get_objective_value()}')#print_solution())
    #print(m.report())
    print(f'All: {time.time() - s}')

    return m.solution.get_objective_value(), None

def solve_next(M, indices_current, output_path, method, reduced, num_subset):

    start = time.time()

    #len(indices_return) = len(indices_current) + 1

    k_prev, n_prev = M.shape

    indices_to_try = [i for i in range(n_prev) if i not in indices_current]

    i_first = indices_to_try[0]
    indices_max = indices_current.copy() + [i_first]

    M_i_first = M[:, indices_max]
    ans_max, P = solve_adversarial_perfect_detectors(M_i_first)

    for i in indices_to_try:
        indices = indices_current.copy() + [i]

        M_i = M[:, indices]
        ans, P = solve_adversarial_perfect_detectors(M_i)

        if ans > ans_max:
            ans_max = ans
            indices_max = indices

    end = time.time()

    

    if not os.path.exists(output_path): # if the directory does not exist
        os.makedirs(output_path) # make the directory

    k, n = M_i_first.shape

    if method == 'trainval':
        if reduced == 'no':
            with open(f"{output_path}/trainval_ans_{n}.txt", "w") as f:
                f.write(f"solver problem value: {ans_max} \n")
                f.write(f"pruned_matrix size: {k}, {n} \n")
                f.write(f'indices {indices_max} \n')
                f.write(f"time taken: {end - start} \n")
        elif reduced == 'yes':
            with open(f"{output_path}/reduced_trainval_ans_{n}_sub_{num_subset}.txt", "w") as f:
                f.write(f"solver problem value: {ans_max} \n")
                f.write(f"pruned_matrix size: {k}, {n} \n")
                f.write(f'indices {indices_max} \n')
                f.write(f"time taken: {end - start} \n")
    elif method == 'val':
        with open(f"{output_path}/val_ans_{n}.txt", "w") as f:
            f.write(f"solver problem value: {ans_max} \n")
            f.write(f"pruned_matrix size: {k}, {n} \n")
            f.write(f'indices {indices_max} \n')
            f.write(f"time taken: {end - start} \n")
    elif method == 'test':
        with open(f"{output_path}/test_ans_{n}.txt", "w") as f:
            f.write(f"solver problem value: {ans_max} \n")
            f.write(f"pruned_matrix size: {k}, {n} \n")
            f.write(f'indices {indices_max} \n')
            f.write(f"time taken: {end - start} \n")


    print(f'pruned matrix shape: {M_i_first.shape}')
    print(ans_max)

    return ans_max, indices_max

def get_indices(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        indices = lines[2].strip(']indices[\n ')
        array = [int(i) for i in indices.split(',')]
        return array