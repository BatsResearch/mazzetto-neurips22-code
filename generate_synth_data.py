
import sys
import time
import pickle
import argparse

import numpy as np
import pandas as pd
from docplex.mp.model import Model

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

    return m.solution.get_objective_value(), P

def load_classes(dataset_folder, method, reduced, num_sub):
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


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', required=True, help='dataset name', type=str)
parser.add_argument('-method', required=True, help='trainval, val, or test', type=str)
parser.add_argument('-reduced', required=True, help='yes, no', type=str)
parser.add_argument('-num_subset', required=True, help='1 to 5 or all', type=str)

args = parser.parse_args()
dataset = args.dataset #CUB
dataset_folder = f'data/class_attribute_matrices/{dataset}'
method = args.method
reduced = args.reduced
num_subset = args.num_subset

all_classes, unseen_classes = load_classes(dataset_folder,
                                           method, reduced, 
                                           num_subset)
M = load_class_attribute_matrix(all_classes, unseen_classes, 
                                dataset_folder, method=method)

filename_attributes = f"data/{dataset}/test_greedy.txt"

attributes = []
with open(filename_attributes) as f:
    lines = f.readlines()
for l in lines:
    attributes.append(int(l.split()[0]))

M = M[:,attributes]

K = M.shape[0]
N = M.shape[1]

_, P = solve_adversarial_perfect_detectors(M)
P_ = np.zeros((K, 2**N))
for i,j in P.keys():
    P_[i,j] = P[i,j].solution_value

# Remove precision problem from the solver
P_ = np.abs(P_)
# Make probability for each class sum to 1 (this works with balanced classes only)
P_ = K*P_/np.sum(P_)

# Create an auxiliary data structure to do the sampling
Q = np.zeros((K,2**N))
for k in range(K):
    Q[k,0] = P_[k,0]
    for i in range(1,2**N):
        Q[k,i] = P_[k,i] + Q[k,i-1]

n_test = 100000
X = []
Y = []
for l in range(n_test):
    c = np.random.randint(K)
    p = np.random.random()
    pos = np.searchsorted(Q[c,:],p)
    x = []
    for i in range(N):
        if (pos & (2**i) > 0):
            x.append(1)
        else:
            x.append(0)
    X.append(x)
    Y.append(c)

dict_res101 = {'features': np.array(X).T,
              'labels': np.array(Y).reshape(len(Y),1)}

with open(f'data/{dataset}_synt_res101.p', 'wb') as handle:
    pickle.dump(dict_res101, handle)

X = np.array(X).T
seen = int(X.shape[1]*0.7)
unseen = X.shape[1] - seen
synt_att_splits = {'test_seen_loc': np.arange(seen + 1, seen + 1 + unseen).reshape(unseen, 1),
                   'test_unseen_loc': np.arange(seen + 1, seen + 1 + unseen).reshape(unseen,1),
                   'train_loc': np.arange(1, seen + 1).reshape(seen, 1),
                   'trainval_loc':np.arange(1, seen + 1).reshape(seen, 1),
                   'val_loc': np.arange(1, seen + 1).reshape(seen, 1),
                   'att': M.T}

with open(f'data/{dataset}_synt_att_splits.p', 'wb') as handle:
    pickle.dump(synt_att_splits, handle)