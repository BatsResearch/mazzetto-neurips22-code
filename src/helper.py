import pickle
import itertools
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import class_weight
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.identifiability import identifiability
from src.datasets import aPY, AwA, LAD, SUN, CUB

def transform_image():
    """
    Get the transform to be used on an image.
    :return: A transform
    """
    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]
    # Remember to check it for video and eval
    return transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=data_mean, 
                                         std=data_std),
                ])


def train_data_loaders(class_idx, unseen_classes, seen_classes,
                       seen_samples, unseen_samples,
                      class_attributes, image_attr, embeddings,
                       dataset,
                      seed=1, n_iter=3, max_iter=500):
    """Returns scores, num_examples, and saves and
    returns list of models.
    """
    
    
    idx_unseen = sorted([class_idx[c] for c in unseen_classes])
    idx_seen = sorted([class_idx[c] for c in seen_classes])

    M = class_attributes[idx_seen]
    M_test = class_attributes[idx_unseen]
    
    # Split attribute matrices
    img_att_seen = image_attr[seen_samples]
    img_att_unseen = image_attr[unseen_samples]
    
    # Split feature matrices
    X_tmp = embeddings[seen_samples]
    X_test = embeddings[unseen_samples]

    # Initialize return
    scores = defaultdict(list)
    num_examples = {}
    models = {}
    C, A = np.shape(M)
    
    for i,a in enumerate(tqdm(range(A))):
        
        
        # Get examples with the attribute
        idx_attr = np.where(img_att_seen[:,a] > 0)[0]
        idx_attr_test = np.where(img_att_unseen[:,a] > 0)[0]
        
        
        # Assign o,1 labe to all the examples
        y_tmp = np.array([1 if i in idx_attr else 0 \
                      for i in range(X_tmp.shape[0])])
        y_test = np.array([1 if i in idx_attr_test else 0 \
                      for i in range(X_test.shape[0])])
        
        num_examples[a] = {"Num pos train":len(idx_attr),
                           "Num pos test": len(idx_attr_test),
                           "Num neg train": np.sum(y_tmp==0),
                          "Num neg test": np.sum(y_test==0)}
        
        for s in range(n_iter):
            # Train logistic models for different splits of train and validation
            X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, 
                                                              test_size=0.33, 
                                                              random_state=s)
            
            try:
                clf = LogisticRegression(random_state=seed, 
                                         max_iter=max_iter, 
                                         class_weight='balanced'
                                         ).fit(X_train, y_train)
                
                y_val_pred = clf.predict(X_val)
                
                valpoint = clf.score(X_val, y_val)
                balanced_val = balanced_accuracy_score(y_val, y_val_pred)
                #print('VAL: ', np.unique(y_val), np.unique(y_val_pred))
                
                y_test_pred = clf.predict(X_test)
                testpoint = clf.score(X_test, y_test)
                balanced_test = balanced_accuracy_score(y_test, y_test_pred)
                #print('TEST: ', np.unique(y_test), np.unique(y_test_pred))
            
            except ValueError:
                print('Not samples from the two classes -> Assign -100 to identify nan')
                scores[a] += [(-100,-100, -100, -100)]
                continue
                
            scores[a] += [(valpoint, balanced_val, 
                           testpoint, balanced_test)]
        
        #print(a, scores[a])
        
        models[a] = clf
          
    with open(f'results/{dataset}detectors/score_detectors.pickle', 'wb') as handle:
        pickle.dump(scores, 
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(f'results/{dataset}detectors/numers_examples_detectors.pickle', 'wb') as handle:
        pickle.dump(num_examples, 
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(f'results/{dataset}detectors/models.pickle', 'wb') as handle:
        pickle.dump(models, 
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            
    return scores, num_examples, models



def compute_identifiability(ds, lbs, ubs, dataset):
     
    #pairs_thresholds = list(itertools.product(lbs, ubs))
    identifiability_results = {}
    for lt, ut in zip(lbs, ubs):
        print(lt, ut)
        class_attributes = ds.get_class_attribute_matrix(continuous=False, 
                                                     bound=None, 
                                                     lb=lt, ub=ut)

        gamma = identifiability(class_attributes)
        edges = []
        for cls, neighs in enumerate(gamma):
            neighs.remove(cls)
            edges += [set(neighs)]
        adj_matrix = np.zeros((class_attributes.shape[0],
                               class_attributes.shape[0]))
        for i, neighs in enumerate(edges):
            adj_matrix[i, list(neighs)] = 1

        identifiability_results[(lt, ut)] = {'signature': class_attributes,
                                             'graph': edges,
                                             'adj_matrix': adj_matrix,
                                             'n_present_att': len(np.where(class_attributes==1)[0]),
                                             'n_uncertain_att': len(np.where(class_attributes==0)[0]),
                                             'n_not_present_att': len(np.where(class_attributes==-1)[0])}
        
    with open(f'results/{dataset}identifiability/data/identifiability_results.pickle', 'wb') as handle:
        pickle.dump(identifiability_results, 
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return identifiability_results

def compute_identifiability_continuous(class_attributes, ts, dataset):
     
    #pairs_thresholds = list(itertools.product(lbs, ubs))
    identifiability_results = {}
    for t in ts:
        #print(t)

        gamma = identifiability_continuous(class_attributes, t)
        print(gamma)
        edges = []
        for cls, neighs in enumerate(gamma):
            if cls in neighs:
                neighs.remove(cls)
            edges += [set(neighs)]
        adj_matrix = np.zeros((class_attributes.shape[0],
                               class_attributes.shape[0]))
        for i, neighs in enumerate(edges):
            adj_matrix[i, list(neighs)] = 1 + class_attributes[i, list(neighs)]
            #print(1 + class_attributes[i, list(neighs)])

        identifiability_results[round(t,1)] = {'signature': class_attributes,
                                             'graph': edges,
                                             'adj_matrix': adj_matrix,
                                             'n_present_att': len(np.where(class_attributes==1)[0]),
                                             'n_uncertain_att': len(np.where(class_attributes==0)[0]),
                                             'n_not_present_att': len(np.where(class_attributes==-1)[0])}
        
        #print('\n\n')
        
    with open(f'results/{dataset}identifiability/data/continuous_identifiability_results.pickle', 'wb') as handle:
        pickle.dump(identifiability_results, 
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return identifiability_results

def plot_adj_matrix(mat, ticks, save=True, title='Add title', file_name='Add file path', size=30):
    plt.figure(figsize=(size,size))
    plt.imshow(mat,
               vmin=0,
               vmax=0.5,
               cmap="Greys",
               interpolation="none")
    plt.title(title)#f'{model}: {dataset} unseen confusion matrix'
    plt.xticks(range(len(ticks)), 
               ticks, rotation=90)
    plt.yticks(range(len(ticks)), 
               ticks, rotation=0)
    if save:
        plt.savefig(file_name)#results/{dataset}identifiability/imgs/models/{model}_abslute_unseen.png
    plt.show()
    
    
def plot_adj_matrix_model(mat, ticks, save=True, title='Add title', file_name='Add file path', size=30):
    plt.figure(figsize=(size,size))
    
    #cmap = plt.cm.gray
    #norm = plt.Normalize(0, 1)
    #rgba = norm(mat)

    plt.imshow(mat,#mat,
               vmin=0,
               vmax=1,
               cmap="Greys",
               interpolation="none")
    plt.title(title)#f'{model}: {dataset} unseen confusion matrix'
    plt.xticks(range(len(ticks)), 
               ticks, rotation=90)
    plt.yticks(range(len(ticks)), 
               ticks, rotation=0)
    if save:
        plt.savefig(file_name)#results/{dataset}identifiability/imgs/models/{model}_abslute_unseen.png
    plt.show()
    
    #return rgba
    
    
def read_dataset(dataset, data):
    if dataset == 'APY/':
        ds = aPY(dataset, data, 'image_data.csv',
                 'attribute_names.txt',
                 'att_splits.mat','testclasses.txt',
                 continuous=True)
    elif dataset == 'CUB/':
        ds = CUB(dataset, data, 'images.txt',
                 'attributes.txt',
                 'image_attribute_labels.txt', 
                 'att_splits.mat', 'testclasses.txt',
                 continuous=True)
    elif dataset == 'AWA2/':
        ds = AwA(dataset, data, 'classes.txt',
                 'predicates.txt',
                 'predicate-matrix-continuous.txt',
                 'att_splits.mat', 'testclasses.txt',
                 continuous=True)
    elif dataset == 'SUN/':
        ds = SUN(dataset, data, 'images.mat',
                 'attributes.mat', 
                 'attributeLabels_continuous.mat', 
                 'att_splits.mat', 'testclasses.txt',
                 continuous=True)
   
    return ds

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