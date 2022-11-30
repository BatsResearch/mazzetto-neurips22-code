import os
import sys
from pathlib import Path
import pickle
import argparse
import scipy.io
import numpy as np
from scipy import io
from collections import Counter

from sklearn.metrics import confusion_matrix, accuracy_score, \
                            balanced_accuracy_score
# directory reach
directory = Path(os.path.abspath(__file__))
# setting path
sys.path.append(os.path.abspath(directory.parent.parent.parent))
from src.helper import read_dataset, \
                       load_class_attribute_matrix, \
                       load_classes
from src.datasets import aPY, AwA, LAD, SUN, CUB


parser = argparse.ArgumentParser()
parser.add_argument('-data', required=True, help='choose between APY, AWA2, AWA1, CUB, SUN, LAD', type = str)
parser.add_argument('-method', required=True, help='trainval, val, test', type = str)
parser.add_argument('-reduced', help='yes to reduce the number of classes, no otherwise', type = str, default='no')
parser.add_argument('-num_subset', help='number of the group of subclasses', type = str, default='all')
parser.add_argument('-num_att', help='number of attributes', type = int, default=15)
args = parser.parse_args()

def predict(X, M,
            p_attr, p_class=None):
    """
    :X: (samples, A)
    :M: (C, A)
    :p_attr: (A)
    :p_class: (C)
    """

    prob = []
    for p in X:
        num = np.prod(M*p + (1-M)*(1-p), axis=1)
        den = np.prod(M*p_attr + (1-M)*(1-p_attr), axis=1)
        prob.append(num / den)
    
    return np.argmax(prob, axis=1)


data = 'data/'
dataset = args.data + '/'
iterations = 1

ds = read_dataset(dataset, data)

images, image_idx, idx_image, imageid_class = ds.get_images()
classes, class_idx, idx_class, classid_imageid = ds.get_classes()
attributes, attr_idx, idx_attr, image_attr = ds.get_attributes()

if dataset == 'SUN/' or dataset == 'LAD/':
    image_attr = np.rint(image_attr)

if dataset == 'LAD/': 
    seen_classes, unseen_classes = ds.get_seen_unseen_classes()
else:
    seen_classes, unseen_classes = ds.get_seen_unseen_classes('testclasses.txt')
seen_samples, seen_old_to_new, seen_new_to_old = ds.get_seen_sample()
unseen_samples, unseen_old_to_new, unseen_new_to_old = ds.get_unseen_sample()

if dataset == 'AWA2/':
    class_attributes = ds.get_class_attribute_matrix('att_splits.mat',
                                                     continuous=True, 
                                                     bound=None, 
                                                     lb=None, ub=None)
    class_attributes = class_attributes/100
else:
    class_attributes = ds.get_class_attribute_matrix(continuous=True, 
                                                     bound=None, 
                                                     lb=None, ub=None)
    
embeddings = ds.get_embeddings()


data_folder = 'data/' + dataset

split_file = 'att_splits.mat'
splits = scipy.io.loadmat(f'{data}{dataset}splits/{split_file}')

train = splits['trainval_loc']
#val = splits['val_loc']
test = splits['test_unseen_loc']

np.random.seed(0)
index_train = sorted(list(np.random.choice(list(range(train.shape[0])), 
                                           size=int(train.shape[0]/100*80), 
                                           replace=False)))
index_val = sorted(list(set(list(range(train.shape[0]))).difference(set(index_train))))

train_val = train[index_val]
train = train[index_train]


res = 'res101.mat'
resnet101 = scipy.io.loadmat(f'{data}{dataset}splits/{res}')

train_classes = np.unique(np.squeeze([i[0][0] \
                                      for i in resnet101['labels'][train-1]]))

test_classes = np.unique(np.squeeze([i[0][0] \
                                     for i in resnet101['labels'][test-1]]))

models = pickle.load(open(f"results/{dataset}detectors/models_held_out.pickle", "rb" ) )

method = args.method
n_attr = args.num_att
reduced = args.reduced
num_subset = args.num_subset

res101 = io.loadmat(data_folder + 'res101.mat')
att_splits = io.loadmat(data_folder + 'att_splits.mat')

train_loc = 'train_loc'
val_loc = 'val_loc'
test_loc = 'test_unseen_loc'


attributes = []

if method == 'trainval':
    if reduced == 'no':
        with open(f'{data_folder}trainval_greedy.txt', 'r') as f:
            for a in f:
                attributes += [int(a.strip())]
    elif reduced == 'yes':
        with open(f'{data_folder}reduced_trainval_greedy_sub_{num_subset}.txt', 'r') as f:
            for a in f:
                attributes += [int(a.strip())]
        what_classes = []
        with open(f'{data_folder}subset_{num_subset}.txt', 'r') as f:
            for a in f:
                what_classes += [int(a.strip())+1]

elif method == 'val':
    with open(f'{data_folder}val_greedy.txt', 'r') as f:
        for a in f:
            attributes += [int(a.strip())]

elif method == 'test':
    with open(f'{data_folder}test_greedy.txt', 'r') as f:
        for a in f:
            attributes += [int(a.strip())]


feat = res101['features']
# Shape -> (dxN)
X_train = feat[:, [i[0]-1 for i in train_val]]
X_val = feat[:, np.squeeze(att_splits[val_loc]-1)]
X_trainval = X_train
X_test = feat[:, np.squeeze(att_splits[test_loc]-1)]


print('Tr:{}; Val:{}; Ts:{}\n'.format(X_train.shape[1], X_val.shape[1], X_test.shape[1]))

labels = res101['labels']
print(labels.shape)
labels_train = labels[[i[0]-1 for i in train_val]]
labels_val = labels[np.squeeze(att_splits[val_loc]-1)]
labels_trainval = labels_train
labels_test = labels[np.squeeze(att_splits[test_loc]-1)]


if reduced == 'yes':
    keep_train  = np.where(np.isin(labels_train, what_classes))[0]
    keep_trainval  = np.where(np.isin(labels_trainval, what_classes))[0]
    print(f"SHAPE di X: {X_train.shape}, {len(labels_train)}")
    X_train = X_train[:,keep_train]
    labels_train = labels_train[keep_train]

    X_trainval = X_trainval[:,keep_trainval]
    labels_trainval = labels_trainval[keep_trainval]



train_labels_seen = np.unique(labels_train)
val_labels_seen = np.unique(labels_val)
trainval_labels_seen = np.unique(labels_trainval)
test_labels_unseen = np.unique(labels_test)



#att = 1
for att in range(1, n_attr+1):
    # Select attributes
    
    
    attributes_ = attributes[:att]
    print(attributes_)

        
    M = class_attributes[:,attributes_]
    if method == 'trainval':
        X = X_trainval
        y_true = labels_trainval
        id_unseen_classes = trainval_labels_seen-1
        
        if reduced == 'no':
            trainval_split = [i[0]-1 for i in train_val]#att_splits[train_loc]-1#np.concatenate((att_splits[train_loc]-1,att_splits[val_loc]-1))
        elif reduced == 'yes':
            trainval_split = np.array([i[0]-1 for i in train_val])[keep_train]
        
        unseen_samples = [us for us in trainval_split]#[us[0] for us in trainval_split]
        print(f'NUM SAMPLES {len(y_true)}')

    elif method == 'val':
        X = X_val
        y_true = labels_val
        id_unseen_classes = val_labels_seen-1
        val_split = att_splits[val_loc]-1
        unseen_samples = [us[0] for us in val_split]
        id_seen_classes = id_unseen_classes#val_labels_seen -1
        
    elif method == 'test':
        X = X_test
        y_true = labels_test
        id_unseen_classes = test_labels_unseen-1
        test_split = att_splits[test_loc]-1
        unseen_samples = [us[0] for us in test_split]
        id_seen_classes = trainval_labels_seen-1
        
    C, A = M.shape

    if method == 'trainval':
        id_seen_classes = id_unseen_classes

    new_to_old_unseen = {new:old for new, old in enumerate(id_unseen_classes)}
    old_to_new_unseen = {old:new for new, old in new_to_old_unseen.items()}

    class_size = [len(classid_imageid[c]) for c in id_unseen_classes]
    sum_classes = np.sum(class_size)
    p_class = [size/sum_classes for size in class_size]
    
    #print(M[id_seen_classes].shape)
    p_attr = np.mean(M[id_seen_classes], axis=0)
    #print(p_attr)
    p_attr[p_attr==0.] = 0.5
    p_attr[p_attr==1.] = 0.5    # disallow degenerated priors

    M = M[id_unseen_classes, :]
    C, A = M.shape


    X = np.zeros((len(y_true), A))#((len(y_true), A))
    print(f"X SHAPE: {len(X)}")

    idx_att_to_x = {at:i for i, at in enumerate(attributes_)}
    idx_x_to_att = {i:at for at, i in idx_att_to_x.items()}

    mod_att = {a:models[a] for a in attributes_}
    for a in mod_att:
        att_model = mod_att[a]
        X_emb = embeddings.numpy()[unseen_samples,:]#.numpy()
        p_att_x = att_model.predict_proba(X_emb)[:,1]
        X[:,idx_att_to_x[a]] = p_att_x
        #print(p_att_x)

    y_pred = predict(X, M,
                p_attr, p_class=[1/len(unseen_classes)]*len(unseen_classes))

    real_y_pred = [new_to_old_unseen[new] for new in y_pred]
    y_true = [class_idx[imageid_class[i]] for i in unseen_samples]

    print(balanced_accuracy_score(y_true, real_y_pred))
    test_acc = balanced_accuracy_score(y_true, real_y_pred)
    
    with open(f'results/{dataset[:-1].upper()}/DAP/confusion_matrix/accuracy_{dataset[:-1].upper()}_method_{method}.txt', 'a') as f:
        f.write(f'{dataset[:-1].upper()}\t{method}\t{att}\t{test_acc}\t{reduced}\t{num_subset}\n')
    