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
# # directory reach
# directory = Path(os.path.abspath(__file__))
# # setting path
# sys.path.append(os.path.abspath(directory.parent.parent.parent))
from src.helper import read_dataset, train_data_loaders
from src.datasets import aPY, AwA, LAD, SUN, CUB


parser = argparse.ArgumentParser()
parser.add_argument('-data', required=True, help='choose between APY, AWA2, AWA1, CUB, SUN, LAD', type = str)
args = parser.parse_args()

data = 'data/'
dataset = args.data + '/'
iterations = 1

ds = read_dataset(dataset, data)

images, image_idx, idx_image, imageid_class = ds.get_images()
classes, class_idx, idx_class, classid_imageid = ds.get_classes()
attributes, attr_idx, idx_attr, image_attr = ds.get_attributes()

if dataset == 'SUN/':
    image_attr = np.rint(image_attr)

seen_samples, seen_old_to_new, seen_new_to_old = ds.get_seen_sample()
unseen_samples, unseen_old_to_new, unseen_new_to_old = ds.get_unseen_sample()
    
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


scores, num_examples, models = train_data_loaders(class_idx, train_classes, test_classes,
                                                  train, test, image_attr, embeddings,
                                                  dataset, seed=50, n_iter=1, max_iter=700)



