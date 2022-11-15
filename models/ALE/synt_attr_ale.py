"""
This code is a slightly adapted version of the ALE implementation at https://github.com/mvp18/Popular-ZSL-Algorithms.

The difference is in the class initialization, and in the evaluate function modified the metrics of interest.
"""


import pickle
import numpy as np
import argparse
from scipy import io, spatial
import time
from random import shuffle
import random
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description="ALE")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-e', '--epochs', default=100, type=int)
parser.add_argument('-es', '--early_stop', default=10, type=int)
parser.add_argument('-norm', '--norm_type', help='std(standard), L2, None', default='std', type=str)
parser.add_argument('-lr', '--lr', default=0.01, type=float)
parser.add_argument('-seed', '--rand_seed', default=42, type=int)
parser.add_argument('-mode', default='hyper', help='type hyper if searching final if using fixed params', type=str)
parser.add_argument('-att', '--att', default=1, type=int)
parser.add_argument('-method', '--method',  type=str,  help='trainval, val, test')


"""
Best Values of (norm, lr) found by validation & corr. test accuracies:
SUN  -> (L2, 0.1)   -> Test Acc : 0.6188
AWA1 -> (L2, 0.01)  -> Test Acc : 0.5656
AWA2 -> (L2, 0.01)  -> Test Acc : 0.5290
CUB  -> (L2, 0.3)   -> Test Acc : 0.4898
APY  -> (L2, 0.04)  -> Test Acc : 0.3276

python ale.py -data AWA2 -norm L2 -lr 0.1
"""

class ALE():
	
    def __init__(self, args):

        self.args = args

        random.seed(self.args.rand_seed)
        np.random.seed(self.args.rand_seed)

        data_folder = '../xlsa17/data/'+args.dataset+'/'
        self.data_folder = data_folder

        attributes = []
        if args.method == 'trainval':
            
            with open(f'{self.data_folder}/trainval_greedy.txt', 'r') as f:
                for a in f:
                    attributes += [int(a.strip())]

        elif args.method == 'val':
            with open(f'{self.data_folder}/val_greedy.txt', 'r') as f:
                for a in f:
                    attributes += [int(a.strip())]
        elif args.method == 'test':
            with open(f'{self.data_folder}/test_greedy.txt', 'r') as f:
                for a in f:
                    attributes += [int(a.strip())]

        self.attributes = list(range(args.att))


        res = f"../../adversarial_grid/{args.dataset}_synt_res101.p"
        res101 = pickle.load(open(res, "rb"))#io.loadmat(self.data_folder+'res101.mat')

        split_file= f"../../adversarial_grid/{args.dataset}_synt_att_splits.p"
        att_splits = pickle.load(open(split_file, "rb"))

        train_loc = 'train_loc'
        val_loc = 'val_loc'
        test_loc = 'test_unseen_loc'

        feat = res101['features']
        # Shape -> (dxN)
        self.X_train = feat[:, np.squeeze(att_splits[train_loc]-1)]
        self.X_val = feat[:, np.squeeze(att_splits[val_loc]-1)]
        self.X_test = feat[:, np.squeeze(att_splits[test_loc]-1)]

        print('Tr:{}; Val:{}; Ts:{}\n'.format(self.X_train.shape[1], self.X_val.shape[1], self.X_test.shape[1]))

        labels = res101['labels']
        self.labels_train = np.squeeze(labels[np.squeeze(att_splits[train_loc]-1)])
        self.labels_val = np.squeeze(labels[np.squeeze(att_splits[val_loc]-1)])
        self.labels_test = np.squeeze(labels[np.squeeze(att_splits[test_loc]-1)])

        train_labels_seen = np.unique(self.labels_train)
        val_labels_unseen = np.unique(self.labels_val)
        test_labels_unseen = np.unique(self.labels_test)

        i=0
        for labels in train_labels_seen:
            self.labels_train[self.labels_train == labels] = i    
            i+=1
        
        j=0
        for labels in val_labels_unseen:
            self.labels_val[self.labels_val == labels] = j
            j+=1
        
        k=0
        for labels in test_labels_unseen:
            self.labels_test[self.labels_test == labels] = k
            k+=1

        sig = att_splits['att'][self.attributes, :]
        print ("SIGNATURE SHAPE: ", sig.shape)
        # Shape -> (Number of attributes, Number of Classes)
        self.train_sig = sig[:, train_labels_seen-1]
        self.val_sig = sig[:, val_labels_unseen-1]
        self.test_sig = sig[:, test_labels_unseen-1]

        if self.args.norm_type=='std':
            print('Standard Scaling\n')
            scaler = preprocessing.StandardScaler()
            scaler.fit(self.X_train.T)

            self.X_train = scaler.transform(self.X_train.T).T
            self.X_val = scaler.transform(self.X_val.T).T
            self.X_test = scaler.transform(self.X_test.T).T

        if self.args.norm_type=='L2':
            print('L2 norm Scaling\n')
            self.X_train = self.normalizeFeature(self.X_train.T).T
            # self.X_val = self.normalizeFeature(self.X_val.T).T
            # self.X_test = self.normalizeFeature(self.X_test.T).T

    def normalizeFeature(self, x):
        # x = N x d (d:feature dimension, N:number of instances)
        x = x + 1e-10
        feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
        feat = x / feature_norm[:, np.newaxis]

        return feat

    def update_W(self, W, idx, train_classes, beta):
        
        for j in idx:
            
            X_n = self.X_train[:, j]
            y_n = self.labels_train[j]
            y_ = train_classes[train_classes!=y_n]
            XW = np.dot(X_n, W)
            gt_class_score = np.dot(XW, self.train_sig[:, y_n])

            for i in range(len(y_)):
                label = random.choice(y_)
                score = 1+np.dot(XW, self.train_sig[:, label])-gt_class_score # acc. to original paper, margin shd always be 1.
                if score>0:
                    Y = np.expand_dims(self.train_sig[:, y_n]-self.train_sig[:, label], axis=0)
                    W += self.args.lr*beta[int(y_.shape[0]/(i+1))]*np.dot(np.expand_dims(X_n, axis=1), Y)
                    break
        return W

    def fit(self):

        print('Training...\n')

        best_val_acc = 0.0
        best_tr_acc = 0.0
        best_val_ep = -1
        best_tr_ep = -1
        
        rand_idx = np.arange(self.X_train.shape[1])

        W = np.random.rand(self.X_train.shape[0], self.train_sig.shape[0])
        W = self.normalizeFeature(W.T).T

        train_classes = np.unique(self.labels_train)

        beta = np.zeros(len(train_classes))
        for i in range(1, beta.shape[0]):
            sum_alpha=0.0
            for j in range(1, i+1):
                sum_alpha+=1/j
            beta[i] = sum_alpha

        for ep in range(self.args.epochs):

            start = time.time()

            shuffle(rand_idx)

            W = self.update_W(W, rand_idx, train_classes, beta)

            tr_acc, _ = self.zsl_acc(self.X_train, W, self.labels_train, self.train_sig)			
            val_acc, _ = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig)

            end = time.time()
            
            elapsed = end-start
            
            print('Epoch:{}; Train Acc:{}; Val Acc:{}; Time taken:{:.0f}m {:.0f}s\n'.format(ep+1, tr_acc, val_acc, elapsed//60, elapsed%60))
            
            if val_acc>best_val_acc:
                best_val_acc = val_acc
                best_val_ep = ep+1
                best_W = np.copy(W)
            
            if tr_acc>best_tr_acc:
                best_tr_ep = ep+1
                best_tr_acc = tr_acc

            if ep+1-best_val_ep>self.args.early_stop:
                print('Early Stopping by {} epochs. Exiting...'.format(self.args.epochs-(ep+1)))
                break

        print('\nBest Val Acc:{} @ Epoch {}. Best Train Acc:{} @ Epoch {}\n'.format(best_val_acc, best_val_ep, best_tr_acc, best_tr_ep))
        
        return best_W

    def zsl_acc(self, X, W, y_true, sig): # Class Averaged Top-1 Accuarcy

        XW = np.dot(X.T, W)# N x k
        dist = 1-spatial.distance.cdist(XW, sig.T, 'cosine')# N x C(no. of classes)
        predicted_classes = np.array([np.argmax(output) for output in dist])
        cm = confusion_matrix(y_true, predicted_classes)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_save = cm.astype('float') #/ cm.sum(axis=1)[:, np.newaxis]
        acc = sum(cm.diagonal())/sig.shape[1]

        to_save = (y_true, predicted_classes)
        
        return acc, to_save

    def evaluate(self):

        best_W = self.fit()

        print('Testing...\n')

        
        
        if args.method == 'trainval':
            test_acc, cm_save = self.zsl_acc(self.X_train, best_W, self.labels_train, self.train_sig)

        elif args.method == 'val':
            test_acc, cm_save = self.zsl_acc(self.X_val, best_W, self.labels_val, self.val_sig)
            
        elif args.method == 'test':
            test_acc, cm_save = self.zsl_acc(self.X_test, best_W, self.labels_test, self.test_sig)


        with open(f'confusion_matrix/synt_{args.dataset}_method_{args.method}_n_att_{args.att}_cm_seed_{self.args.rand_seed}.pickle', 'wb') as handle:
            pickle.dump(cm_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'confusion_matrix/accuracy_synt_{args.dataset}_method_{args.method}.txt', 'a') as f:
            f.write(f'{args.dataset}\t{args.method}\t{args.att}\t{test_acc}\n')
		
        print('Test Acc:{}'.format(test_acc))

if __name__ == '__main__':
	
    args = parser.parse_args()
    print('Dataset : {}\n'.format(args.dataset))

    clf = ALE(args)	
    clf.evaluate()
