"""
This code is a slightly adapted version of the ALE implementation at https://github.com/mvp18/Popular-ZSL-Algorithms.

The difference is in the class initialization, and in the evaluate function modified the metrics of interest.
"""


import pickle
import numpy as np
import argparse
from scipy import io, spatial, linalg
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description="SAE")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-mode', '--mode', help='train/test, if test set alpha, gamma to best values below', default='train', type=str)
parser.add_argument('-ld1', '--ld1', default=5, help='best value for F-->S during test, lower bound of variation interval during train', type=float)
parser.add_argument('-ld2', '--ld2', default=5, help='best value for S-->F during test, upper bound of variation interval during train', type=float)
parser.add_argument('-att', '--att', default=1, type=int)
parser.add_argument('-method', '--method',  type=str)
parser.add_argument('-reduced', help='yes to reduce the number of classes, no otherwise', type = str, default='no')
parser.add_argument('-num_subset', help='number of the group of subclasses', type = str, default='nan')



""" 
Range of Lambda for Validation:
AWA1 -> 2-8 for [F-->S] and 0.4-1.6 for [S-->F]
AWA2 -> 0.1-1.6
CUB  -> 50-5000 for [F-->S] and 0.05-5 for [S-->F]
SUN  -> 0.005-5
APY  -> 0.5-50
Best Value of Lambda found by validation & corr. test accuracies:
		   				
AWA1 -> 0.5134 @ 3.0  [F-->S] 0.5989 @ 0.8  [S-->F]
AWA2 -> 0.5166 @ 0.6  [F-->S] 0.6051 @ 0.2  [S-->F]
CUB  -> 0.3948 @ 100  [F-->S] 0.4670 @ 0.2  [S-->F]
SUN  -> 0.5285 @ 0.32 [F-->S] 0.5986 @ 0.16 [S-->F]
APY  -> 0.1607 @ 2.0  [F-->S] 0.1650 @ 4.0  [S-->F] 

python sae.py -data AWA2 -mode val -ld1 0.1 -ld2 1.6 -att 1 -method greedy
"""

class SAE():
    
    def __init__(self, args):

        self.args = args
        data_folder = '../xlsa17/data/' + args.dataset+'/'
        self.data_folder = data_folder
        
        attributes = []

        if args.method == 'trainval':
            if args.reduced == 'no':
                with open(f'{self.data_folder}trainval_greedy.txt', 'r') as f:
                    for a in f:
                        attributes += [int(a.strip())]
            elif args.reduced == 'yes':
                with open(f'{self.data_folder}reduced_trainval_greedy_sub_{args.num_subset}.txt', 'r') as f:
                    for a in f:
                        attributes += [int(a.strip())]
                what_classes = []
                with open(f'{self.data_folder}subset_{args.num_subset}.txt', 'r') as f:
                    for a in f:
                        what_classes += [int(a.strip())+1]

                print(f"NUM REDUCED CLASSES: {len(what_classes)}")

        elif args.method == 'val':
            with open(f'{self.data_folder}/val_greedy.txt', 'r') as f:
                for a in f:
                    attributes += [int(a.strip())]
        elif args.method == 'test':
            with open(f'{self.data_folder}/test_greedy.txt', 'r') as f:
                for a in f:
                    attributes += [int(a.strip())]

        self.attributes = attributes[:args.att]
               
        res101 = io.loadmat(data_folder + 'res101.mat')
        att_splits=io.loadmat(data_folder + 'att_splits.mat')

        train_loc = 'train_loc'
        val_loc = 'val_loc'
        test_loc = 'test_unseen_loc'

        feat = res101['features']
        self.X_train = feat[:, np.squeeze(att_splits[train_loc]-1)]
        self.X_val = feat[:, np.squeeze(att_splits[val_loc]-1)]
        self.X_test = feat[:, np.squeeze(att_splits[test_loc]-1)]

        print('Tr:{}; Val:{}; Ts:{}\n'.format(self.X_train.shape[1], self.X_val.shape[1], self.X_test.shape[1]))

        labels = res101['labels']
        self.labels_train = labels[np.squeeze(att_splits[train_loc]-1)]
        self.labels_val = labels[np.squeeze(att_splits[val_loc]-1)]
        self.labels_test = labels[np.squeeze(att_splits[test_loc]-1)]

        if args.reduced == 'yes':
            keep_train  = np.where(np.isin(self.labels_train, what_classes))[0]
            #keep_trainval  = np.where(np.isin(labels_trainval, what_classes))[0]
            print(f"NUM X KEPT: {keep_train}")
            self.X_train = self.X_train[:,keep_train]
            self.labels_train = self.labels_train[keep_train]
            print(f"SHAPE di X: {self.X_train.shape}")

            #self.X_trainval = self.X_trainval[:,keep_trainval]
            #labels_trainval = labels_trainval[keep_trainval]


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
        print ("SIGNATURE SHAPE: ", sig.shape, )
        
        #sig = att_splits['att']# k x C
        self.train_sig = sig[:, train_labels_seen-1]
        self.val_sig = sig[:, val_labels_unseen-1]
        self.test_sig = sig[:, test_labels_unseen-1]
        print ("SIGNATURE TRAIN: ", self.train_sig.shape, len(train_labels_seen))

        self.train_att = np.zeros((self.X_train.shape[1], self.train_sig.shape[0]))
        for i in range(self.train_att.shape[0]):
            if args.reduced == 'no':
                #print(self.labels_train[i])
                self.train_att[i] = self.train_sig.T[self.labels_train[i][0]]
            elif args.reduced == 'yes':
                #print(self.labels_train[i])
                self.train_att[i] = self.train_sig.T[self.labels_train[i]]
        print ("SIGNATURE TRAIN_ATT: ", self.train_att.shape)
        self.X_train = self.normalizeFeature(self.X_train.T).T

    def normalizeFeature(self, x):
        # x = N x d (d:feature dimension, N:number of instances)
        x = x + 1e-10
        feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
        feat = x / feature_norm[:, np.newaxis]

        return feat

    def find_W(self, X, S, ld):

        # INPUTS:
        # X: d x N - data matrix
        # S: Number of Attributes (k) x N - semantic matrix
        # ld: regularization parameter
        #
        # Return :
        # 	W: kxd projection matrix

        A = np.dot(S, S.T)
        B = ld*np.dot(X, X.T)
        C = (1+ld)*np.dot(S, X.T)
        W = linalg.solve_sylvester(A, B, C)
        
        return W

    def find_lambda(self):

        print('Training...\n')

        best_acc_F2S = 0.0
        best_acc_S2F = 0.0

        ld = self.args.ld1

        while (ld<=self.args.ld2):
            
            W = self.find_W(self.X_train, self.train_att.T, ld)
            acc_F2S, acc_S2F, _, _ = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig, 'val')
            print('Val Acc --> [F-->S]:{} [S-->F]:{} @ lambda = {}\n'.format(acc_F2S, acc_S2F, ld))
            
            if acc_F2S>best_acc_F2S:
                best_acc_F2S = acc_F2S
                lambda_F2S = ld
                best_W_F2S = np.copy(W)

            if acc_S2F>best_acc_S2F:
                best_acc_S2F = acc_S2F
                lambda_S2F = ld
                best_W_S2F = np.copy(W)
            
            ld*=2			

        print('\nBest Val Acc --> [F-->S]:{} @ lambda = {} [S-->F]:{} @ lambda = {}\n'.format(best_acc_F2S, lambda_F2S, best_acc_S2F, lambda_S2F))
        
        return best_W_F2S, best_W_S2F

    def zsl_acc(self, X, W, y_true, sig, mode='val'): # Class Averaged Top-1 Accuarcy

        if mode=='F2S':
            # [F --> S], projecting data from feature space to semantic space
            F2S = np.dot(X.T, self.normalizeFeature(W).T)# N x k
            dist_F2S = 1-spatial.distance.cdist(F2S, sig.T, 'cosine')# N x C(no. of classes)
            pred_F2S = np.array([np.argmax(y) for y in dist_F2S])
            cm_F2S = confusion_matrix(y_true, pred_F2S)
            cm_F2S = cm_F2S.astype('float')/cm_F2S.sum(axis=1)[:, np.newaxis]
            cm_save = cm_F2S.astype('float')
            acc_F2S = sum(cm_F2S.diagonal())/sig.shape[1]

            to_save = (y_true, pred_F2S)

            return acc_F2S, to_save

        if mode=='S2F':
            # [S --> F], projecting from semantic to visual space
            S2F = np.dot(sig.T, self.normalizeFeature(W))
            dist_S2F = 1-spatial.distance.cdist(X.T, self.normalizeFeature(S2F), 'cosine')
            pred_S2F = np.array([np.argmax(y) for y in dist_S2F])
            cm_S2F = confusion_matrix(y_true, pred_S2F)
            cm_S2F = cm_S2F.astype('float')/cm_S2F.sum(axis=1)[:, np.newaxis]
            cm_save = cm_S2F.astype('float')
            acc_S2F = sum(cm_S2F.diagonal())/sig.shape[1]

            to_save = (y_true, pred_S2F)

            return acc_S2F, to_save	

        if mode=='val':
            # [F --> S], projecting data from feature space to semantic space
            F2S = np.dot(X.T, self.normalizeFeature(W).T)# N x k
            dist_F2S = 1-spatial.distance.cdist(F2S, sig.T, 'cosine')# N x C(no. of classes)
            # [S --> F], projecting from semantic to visual space
            S2F = np.dot(sig.T, self.normalizeFeature(W))
            dist_S2F = 1-spatial.distance.cdist(X.T, self.normalizeFeature(S2F), 'cosine')
            
            pred_F2S = np.array([np.argmax(y) for y in dist_F2S])
            pred_S2F = np.array([np.argmax(y) for y in dist_S2F])
            
            cm_F2S = confusion_matrix(y_true, pred_F2S)
            cm_F2S = cm_F2S.astype('float')/cm_F2S.sum(axis=1)[:, np.newaxis]
            save_F2S = (y_true, pred_F2S)

            cm_S2F = confusion_matrix(y_true, pred_S2F)
            cm_S2F = cm_S2F.astype('float')/cm_S2F.sum(axis=1)[:, np.newaxis]
            save_S2F = (y_true, pred_S2F)
            
            acc_F2S = sum(cm_F2S.diagonal())/sig.shape[1]
            acc_S2F = sum(cm_S2F.diagonal())/sig.shape[1]

            # acc = acc_F2S if acc_F2S>acc_S2F else acc_S2F

            return acc_F2S, acc_S2F, save_F2S, save_S2F

    def evaluate(self):

        if self.args.mode=='train': best_W_F2S, best_W_S2F = self.find_lambda()
        else: 
            np.random.seed(0)
            index_train = sorted(list(np.random.choice(list(range(self.X_train.shape[1])), size=int(self.X_train.shape[1]/100*80), replace=False)))
            index_val = sorted(list(set(list(range(self.X_train.shape[1]))).difference(set(index_train))))
            best_W_F2S = self.find_W(self.X_train[:,index_train], self.train_att.T[:,index_train], self.args.ld1)
            best_W_S2F = self.find_W(self.X_train[:,index_train], self.train_att.T[:,index_train], self.args.ld2)

        

        if args.method == 'trainval':
            test_acc_F2S, cm_F2S = self.zsl_acc(self.X_train[:,index_val], best_W_F2S, self.labels_train[index_val], self.train_sig, 'F2S')
            test_acc_S2F, cm_S2F = self.zsl_acc(self.X_train[:, index_val], best_W_S2F, self.labels_train[index_val], self.train_sig, 'S2F')
        
        elif args.method == 'val':
            # VAL
            test_acc_F2S, cm_F2S = self.zsl_acc(self.X_val, best_W_F2S, self.labels_val, self.val_sig, 'F2S')
            test_acc_S2F, cm_S2F = self.zsl_acc(self.X_val, best_W_S2F, self.labels_val, self.val_sig, 'S2F')

        elif args.method == 'test':
            # TEST
            test_acc_F2S, cm_F2S = self.zsl_acc(self.X_test, best_W_F2S, self.labels_test, self.test_sig, 'F2S')
            test_acc_S2F, cm_S2F = self.zsl_acc(self.X_test, best_W_S2F, self.labels_test, self.test_sig, 'S2F')


        if args.reduced == 'no':
            with open(f'confusion_matrix/{args.dataset}_method_{args.method}_n_att_{args.att}_cm_F2S.pickle', 'wb') as handle:
                pickle.dump(cm_F2S, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif args.reduced == 'yes':
            with open(f'confusion_matrix/reduced_{args.dataset}_method_{args.method}_n_att_{args.att}_sub_{args.num_subset}_cm_F2S.pickle', 'wb') as handle:
                pickle.dump(cm_F2S, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if args.reduced == 'no':
            with open(f'confusion_matrix/{args.dataset}_method_{args.method}_n_att_{args.att}_cm_S2F.pickle', 'wb') as handle:
                pickle.dump(cm_S2F, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif args.reduced == 'yes':
            with open(f'confusion_matrix/reduced_{args.dataset}_method_{args.method}_n_att_{args.att}_sub_{args.num_subset}_cm_S2F.pickle', 'wb') as handle:
                pickle.dump(cm_S2F, handle, protocol=pickle.HIGHEST_PROTOCOL)


        print('Test Acc --> [F-->S]:{} [S-->F]:{}'.format(test_acc_F2S, test_acc_S2F))


        with open(f'confusion_matrix/accuracy_{args.dataset}_method_{args.method}.txt', 'a') as f:
            f.write(f'{args.dataset}\t{args.method}\t{args.att}\t{test_acc_F2S}\t{test_acc_S2F}\t{args.reduced}\t{args.num_subset}\n')
		
        

if __name__ == '__main__':

    args = parser.parse_args()
    print('Dataset : {}\n'.format(args.dataset))
    clf = SAE(args)
    clf.evaluate()
