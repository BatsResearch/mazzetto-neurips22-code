"""
This code is a slightly adapted version of the ALE implementation at https://github.com/mvp18/Popular-ZSL-Algorithms.

The difference is in the class initialization, and in the evaluate function modified the metrics of interest.
"""

import numpy as np
import argparse
import pickle
from scipy import io
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description="ESZSL")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-mode', '--mode', help='train/test, if test set alpha, gamma to best values below', default='train', type=str)
parser.add_argument('-alpha', '--alpha', default=0, type=int)
parser.add_argument('-gamma', '--gamma', default=0, type=int)
parser.add_argument('-att', '--att', default=1, type=int)
parser.add_argument('-method', '--method',  type=str)
parser.add_argument('-reduced', help='yes to reduce the number of classes, no otherwise', type = str, default='no')
parser.add_argument('-num_subset', help='number of the group of subclasses', type = str, default='all')
#parser.add_argument('-unseen',  help='seen (no) unseen (yes)', type=str)

"""
Alpha --> Regularizer for Kernel/Feature Space
Gamma --> Regularizer for Attribute Space
Best Values of (Alpha, Gamma) found by validation & corr. test accuracies:
AWA1 -> (3, 0)  -> Test Acc : 0.5680
AWA2 -> (3, 0)  -> Test Acc : 0.5482
CUB  -> (3, -1) -> Test Acc : 0.5394
SUN  -> (3, 2)  -> Test Acc : 0.5569
APY  -> (3, -1) -> Test Acc : 0.3856
LAD -> (-10, -3) -> Test Acc : 0.2525

python attr_zsl.py -data APY -mode test -alpha 3 -gamma -1  -att 1 -method greedy
"""

class ESZSL():
	
	def __init__(self):

		self.data_folder = 'data/' + args.dataset+'/'
		
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


		elif args.method == 'val':
			with open(f'{self.data_folder}val_greedy.txt', 'r') as f:
				for a in f:
					attributes += [int(a.strip())]
		elif args.method == 'test':
			with open(f'{self.data_folder}test_greedy.txt', 'r') as f:
				for a in f:
					attributes += [int(a.strip())]

		self.attributes = attributes[:args.att]

		res101 = io.loadmat(self.data_folder+'res101.mat')
		att_splits=io.loadmat(self.data_folder+'att_splits.mat')

		train_loc = 'train_loc'
		val_loc = 'val_loc'
		test_loc = 'test_unseen_loc'

		feat = res101['features']
		# Shape -> (dxN)
		self.X_train = feat[:, np.squeeze(att_splits[train_loc]-1)]
		self.X_val = feat[:, np.squeeze(att_splits[val_loc]-1)]
		self.X_trainval = np.concatenate((self.X_train, self.X_val), axis=1)
		self.X_test = feat[:, np.squeeze(att_splits[test_loc]-1)]

		print('Tr:{}; Val:{}; Ts:{}\n'.format(self.X_train.shape[1], self.X_val.shape[1], self.X_test.shape[1]))

		labels = res101['labels']
		print(labels.shape)
		labels_train = labels[np.squeeze(att_splits[train_loc]-1)]
		self.labels_val = labels[np.squeeze(att_splits[val_loc]-1)]
		labels_trainval = np.concatenate((labels_train, self.labels_val), axis=0)
		self.labels_test = labels[np.squeeze(att_splits[test_loc]-1)]


		if args.reduced == 'yes':
			keep_train  = np.where(np.isin(labels_train, what_classes))[0]
			keep_trainval  = np.where(np.isin(labels_trainval, what_classes))[0]
			print(f"SHAPE di X: {self.X_train.shape}")
			self.X_train = self.X_train[:,keep_train]
			labels_train = labels_train[keep_train]

			self.X_trainval = self.X_trainval[:,keep_trainval]
			labels_trainval = labels_trainval[keep_trainval]


		train_labels_seen = np.unique(labels_train)
		val_labels_unseen = np.unique(self.labels_val)
		trainval_labels_seen = np.unique(labels_trainval)
		test_labels_unseen = np.unique(self.labels_test)
		print(train_labels_seen)

		i=0
		for labels in train_labels_seen:
			labels_train[labels_train == labels] = i    
			i+=1
		
		j=0
		for labels in val_labels_unseen:
			self.labels_val[self.labels_val == labels] = j
			j+=1
		
		k=0
		for labels in trainval_labels_seen:
			labels_trainval[labels_trainval == labels] = k
			k+=1
		
		l=0
		for labels in test_labels_unseen:
			self.labels_test[self.labels_test == labels] = l
			l+=1

		self.labels_trainval = labels_trainval

		self.gt_train = np.zeros((labels_train.shape[0], len(train_labels_seen)))
		self.gt_train[np.arange(labels_train.shape[0]), np.squeeze(labels_train)] = 1

		self.gt_trainval = np.zeros((labels_trainval.shape[0], len(trainval_labels_seen)))
		self.gt_trainval[np.arange(labels_trainval.shape[0]), np.squeeze(labels_trainval)] = 1

		if args.reduced == 'no':
			sig = att_splits['att'][self.attributes, :]
		elif args.reduced == 'yes':
			sig = att_splits['att'][self.attributes, :]
		print ("SIGNATURE SHAPE: ", sig.shape)
		# Shape -> (Number of attributes, Number of Classes)
		print(len(train_labels_seen-1))
		self.train_sig = sig[:, train_labels_seen-1]
		self.val_sig = sig[:, val_labels_unseen-1]
		self.trainval_sig = sig[:, trainval_labels_seen-1]
		self.test_sig = sig[:, test_labels_unseen-1]

	def find_W(self, X, y, sig, alpha, gamma):

		part_0 = np.linalg.pinv(np.matmul(X, X.T) + (10**alpha)*np.eye(X.shape[0]))
		part_1 = np.matmul(np.matmul(X, y), sig.T)
		part_2 = np.linalg.pinv(np.matmul(sig, sig.T) + (10**gamma)*np.eye(sig.shape[0]))

		W = np.matmul(np.matmul(part_0, part_1), part_2) # Feature Dimension x Number of Attributes

		return W

	def fit(self):

		print('Training...\n')

		best_acc = 0.0

		for alph in range(-10, 10):
			for gamm in range(-10, 10):
				W = self.find_W(self.X_train, self.gt_train, self.train_sig, alph, gamm)
				acc, _, slsl = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig)
				print('Val Acc:{}; Alpha:{}; Gamma:{}\n'.format(acc, alph, gamm))
				if acc>best_acc:
					best_acc = acc
					alpha = alph
					gamma = gamm

		print('\nBest Val Acc:{} with Alpha:{} & Gamma:{}\n'.format(best_acc, alpha, gamma))
		
		return alpha, gamma

	def zsl_acc(self, X, W, y_true, sig): # Class Averaged Top-1 Accuarcy

		class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
		predicted_classes = np.array([np.argmax(output) for output in class_scores])
		#y_true = [int(i[0]) for i in y_true]
		print(f"true values: {np.unique(y_true)}")
		print(f"predicted: {np.unique(predicted_classes)}")
		cm = confusion_matrix(y_true, predicted_classes)
		cm_save = cm.astype('float')
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		acc = sum(cm.diagonal())/sig.shape[1]

		to_save = (y_true, predicted_classes)

		return acc, cm, to_save


	def evaluate(self, alpha, gamma):

		print('Testing...\n')

		if args.method == 'trainval':
			np.random.seed(0)
			index_train = sorted(list(np.random.choice(list(range(self.X_trainval.shape[1])), size=int(self.X_trainval.shape[1]/100*80), replace=False)))
			index_val = sorted(list(set(list(range(self.X_trainval.shape[1]))).difference(set(index_train))))

			best_W = self.find_W(self.X_trainval[:,index_train], self.gt_trainval[index_train], self.trainval_sig, alpha, gamma) # combine train and val
			test_acc, _, cm_save = self.zsl_acc(self.X_trainval[:,index_val], best_W, self.labels_trainval[index_val], self.trainval_sig)

		elif args.method == 'val':
			best_W = self.find_W(self.X_train, self.gt_train, self.train_sig, alpha, gamma) # combine train and val
			test_acc, _, cm_save = self.zsl_acc(self.X_val, best_W, self.labels_val, self.val_sig)
			
		elif args.method == 'test':
			best_W = self.find_W(self.X_trainval, self.gt_trainval, self.trainval_sig, alpha, gamma) # combine train and val
			test_acc, _, cm_save = self.zsl_acc(self.X_test, best_W, self.labels_test, self.test_sig)
		
		#test_acc, _, cm_save = self.zsl_acc(self.X_test, best_W, self.labels_test, self.test_sig)
		if args.reduced == 'no':
			with open(f'results/{args.dataset}/ESZSL/confusion_matrix/{args.dataset}_method_{args.method}_n_att_{args.att}_cm.pickle', 'wb') as handle:
				pickle.dump(cm_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
		elif args.reduced == 'yes':
			with open(f'results/{args.dataset}/ESZSL/confusion_matrix/reduced_{args.dataset}_method_{args.method}_n_att_{args.att}_sub_{args.num_subset}_cm.pickle', 'wb') as handle:
				pickle.dump(cm_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

		with open(f'results/{args.dataset}/ESZSL/confusion_matrix/accuracy_{args.dataset}_method_{args.method}.txt', 'a') as f:
			f.write(f'{args.dataset}\t{args.method}\t{args.att}\t{test_acc}\t{args.reduced}\t{args.num_subset}\n')
		
		print('Test Acc:{}'.format(test_acc))
		print(args.method)

if __name__ == '__main__':
	
	args = parser.parse_args()
	print('Dataset : {}\n'.format(args.dataset))
	
	clf = ESZSL()
	
	if args.mode=='train': 
		args.alpha, args.gamma = clf.fit()
	
	clf.evaluate(args.alpha, args.gamma)


