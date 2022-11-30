
"""
This code is adapted from https://github.com/hbdat/cvpr20_DAZLE

"""

import copy
import os,sys
import argparse
pwd = os.getcwd()
parent = '/'.join(pwd.split('/')[:-1])
sys.path.insert(0,parent)
os.chdir(parent)
#%%
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from core.DAZLE import DAZLE
from core.AWA2DataLoader import AWA2DataLoader
from core.SUNDataLoader import SUNDataLoader
from core.APYDataLoader import APYDataLoader
from core.CUBDataLoader import CUBDataLoader
from core.helper_func import eval_zs_gzsl,visualize_attention#,get_attribute_attention_stats
from global_setting import NFS_path
import importlib
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle


parser = argparse.ArgumentParser(description="DAZLE")
parser.add_argument('--dataset', help='choose between APY, AWA2, CUB, SUN', default='AWA2', type=str)
parser.add_argument('--path', help='path of data folder', default='data', type=str)
parser.add_argument('--method', help='trainval, test, val', default='trainval', type=str)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

dataset = args.dataset
if dataset == 'SUN':
    dataloader = SUNDataLoader(NFS_path, device, 
                               is_scale=False,
                               is_balance=True)
elif dataset == 'CUB':
    dataloader = CUBDataLoader(NFS_path, device,
                               is_balance=False)
elif dataset == 'AWA2':
    dataloader = AWA2DataLoader(NFS_path,device)
elif dataset == 'APY':
    dataloader = APYDataLoader(NFS_path,device)

def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return lr


ds = copy.deepcopy(dataloader)


data_folder = f'{args.path}/{dataset}'
method = args.method
attributes = []
if method == 'trainval':
    with open(f'{data_folder}/trainval_greedy.txt', 'r') as f:
        for a in f:
            attributes += [int(a.strip())]
elif method == 'val':
    with open(f'{data_folder}/val_greedy.txt', 'r') as f:
        for a in f:
            attributes += [int(a.strip())]
elif method == 'test':
    with open(f'{data_folder}/test_greedy.txt', 'r') as f:
        for a in f:
            attributes += [int(a.strip())]
            
seed_all_acc = []
for seed in [0, 50, 100, 214, 800]
    for n_att in range(1,len(attributes)+1):
        print(f"Num of attributes: {n_att}")
        attributes_ = attributes[:n_att]
        dataloader.att = ds.att[:,attributes_]
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        batch_size = 50
        nepoches = 10
        niters = dataloader.ntrain * nepoches//batch_size
        dim_f = 2048
        dim_v = 300
        init_w2v_att = dataloader.w2v_att[attributes_,:]
        att = dataloader.normalize_att#
        att[att<0] = 0
        att = att[:,attributes_]
        normalize_att = ds.normalize_att
        normalize_att = normalize_att[:,attributes_]
        #assert (att.min().item() == 0 and att.max().item() == 1)
        print(att.shape, normalize_att.shape)

        trainable_w2v = True
        lambda_ = 0.1#0.1
        bias = 0
        prob_prune = 0
        uniform_att_1 = False
        if dataset == 'SUN':
            uniform_att_2 = True
        else:
            uniform_att_2 = False

        seenclass = dataloader.seenclasses
        unseenclass = dataloader.unseenclasses
        desired_mass = 1#unseenclass.size(0)/(seenclass.size(0)+unseenclass.size(0))
        report_interval = niters//nepoches#10000//batch_size#

        model = DAZLE.DAZLE(dim_f,dim_v,init_w2v_att,att,normalize_att,
                    seenclass,unseenclass,
                    lambda_,
                    trainable_w2v,normalize_V=True,normalize_F=True,is_conservative=True,
                    uniform_att_1=uniform_att_1,uniform_att_2=uniform_att_2,
                    prob_prune=prob_prune,desired_mass=desired_mass, is_conv=False,
                    is_bias=True)
        model.to(device)

        setup = {'pmp':{'init_lambda':0.1,'final_lambda':0.1,'phase':0.8},
                 'desired_mass':{'init_lambda':-1,'final_lambda':-1,'phase':0.8}}
        print(setup)

        params_to_update = []
        params_names = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                params_names.append(name)
                print("\t",name)
        #%%
        lr = 0.0001
        weight_decay = 0.0001#0.000#0.#
        if dataset == 'SUN' or dataset == 'CUB':
            momentum = 0.9
        else:
            momentum = 0.#
        #%%
        lr_seperator = 1
        lr_factor = 1
        print('default lr {} {}x lr {}'.format(params_names[:lr_seperator],lr_factor,params_names[lr_seperator:]))
        optimizer  = optim.RMSprop( params_to_update ,lr=lr,weight_decay=weight_decay, momentum=momentum)
        print('-'*30)
        print('learing rate {}'.format(lr))
        print('trainable V {}'.format(trainable_w2v))
        print('lambda_ {}'.format(lambda_))
        print('optimized seen only')
        print('optimizer: RMSProp with momentum = {} and weight_decay = {}'.format(momentum,weight_decay))
        print('-'*30)

        best_performance = [0,0,0,0]
        for i in range(0,niters):
            model.train()
            optimizer.zero_grad()

            batch_label, batch_feature, batch_att = dataloader.next_batch(batch_size)
            out_package = model(batch_feature)

            in_package = out_package
            in_package['batch_label'] = batch_label

            out_package=model.compute_loss(in_package)
            loss,loss_CE,loss_cal = out_package['loss'],out_package['loss_CE'],out_package['loss_cal']

            loss.backward()
            optimizer.step()
            if i%report_interval==0:
                print('-'*30)
                acc_seen, acc_novel, H, acc_zs, preds = eval_zs_gzsl(dataloader,model,device,bias_seen=-bias,bias_unseen=bias)

                if H > best_performance[2]:
                    best_performance = [acc_seen, acc_novel, H, acc_zs, preds]
                stats_package = {'iter':i, 'loss':loss.item(), 'loss_CE':loss_CE.item(),
                                 'loss_cal': loss_cal.item(),
                                 'acc_seen':best_performance[0], 'acc_novel':best_performance[1], 'H':best_performance[2], 'acc_zs':best_performance[3]}

                print(stats_package)

        seed_all_acc.append(best_performance)
        with open(f'results/{args.dataset}/DAZLE/confusion_matrix/accuracy_{dataset}_method_{method}.txt', 'a') as f:
            f.write(f'{dataset}\t{method}\t{n_att}\t{best_performance[3]}')
        print(seed)
        print('\n\n\n\n')


