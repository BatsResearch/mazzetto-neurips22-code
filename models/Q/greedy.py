import os
import time
import argparse

import numpy as np
import pandas as pd

from adversarial_helper import load_classes, load_class_attribute_matrix, \
                                solve_next, get_indices


parser = argparse.ArgumentParser()
parser.add_argument('-num_attr', required=True, help='enter number of attributes', type = float)
parser.add_argument('-data', required=True, help='choose between APY, AWA2, AWA1, CUB, SUN', type = str)

parser.add_argument('-in_path', required=True, help='input data path: /', type = str)
parser.add_argument('-out_path', required=True, help='output data path: results/', type = str)

parser.add_argument('-method', required=True, help='trainval, val, test', type = str)
parser.add_argument('-reduced', required=True, help='yes to reduce the number of classes, no otherwise', type = str)
parser.add_argument('-num_subset', required=True, help='number of the group of subclasses', type = str)


if __name__ == '__main__':

    args = parser.parse_args()

    #offset = args.offset
    num_attr = int(args.num_attr)

    dataset_folder = f'data/class_attribute_matrices/{args.data}'
    output_path = f'{args.out_path}/{args.data}/greedy'

    all_classes, unseen_classes = load_classes(dataset_folder, args.method, args.reduced, args.num_subset)
    print(all_classes, unseen_classes)
    
    M = load_class_attribute_matrix(all_classes, unseen_classes, dataset_folder, method=args.method)
    print(M.shape)
    i = num_attr
    if args.method == 'trainval':
        if args.reduced == 'no':
            greedy_file_previous = f'{output_path}/trainval_ans_{i}.txt'
        elif args.reduced == 'yes':
            greedy_file_previous = f'{output_path}/reduced_trainval_ans_{i}_sub_{args.num_subset}.txt'
    elif args.method == 'val':
        greedy_file_previous = f'{output_path}/val_ans_{i}.txt'
    elif args.method == 'test':
        greedy_file_previous = f'{output_path}/test_ans_{i}.txt'

    while (not os.path.isfile(greedy_file_previous)) and i > 0:
        i -= 1
        print(f'trying i: {i}')
        if args.method == 'trainval':
            if args.reduced == 'no':
                greedy_file_previous = f'{output_path}/trainval_ans_{i}.txt'
            elif args.reduced == 'yes':
                greedy_file_previous = f'{output_path}/reduced_trainval_ans_{i}_sub_{args.num_subset}.txt'
        elif args.method == 'val':
            greedy_file_previous = f'{output_path}/val_ans_{i}.txt'
        elif args.method == 'test':
            greedy_file_previous = f'{output_path}/test_ans_{i}.txt'

    if i == 0:
        indices_current = [] 
    else:
        indices_current = get_indices(greedy_file_previous)
    
    while i < num_attr:
        ans, indices_current = solve_next(M, indices_current, output_path,
                                          args.method, args.reduced, args.num_subset)
        i += 1
