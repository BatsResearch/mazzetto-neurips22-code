import os
import argparse

import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('-data', required=True, help='choose between APY, AWA2, AWA1, CUB, SUN, LAD', type = str)
parser.add_argument('-method', required=True, help='trainval, val, test', type = str)
parser.add_argument('-reduced', help='yes to reduce the number of classes, no otherwise', type = str, default='nan')
parser.add_argument('-num_subset', help='number of the group of subclasses', type = str, default='nan')

def find_max(input_path, method, reduced, num_subset):

    list_dir = os.listdir(input_path)
    if reduced == 'no':
        start = method
        end = '.txt'
    elif reduced == 'yes':
        start = 'reduced'
        end = f'sub_{num_subset}.txt'
    
    max_att = 0
    for i in list_dir:
        if (i.startswith(start)) and (i.endswith(end)):

            if reduced == 'no':
                s_name = int(i.split('_')[-1].split('.')[0])
            else:
                s_name = int(i.split('_')[3])
            if s_name > max_att:
                max_att = s_name
            else:
                continue
    return max_att

def get_list_attributes(input_path, method, max_att, reduced, num_subset):

    if reduced == 'no':
        path = f'{input_path}{method}_ans_{max_att}.txt'
       
    elif reduced == 'yes':
        path = f'{input_path}reduced_{method}_ans_{max_att}_sub_{num_subset}.txt'
        
    with open(path, 'r') as f:
        for l in f:
            if l.startswith('indices'):
                list_indices = l.strip().split('[')[-1][:-1].split(',')
                list_indices = [int(i.strip()) for i in list_indices]
    
    return list_indices

def write_list_on_file(output_path, file_name, lst):
    with open(f'{output_path}{file_name}', 'w') as f:
        for idx in lst:
            f.write(f'{idx}\n')


if __name__ == '__main__':

    args = parser.parse_args()
    method = args.method
    reduced = args.reduced
    num_subset = args.num_subset
    
    input_path = f'results/{args.data}/greedy/'

    max_att = find_max(input_path, method, reduced, num_subset)
    print(max_att)
    list_indices = get_list_attributes(input_path, method, max_att, reduced, num_subset)
    print(list_indices)
                
    output_path = f'data/{args.data}/'
    if reduced == 'no':
        file_name = f'{method}_greedy.txt'
    elif reduced == 'yes':
        file_name = f'reduced_{method}_greedy_sub_{num_subset}.txt'
    

    write_list_on_file(output_path, file_name, list_indices)
    print('Write attributes..')

    path_all = f'{input_path}/greedy_results.txt'
    with open(path_all, 'a') as f_write:
        for a in range(1, max_att+1):
            if reduced == 'no':
                path = f'{input_path}{method}_ans_{a}.txt'
            elif reduced == 'yes':
                path = f'{input_path}reduced_{method}_ans_{a}_sub_{num_subset}.txt'
            
            with open(path, 'r') as f:
                for l in f:
                    if l.startswith('solver'):
                        Q_value = float(l.strip().split(':')[-1].strip())
                        f_write.write(f"{method}\t{a}\t{Q_value}\t{reduced}\t{num_subset}\n")
    print('Write results..')


