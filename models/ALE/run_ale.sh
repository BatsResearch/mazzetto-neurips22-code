#!/bin/bash
for seed in 42 20 32 43 53
    do
    for att in $(seq 1 1 15)
        do
            # aPY
            python attr_ale.py -data APY -norm L2 -lr 0.04 -seed $seed -att $att -method test
            python attr_ale.py -data APY -norm L2 -lr 0.04 -seed $seed -att $att -method val
            python attr_ale.py -data APY -norm L2 -lr 0.04 -seed $seed -att $att -method trainval
            
            python synt_attr_ale.py -data APY -norm L2 -lr 0.04 -seed $seed -att $att -method test
            
            # AwA2
            python attr_ale.py -data AWA2 -norm L2 -lr 0.01 -seed $seed -att $att -method test
            python attr_ale.py -data AWA2 -norm L2 -lr 0.01 -seed $seed -att $att -method val
            python attr_ale.py -data AWA2 -norm L2 -lr 0.01 -seed $seed -att $att -method trainval

            python synt_attr_ale.py -data AWA2 -norm L2 -lr 0.01 -seed $seed -att $att -method test
            
            # CUB 
            python attr_ale.py -data CUB -norm L2 -lr 0.3 -seed $seed -att $att -method test
            python attr_ale.py -data CUB -norm L2 -lr 0.3 -seed $seed -att $att -method val
            python attr_ale_ccv.py -data CUB -norm L2 -lr 0.3 -seed $seed -att $att -method trainval -reduced yes -num_subset 1
            
            python synt_attr_ale.py -data CUB -norm L2 -lr 0.3 -seed $seed -att $att -method test
            
            
            # SUN
            python attr_ale.py -data SUN -norm L2 -lr 0.1 -seed $seed -att $att -method test
            python attr_ale.py -data SUN -norm L2 -lr 0.1 -seed $seed -att $att -method val
            python attr_ale.py -data SUN -norm L2 -lr 0.1 -seed $seed -att $att -method trainval -reduced yes -num_subset 1

            python synt_attr_ale.py -data SUN -norm L2 -lr 0.1 -seed $seed -att $att -method test
            
        done
    done
