#!/bin/bash
for seed in 42 20 32 43 53
    do
    for att in $(seq 1 1 15)
        do
            # aPY
            python attr_sje.py -data APY -norm L2 -lr 0.01 -mr 1.5 -seed $seed -att $att -method test
            python attr_sje.py -data APY -norm L2 -lr 0.01 -mr 1.5 -seed $seed -att $att -method val
            python attr_sje.py -data APY -norm L2 -lr 0.01 -mr 1.5 -seed $seed -att $att -method trainval

            python synt_attr_sje.py -data APY -norm L2 -lr 0.01 -mr 1.5 -seed $seed -att $att -method test

            # AwA2
            python attr_sje.py -data AWA2 -norm L2 -lr 1 -mr 2.5 -seed $seed -att $att -method test
            python attr_sje.py -data AWA2 -norm L2 -lr 1 -mr 2.5 -seed $seed -att $att -method val
            python attr_sje.py -data AWA2 -norm L2 -lr 1 -mr 2.5 -seed $seed -att $att -method trainval

            python synt_attr_sje.py -data AWA2 -norm L2 -lr 1 -mr 2.5 -seed $seed -att $att -method test
            
            # CUB
            python attr_sje.py -data CUB -norm std -lr 0.1 -mr 4 -seed $seed -att $att -method test
            python attr_sje.py -data CUB -norm std -lr 0.1 -mr 4 -seed $seed -att $att -method val
            python attr_sje_ccv.py -data CUB -norm std -lr 0.1 -mr 4 -seed $seed -att $att -method trainval -reduced yes -num_subset 1

            python synt_attr_sje.py -data CUB -norm std -lr 0.1 -mr 4 -seed $seed -att $att -method test
            
            # SUN
            python attr_sje.py -data SUN -norm std -lr 1 -mr 2 -seed $seed -att $att -method test
            python attr_sje.py -data SUN -norm std -lr 1 -mr 2 -seed $seed -att $att -method val
            python attr_sje.py -data SUN -norm std -lr 1 -mr 2 -seed $seed -att $att -method trainval
            
            python synt_attr_sje.py -data SUN -norm std -lr 1 -mr 2 -seed $seed -att $att -method test
        done
    done
