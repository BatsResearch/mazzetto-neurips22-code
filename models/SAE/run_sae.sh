#!/bin/bash


for att in $(seq 1 1 15)
    do
        # aPY
        python attr_sae.py -data APY -mode val -ld1 2.0 -ld2 4.0 -att $att -method test
        python attr_sae.py -data APY -mode val -ld1 2.0 -ld2 4.0 -att $att -method val
        python attr_sae.py -data APY -mode val -ld1 2.0 -ld2 4.0 -att $att -method trainval
        
        python synt_attr_sae.py -data APY -mode val -ld1 2.0 -ld2 4.0 -att $att -method test
        
        # AwA2
        python attr_sae.py -data AWA2 -mode val -ld1 0.6 -ld2 0.2 -att $att -method test
        python attr_sae.py -data AWA2 -mode val -ld1 0.6 -ld2 0.2 -att $att -method val
        python attr_sae.py -data AWA2 -mode val -ld1 0.6 -ld2 0.2 -att $att -method trainval

        python synt_attr_sae.py -data AWA2 -mode val -ld1 0.6 -ld2 0.2 -att $att -method test
    
        # CUB
        python attr_sae.py -data CUB -mode val -ld1 100 -ld2 0.2 -att $att -method test
        python attr_sae.py -data CUB -mode val -ld1 100 -ld2 0.2 -att $att -method val
        python attr_sae.py -data CUB -mode val -ld1 100 -ld2 0.2 -att $att -method trainval -reduced yes -num_subset $sub

        python synt_attr_sae.py -data CUB -mode val -ld1 100 -ld2 0.2 -att $att -method test
    
        # SUN
        python attr_sae.py -data SUN -mode val -ld1 0.32 -ld2 0.16 -att $att -method test
        python attr_sae.py -data SUN -mode val -ld1 0.32 -ld2 0.16 -att $att -method val
        python attr_sae.py -data SUN -mode val -ld1 0.32 -ld2 0.16 -att $att -method trainval -reduced yes -num_subset $sub

        python synt_attr_sae.py -data SUN -mode val -ld1 0.32 -ld2 0.16 -att $att -method test      
    done

