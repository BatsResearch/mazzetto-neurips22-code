#!/bin/bash

for att in $(seq 1 1 15)
    do
        # aPY
        python attr_zsl.py -data APY -mode test -alpha 3 -gamma -1  -att $att -method test
        python attr_zsl.py -data APY -mode test -alpha 3 -gamma -1  -att $att -method val
        python attr_zsl.py -data APY -mode test -alpha 3 -gamma -1  -att $att -method trainval

        python synt_attr_zsl.py -data APY -mode test -alpha 3 -gamma -1  -att $att -method test
        
        # CUB
        python attr_zsl.py -data CUB -mode test -alpha 3 -gamma -1  -att $att -method test
        python attr_zsl.py -data CUB -mode test -alpha 3 -gamma -1  -att $att -method val
        python attr_zsl.py -data CUB -mode test -alpha 3 -gamma -1  -att $att -method trainval -reduced yes -num_subset $sub

        python synt_attr_zsl.py -data CUB -mode test -alpha 3 -gamma -1  -att $att -method test

        # SUN
        python attr_zsl.py -data SUN -mode test -alpha 3 -gamma 2  -att $att -method test
        python attr_zsl.py -data SUN -mode test -alpha 3 -gamma 2  -att $att -method val
        python attr_zsl.py -data SUN -mode test -alpha 3 -gamma 2  -att $att -method trainval -reduced yes -num_subset $sub

        python synt_attr_zsl.py -data SUN -mode test -alpha 3 -gamma 2  -att $att -method test
        
        # AwA2
        python attr_zsl.py -data AWA2 -mode test -alpha 3 -gamma 0  -att $att -method test
        python attr_zsl.py -data AWA2 -mode test -alpha 3 -gamma 0  -att $att -method val
        python attr_zsl.py -data AWA2 -mode test -alpha 3 -gamma 0  -att $att -method trainval

        python synt_attr_zsl.py -data AWA2 -mode test -alpha 3 -gamma 0  -att $att -method test
    done

