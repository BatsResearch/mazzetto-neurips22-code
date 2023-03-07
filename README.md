# Tight Lower Bounds on Worst-Case Guarantees for Zero-Shot Learning with Attributes

In this folder you can find the code to reproduce the experiments of the paper [__"Tight Lower Bounds on Worst-Case Guarantees for Zero-Shot Learning with Attributes"__](https://arxiv.org/pdf/2205.13068.pdf) published in the proceedings of NeurIPS 2022. 


## Reproduce experiments

### Download data
We used four datasets to validate our results: Awa2, CUB, SUN, and aPY. Following previous works, we adopt the splits published along with the paper [__"Zero Shot Learning: the Good, the Bad, and the Ugly"__](https://arxiv.org/pdf/1703.04394.pdf). You can download the data [here](https://drive.google.com/drive/folders/1N1T3acUmB3rsbUmYEEJKc82baV_kTWv3?usp=share_link).

Before execution unzip the data folder in the root of the repository, and create the following folder 

```
results/
  |- APY/
     |- greedy
     |- detectors
     |- DAP
        |- confusion_matrix
     |- ESZSL
        |- confusion_matrix
     |- SJE
        |- confusion_matrix
     |- SAE
        |- confusion_matrix
     |- ALE
        |- confusion_matrix
     |- DAZLE
        |- confusion_matrix
     |- Q
        |- confusion_matrix
```


Within `results/`, be sure to create a folder for each dataset.

### Requirements

To compute the bound you need to install the [IBM Optimizer](https://www.ibm.com/docs/en/icos/12.9.0?topic=docplex-python-modeling-api) for Python.

### Compare lower bound with empirical error

We compare our lower bound with the empirical error of SOTA algorithms for ZSL with attributes. You can find the original codebases for the algorithms here:

- [DAP](https://github.com/zhanxyz/Animals_with_Attributes)
- [ESZSL, SAE, ALE, SJE](https://github.com/mvp18/Popular-ZSL-Algorithms)
- [DAZLE](https://github.com/hbdat/cvpr20_DAZLE)


#### 1. Greedly select most meaningful attributes and compute the bound

For APY and AWA2 run:

```
python models/Q/greedy.py -num_attr 15 -data $DATASET -in_path / -out_path results/ -method test -reduced no -num_subset all
```

For SUN and CUB we run the same algorithm on different subsets of classes:

```
for DATASET in SUN CUB
    do
    for S in $(seq 3 1 5)
        do
        python models/Q/greedy.py -num_attr 15 -data $DATASET -in_path / -out_path results/ -method test -reduced yes -num_subset $S
    done
done
```

Reformat output files.

For APY and AWA2 run:

```
python models/Q/reformat_results.py -data $DATASET -method test -reduced no -num_subset all
```

For SUN and CUB:

```
for DATASET in SUN CUB
    do
    for S in $(seq 3 1 5)
        do
        python models/Q/reformat_results.py -data $DATASET -method test -reduced yes -num_subset $S
    done
done
```

#### 2. Run models

In the remaining sections, we show how to run each model for the APY dataset. The reader can use the same commands to run the scripts on the other datasets. Refer to  [Appendix D.3](https://arxiv.org/pdf/2205.13068.pdf) to properly set the model parameters. 

#### Run DAP

First we train the attribute detectors on the seen classes.

```
python train_attribute_detectors.py -data APY
```

Then run DAP.

```
python models/DAP/dap.py -data APY -method test -reduced no -num_subset all -num_att 15
```

#### Run ESZSL

```
for att in $(seq 1 1 15)
    do
    python models/ESZSL/attr_zsl.py -data APY -method test -alpha 3 -gamma -1 -att $att -reduced no -num_subset all
done
```

#### Run ALE

```
for att in $(seq 1 1 15)
    do
    python models/ALE/attr_ale.py -data APY -norm L2 -lr 0.04 -seed 42 -att $att -method test -reduced no -num_subset all
done
```

#### Run SAE

```
for att in $(seq 1 1 15)
    do
    python models/SAE/attr_sae.py -data APY -mode val -ld1 100 -ld2 0.2 -att $att -method test -reduced no -num_subset all
done
```

#### Run SJE

```
for att in $(seq 1 1 15)
    do
    python models/SJE/attr_sje.py -data APY -norm std -lr 0.1 -mr 4 -seed 42 -att $att -method test -reduced no -num_subset all
done
```

#### Run DAZLE

Execute the pre-processing procedure indicated in the [official repository](https://github.com/hbdat/cvpr20_DAZLE). Refer to the folder `models/DAZLE/core/` to get the dataset loaders. Then, train and evaluate the model running the following:

```
python dazle.py --dataset APY --path data --method test
```

After each run, in the folders <MODEL_NAME>/confusion_matrix you can find the model accuracy for increasing number of attributes and the confusion matrices to measure the pairwise misclassifications.

## Generate synthetic data

To generate the synthetic data, you can run the following script.

```
python generate_synth_data.py -dataset APY -method test -reduced no -num_subset all
```

You can then run each model again specifying the correct paths.

# Citation  

If you make usage of our bound in your work, plese cite us :)

```
@inproceedings{
  mazzetto2022tight,
  title={Tight Lower Bounds on Worst-Case Guarantees for Zero-Shot Learning with Attributes},
  author={Alessio Mazzetto and Cristina Menghini and Andrew Yuan and Eli Upfal and Stephen Bach},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=tzNWhvOomsK}
}
```
