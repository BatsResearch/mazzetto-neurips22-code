# Tight Lower Bounds on Worst-Case Guarantees for Zero-Shot Learning with Attributes

In this folder you can find the code to reproduce the experiments of the paper __"Tight Lower Bounds on Worst-Case Guarantees for Zero-Shot Learning with Attributes"__ published in the proceedings of NeurIPS 2022. 

Read our paper [here](https://arxiv.org/pdf/2205.13068.pdf).

## Repository organization

1. Download data
2. Models implementation
3. Instructions to run the code
4. Visualize results

## 1. Download data

We used four datasets to validate our results: Awa2, CUB, SUN, and aPY. Following previous works, we adopt the splits published along with the paper [__"Zero Shot Learning: the Good, the Bad, and the Ugly"__](https://arxiv.org/pdf/1703.04394.pdf). You can download the data [here](https://drive.google.com/drive/folders/1N1T3acUmB3rsbUmYEEJKc82baV_kTWv3?usp=share_link).

After you download the zip folder, unzip it in the root of the repository for execution.

## 2. Models implementation

We compare our lower bound with the empirical error of SOTA algorithms for ZSL with attributes. In the folder `models`, you can find the implementation of each of them adapted to our experiments. More general versions of the algorithms can be found at the following links:

- [DAP](https://github.com/zhanxyz/Animals_with_Attributes)
- [ESZSL, SAE, ALE, SJE](https://github.com/mvp18/Popular-ZSL-Algorithms)
- [DAZLE](https://github.com/hbdat/cvpr20_DAZLE)

For the execution, be sure to create the folder `confusion_matrix` in each model's subfolder.


