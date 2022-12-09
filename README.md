# Feature-Oriented Sampling for Fast and Scalable GNN Training

Accepted paper of ICDM 2022.

## envs

```
pip install -r requirements.txt
```

## download dataset

* for reddit and ogbn-products: come with dgl/ogb packages
    * reddit: `dgl.data.RedditDataset(raw_dir=ROOT_PATH, self_loop=True)`
    * ogbn-products: `ogb.nodeproppred.DglNodePropPredDataset(root=ROOT_PATH, name='ogbn-products')`
* for yelp and amazon: download from [GraphSAINT repo](https://github.com/GraphSAINT/GraphSAINT)

we denote the root directory that contains dataset as <ROOT_PATH>

## reproduce 


For FOS-SAINT, enter the `saint` folder and execute:
```
OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=0 python train.py --root <ROOT_PATH> --n-hidden 128 --dataset reddit --lr 0.01 --dropout 0.2 --node-budget 9000 --decomp 4 --n-epochs 40 --val-every 1
OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=0 python train.py --root <ROOT_PATH> --n-hidden 512 --dataset yelp --lr 0.1 --dropout 0.1 --node-budget 29000 --decomp 8 --n-epochs 90 --val-every 1
OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=0 python train.py --root <ROOT_PATH> --n-hidden 512 --dataset ogbn-products --lr 0.001 --dropout 0.3 --node-budget 36000 --decomp 4 --n-epochs 200 --val-every 10
OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=0 python train.py --root <ROOT_PATH> --n-hidden 512 --dataset amazon --lr 0.1 --dropout 0.1 --node-budget 40000 --decomp 4 --n-epochs 40 --val-every 1
```

Similar commands are given in respective folders for FOS-GCN and FOS-SAGE.

Note that the validation/test phase is sometimes implemented on CPU only (come with the original implementation) and very slow. Thus if you want to verify the pure training time or GPU/CPU utilization during training, remember to comment out the valid/test code or set `args.val_every` to a very large value.

All scripts are developed based on official examples from DGL repo (https://github.com/dmlc/dgl/tree/master/examples Commit: a04a8d).
