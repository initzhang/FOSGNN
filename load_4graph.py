import dgl
import time
import json
import os.path
import torch
import torch as th
import numpy as np
from dgl import backend as F1
from dgl.convert import from_scipy
from dgl.data.utils import generate_mask_tensor
from sklearn.preprocessing import StandardScaler
from scipy import sparse as sp
import mylog
mlog = mylog.get_logger()

def load_original(name, root='/home/data/xzhanggb/datasets'):
    assert name in ['ogbn-products', 'yelp', 'amazon', 'reddit']
    if name == 'ogbn-products':
        return load_ogb(name,root)
    elif name == 'reddit':
        return load_reddit(root)
    else:
        return load_yelp_amazon(name,root)

def load_reddit(root):
    tic = time.time()
    from dgl.data import RedditDataset
    data = RedditDataset(raw_dir=root, self_loop=True)
    g = data[0]
    mlog(f'finish loading Reddit, time elapsed: {time.time() - tic:.2f}s')
    return g, data.num_classes

def load_ogb(name, root):
    tic = time.time()
    from ogb.nodeproppred import DglNodePropPredDataset

    mlog(f'load {name}')
    data = DglNodePropPredDataset(root=root, name=name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['label'] = labels
    in_feats = graph.ndata['feat'].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    mlog(f"finish loading {name}, time elapsed: {time.time() - tic:.2f}s")
    return graph, num_labels

def load_yelp_amazon(name, root):
    tic = time.time()
    assert name in ["yelp", 'amazon']
    root = root + '/' + name
    mlog(f"loading {name}")
    adj_full = sp.load_npz(f'{root}/adj_full.npz').astype(bool)
    adj_train = sp.load_npz(f'{root}/adj_train.npz').astype(bool)
    role = json.load(open(f'{root}/role.json'))
    feats = np.load(f'{root}/feats.npy')
    # ---- normalize feats ----
    train_nodes = np.array(role['tr'])
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    # ---- prepare labels -----
    if os.path.isfile(f"{root}/labels.npy"):
        labels = np.load(f"{root}/labels.npy")
        num_classes = labels.shape[1]
    else:
        class_map = json.load(open(f'{root}/class_map.json'))
        class_map = {int(k): v for k, v in class_map.items()}
        assert len(class_map) == feats.shape[0]
        num_vertices = adj_full.shape[0]
        if isinstance(list(class_map.values())[0],list):
            num_classes = len(list(class_map.values())[0])
            labels = np.zeros((num_vertices, num_classes))
            for k, v in class_map.items():
                labels[k] = np.array(v)
        else:
            num_classes = max(class_map.values()) - min(class_map.values()) + 1
            labels = np.zeros((num_vertices, num_classes))
            offset = min(class_map.values())
            for k,v in class_map.items():
                labels[k][v-offset] = 1

    # ---- prepare graph -----
    def create_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return mask
    g = from_scipy(adj_full)
    train_mask = create_mask(role['tr'], labels.shape[0])
    val_mask = create_mask(role['va'], labels.shape[0])
    test_mask = create_mask(role['te'], labels.shape[0])
    g.ndata['train_mask'] = generate_mask_tensor(train_mask)
    g.ndata['val_mask'] = generate_mask_tensor(val_mask)
    g.ndata['test_mask'] = generate_mask_tensor(test_mask)
    g.ndata['feat'] = F1.tensor(feats, dtype=F1.data_type_dict['float32'])
    g.ndata['label'] = F1.tensor(labels, dtype=F1.data_type_dict['float32'])
    mlog(f"finish loading {name}, time elapsed: {time.time() - tic:.2f}s")
    return g, num_classes

def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g

if __name__ == '__main__':
    g, _ = load_original('ogbn-products')
    print(g.formats())
    print(g.num_nodes())
    print(g.num_edges())




