import os
import dgl
import time
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F

from modules import GCNNet
from sampler import MOSSampler
from utils import SavingLogger, evaluate, save_log_dir

import sys
sys.path.append('..')

import mylog
from load_4graph import load_original
mlog = mylog.get_logger()

def main(args):
    np.random.seed(0)
    torch.manual_seed(0)

    multilabel_data = set(['yelp', 'amazon'])
    multilabel = args.dataset in multilabel_data

    g, n_classes = load_original(args.dataset, args.root)

    #random perm all nodes
    rand_perm = np.random.permutation(np.arange(g.num_nodes())).astype(np.int64)
    g = g.reorder_graph('custom', permute_config={'nodes_perm':rand_perm}, store_ids=False)
    mlog("input graph has been randomly reordered")

    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]

    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['label']


    in_feats = g.ndata['feat'].shape[1]
    n_nodes = g.num_nodes()
    n_edges = g.num_edges()

    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    # set device for dataset tensors
    if args.gpu < 0:
        cuda = False
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        cuda = True
        torch.cuda.set_device(args.gpu)

    mlog("""----Data statistics------'
    #Nodes %d
    #Edges %d
    #Classes/Labels (multi binary labels) %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
          (n_nodes, n_edges, n_classes,
           n_train_samples,
           n_val_samples,
           n_test_samples))

    # load sampler
    subg_iter = MOSSampler(args.node_budget, args.dataset, g,
                                 train_nid, args.decomp)

    mlog(f"subg_iter prepared")

    mlog(f"labels shape: {g.ndata['label'].shape}")
    mlog(f"features shape: {g.ndata['feat'].shape}")

    model = GCNNet(
        in_dim=in_feats,
        hid_dim=args.n_hidden,
        out_dim=n_classes,
        arch=args.arch,
        dropout=args.dropout,
        batch_norm=not args.no_batch_norm,
        aggr=args.aggr
    )

    if cuda:
        model.cuda()

    log_dir = save_log_dir(args)
    s_logger = SavingLogger(os.path.join(log_dir, 'args'))
    s_logger.write(args)

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # set train_nids to cuda tensor
    if cuda:
        if isinstance(train_nid, np.ndarray):
            train_nid = torch.from_numpy(train_nid).cuda()
        elif isinstance(train_nid, torch.Tensor):
            train_nid = train_nid.cuda()
        mlog("GPU memory allocated before training(MB) {:.2f}".format(
              torch.cuda.memory_allocated(device=train_nid.device) / 1024 / 1024))
    start_time = time.time()
    best_f1 = -1

    converge_time = []
    converge_val_f1 = []
    for epoch in range(args.n_epochs):
        epoch_tic = time.time()
        for j, subg in enumerate(subg_iter):
            if cuda:
                model.cuda()
                subg = subg.to(torch.cuda.current_device())
            model.train()
            # forward
            pred = model(subg)

            batch_labels = subg.ndata['label']

            if multilabel:
                loss = F.binary_cross_entropy_with_logits(pred, batch_labels, reduction='sum',
                                                          weight=subg.ndata['l_n'].unsqueeze(1))
            else:
                loss = F.cross_entropy(pred, batch_labels, reduction='none')
                loss = (subg.ndata['l_n'] * loss).sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if j == len(subg_iter) - 1:
                epoch_time = time.time() - epoch_tic
                mlog(f"epoch: {epoch+1}/{args.n_epochs}, epoch time: {epoch_time:.4f}s, training loss: {loss.item():.4f}")

        # evaluate
        #if epoch >= args.n_epochs*0.2 and epoch % args.val_every == 0:
        if epoch % args.val_every == 0:
            val_f1_mic, val_f1_mac = evaluate(
                model, g, labels, val_mask, multilabel)
            mlog("Val F1-mic {:.4f}, Val F1-mac {:.4f}".format(val_f1_mic, val_f1_mac))

            converge_time.append(epoch_time)
            converge_val_f1.append(val_f1_mic)

            if val_f1_mic > best_f1:
                best_f1 = val_f1_mic
                mlog(f'new best val f1: {best_f1}')
                torch.save(model.state_dict(), os.path.join(
                    log_dir, 'best_model.pkl'))

    # use best val to test
    model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pkl')))
    test_f1_mic, _ = evaluate(
        model, g, labels, test_mask, multilabel)
    mlog(f"BestVal Test F1-mic {test_f1_mic:.4f}")
    #mlog(converge_time)
    #mlog(converge_val_f1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data source params
    parser.add_argument("--dataset", type=str, choices=['ogbn-products', 'yelp', 'amazon', 'reddit'],
                        default='ogbn-products')
    parser.add_argument("--root", type=str, default="/home/data/xzhanggb/datasets",
                        help="root directory of all datasets")

    # cuda params
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index. Default: 0")

    # sampler params
    parser.add_argument("--node-budget", type=int, default=10000,
                        help="batch size of MOS-GCN")
    parser.add_argument("--decomp", type=int, default=1,
                        help="number of decomposition, set to 1 for vanilla MOS-GCN")

    # model params
    parser.add_argument("--n-hidden", type=int, default=512,
                        help="Number of hidden gcn units")
    parser.add_argument("--arch", type=str, default="1-1-0",
                        help="Network architecture. 1 means an order-1 layer (self feature plus 1-hop neighbor "
                             "feature), and 0 means an order-0 layer (self feature only)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("--no-batch-norm", action='store_true',
                        help="Whether to use batch norm")
    parser.add_argument("--aggr", type=str, default="concat", choices=['mean', 'concat'],
                        help="How to aggregate the self feature and neighbor features")

    # training params
    parser.add_argument("--n-epochs", type=int, default=40,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--val-every", type=int, default=1,
                        help="Frequency of evaluation on the validation set in number of epochs")
    parser.add_argument("--log-dir", type=str, default='logs',
                        help="Log file/best model will be saved to ./log_dir")


    args = parser.parse_args()

    mlog(args)
    main(args)
