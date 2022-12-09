import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
from sklearn.metrics import f1_score
from model import SAGE

import sys
sys.path.append('..')

import mylog
mlog = mylog.get_logger()
from load_4graph import load_original, inductive_split
from mos_sampler import MOSNeighborSamplerInductive

def compute_acc(pred, labels, multilabel=False):
    """
    Compute the f1-micro of prediction given the labels.
    """
    y_pred = pred.cpu()
    labels = labels.cpu()
    if multilabel:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    else:
        y_pred = th.argmax(y_pred, dim=1)

    return f1_score(labels, y_pred, average="micro")

def evaluate(model, g, nfeat, labels, nids, device, multilabel):
    """
    Evaluate the model on the validation/test set specified by nids.
    g : The entire graph.
    nfeat : The features of all the nodes.
    labels : The labels of all the nodes.
    nids : list of node Ids for validation/test
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device, args.valbs, args.valwk)
    model.train()
    if isinstance(nids, list):
        accs = [compute_acc(pred[nid], labels[nid], multilabel) for nid in nids]
    else:
        assert isinstance(nids, th.Tensor)
        accs = compute_acc(pred[nids], labels[nids], multilabel)
    return accs

def load_subtensor_mos(nfeat, labels, final_ranges, device, seeds=None):
    batch_inputs = th.cat([nfeat[st:end].to(device) for st,end in final_ranges])
    if seeds is None:
        batch_labels = th.cat([labels[st:end].to(device) for st,end in final_ranges])
    else:
        batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

#### Entry point
def run(args, device, data):
    multilabel = False
    if args.dataset in ['yelp', 'amazon']:
        multilabel = True

    # Unpack data
    n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
    val_nfeat, val_labels, test_nfeat, test_labels = data
    in_feats = train_nfeat.shape[1]
    train_mask = train_g.ndata['train_mask']
    train_nid = th.nonzero(train_mask, as_tuple=True)[0]
    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(test_g.ndata['test_mask'], as_tuple=True)[0]

    # dataloader
    dataloader = MOSNeighborSamplerInductive(args.node_budget, args.num_layers, train_g, None, args.decomp)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    if multilabel:
        loss_fcn = nn.BCEWithLogitsLoss()
    else:
        loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # best 
    best_eval = 0
    best_test = 0

    # Training loop
    for epoch in range(1, args.num_epochs+1):
        tic = time.time()

        for step, (final_ranges, seeds, subgs) in enumerate(dataloader):
            
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor_mos(train_nfeat, train_labels, final_ranges, device, seeds)

            subgs = subgs.int().to(device)

            # Compute loss and prediction
            batch_pred = model(subgs, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        toc = time.time()
        mlog('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Time(s) {:.4f}'.format(
            epoch, step, loss.item(), toc-tic))

        if epoch % args.eval_every == 0:
            eval_acc = evaluate(model, val_g, val_nfeat, val_labels, val_nid, device, multilabel)
            test_acc = evaluate(model, test_g, test_nfeat, test_labels, test_nid, device, multilabel)
            mlog('Eval Acc: {:.4f}, Test Acc: {:.4f}'.format(eval_acc, test_acc))
            if eval_acc > best_eval:
                mlog('new best eval: {:.4f}'.format(eval_acc))
                best_eval = eval_acc
                best_test = test_acc

    mlog(f'BestVal Test F1-mic: {best_test:.4f}')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0, help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='ogbn-products')
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=512)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--node-budget', type=int, default=40000)
    argparser.add_argument('--decomp', type=int, default=8)
    argparser.add_argument('--log-every', type=int, default=100) # useless, now log every epoch
    argparser.add_argument('--eval-every', type=int, default=10)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--dropout', type=float, default=0.3)
    argparser.add_argument('--valbs', type=int, default=20000) # validation batch size
    argparser.add_argument('--valwk', type=int, default=4) # validation workers
    argparser.add_argument('--reorder', action='store_true')
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    g, n_classes = load_original(args.dataset)

    if args.reorder:
        # for datasets that have little training nodes
        rand_perm = th.randperm(g.num_nodes())
        g = g.reorder_graph('custom', permute_config={'nodes_perm':rand_perm}, store_ids=False)
        mlog('rand reorder finish')

    train_g, val_g, test_g = inductive_split(g)
    train_nfeat = train_g.ndata.pop('feat')
    val_nfeat = val_g.ndata.pop('feat')
    test_nfeat = test_g.ndata.pop('feat')
    train_labels = train_g.ndata.pop('label')
    val_labels = val_g.ndata.pop('label')
    test_labels = test_g.ndata.pop('label')


    # Pack data
    data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
           val_nfeat, val_labels, test_nfeat, test_labels

    mlog(args)
    run(args, device, data)
