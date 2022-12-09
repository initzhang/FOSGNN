import os
import dgl
import torch
import numpy as np
from sklearn.metrics import f1_score

class SavingLogger(object):
    '''A custom logger to log stdout to a logging file.'''
    def __init__(self, path):
        """Initialize the logger.

        Parameters
        ---------
        path : str
            The file path to be stored in.
        """
        self.path = path

    def write(self, s):
        with open(self.path, 'a') as f:
            f.write(str(s))
        return

def save_log_dir(args):
    prefix = f'./{args.log_dir}/{args.dataset}'
    infix = f'{args.node_budget}_{args.decomp}'
    suffix = f'{args.arch}_{args.n_hidden}_{args.lr}_{args.dropout}'

    log_dir = f"{prefix}_{infix}_{suffix}"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def calc_f1(y_true, y_pred, multilabel):
    if multilabel:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="micro"), \
        f1_score(y_true, y_pred, average="macro")

def evaluate(model, g, labels, mask, multilabel=False):
    model.cpu()
    model.eval()
    with torch.no_grad():
        logits = model(g)
        logits = logits[mask]
        labels = labels[mask]
        f1_mic, f1_mac = calc_f1(labels.cpu().numpy(),
                                 logits.cpu().numpy(), multilabel)
        return f1_mic, f1_mac
