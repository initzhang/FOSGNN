import torch as th
import torch.nn as nn
import torch.functional as F
import dgl
import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import tqdm

class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()

        assert n_layers > 1
        # input layer
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layer
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(dglnn.GraphConv(n_hidden, n_classes))

        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, x):
        if blocks[0].is_block:
            # normal gcn
            h = x
            for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.dropout(h)
        else:
            # transductive mos
            subg, block = blocks
            h = x
            for layer in self.layers[:-1]:
                h = layer(subg, h)
                h = self.dropout(h)

            # slice out the input nodes for block
            internal_input_nids = block.ndata[dgl.NID]['_N'].to('cuda')
            h = self.layers[-1](block, h[internal_input_nids])
        return h

    def inference(self, g, x, device, batch_size, num_workers):
        """
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()).to(g.device),
                sampler,
                device=device if num_workers == 0 else None,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers)

            #for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y
