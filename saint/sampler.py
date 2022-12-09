"""
GraphSAINT requires norm during sampling, so here we modify the basic MOS sampler
"""
import math
import os
import time
import torch as th
import random
import numpy as np
import dgl

import sys
sys.path.append('..')

import mylog
mlog = mylog.get_logger()

class MOSSampler(object):
    def __init__(self, node_budget, dn, g, train_nid, num_decomp=1):
        """
        :param dn: name of dataset
        :param g: full graph
        :param train_nid: ids of training nodes
        :param node_budget: expected number of sampled nodes
        :param num_decomp: expected decomposition of MOS-GCN
        """

        if os.environ['CUDA_VISIBLE_DEVICES'] == '':
            self.device = th.device('cpu')
        else:
            assert th.cuda.is_available()
            self.device = th.cuda.current_device()

        self.g = g
        self.train_g: dgl.graph = g.subgraph(train_nid)
        self.dn = dn
        self.node_budget = node_budget
        self.num_decomp = num_decomp
        self.num_train_nodes = train_nid.shape[0]


        self.num_batch = math.ceil(self.num_train_nodes / node_budget)
        self.N = 50 * self.num_batch * self.num_decomp
        self.start_idxs = np.random.choice(range(self.num_train_nodes),size=(self.N,))

        t = time.perf_counter()
        aggr_norm, loss_norm = self.__direct_norm__()
        mlog(f'Normalization time: {time.perf_counter() - t:.2f}s')
        self.train_g.ndata['l_n'] = loss_norm
        self.train_g.edata['w'] = aggr_norm


        self.__compute_degree_norm()
        self.__clear__()
        self.__sep__()
        mlog(f"The number of subgraphs is: {len(self.start_idxs)//self.num_decomp}")

    def __direct_norm__(self):
        loss_norm = th.ones((self.train_g.num_nodes(),)) * self.node_budget / self.train_g.num_nodes()
        aggr_norm = self.train_g.in_degrees()[self.train_g.edges()[1]].type(th.float32)

        return aggr_norm, loss_norm

    def __clear__(self):
        self.g = None

    def __compute_degree_norm(self):

        self.train_g.ndata['train_D_norm'] = 1. / self.train_g.in_degrees().float().clamp(min=1).unsqueeze(1)
        self.g.ndata['full_D_norm'] = 1. / self.g.in_degrees().float().clamp(min=1).unsqueeze(1)

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        self.n = 0
        return self

    def __sep__(self):
        self.ndata_fields = [x for x in self.train_g.ndata.keys() if x != dgl.NID]
        self.ndata_values = {k : self.train_g.ndata.pop(k) for k in self.ndata_fields}

    def __randperm__(self):
        t = time.perf_counter()
        rdperm = th.randperm(self.train_g.num_nodes())
        self.train_g = self.train_g.reorder_graph('custom', permute_config={'nodes_perm':rdperm})
        self.ndata_values = {k : v[rdperm] for k,v in self.ndata_values.items()}
        mlog(f"random perm train_g time: {time.perf_counter()-t:.2f}s")

    def __next__(self):
        if self.n < self.num_decomp * self.num_batch:
            st_list = self.start_idxs[self.n : self.n+self.num_decomp]
            range_list = []
            for st in st_list:
                end = st+self.node_budget//self.num_decomp
                if end > self.num_train_nodes:
                    range_list.append((st, self.num_train_nodes))
                    range_list.append((0, end-self.num_train_nodes))
                else:
                    range_list.append((st, end))

            final_ranges = combine_range(range_list)
            subg_ndata = {k : th.cat([self.ndata_values[k][st:end].to(self.device, non_blocking=True)
                for st,end in final_ranges]) for k in self.ndata_fields}

            cur_ids = np.concatenate([np.arange(st,end) for st,end in final_ranges])
            result = self.train_g.subgraph(cur_ids).to(self.device)

            result.ndata.update(subg_ndata)
            self.n += self.num_decomp
            return result
        else:
            random.shuffle(self.start_idxs)
            raise StopIteration()

def combine_range(lis):
    """
    given several ranges in the format of [(st1, end1), (st2, end2)...]
    return combined ranges without duplicate
    """
    lis.sort(key=lambda x:x[0])
    res = [lis[0]]
    for st, end in lis[1:]:
        if st < res[-1][1]:
            if end <= res[-1][1]:
                continue
            else:
                prev = list(res.pop())
                prev[1] = end
                res.append(tuple(prev))
        elif st == res[-1][1]:
            prev = list(res.pop())
            prev[1] = end
            res.append(tuple(prev))
        else:
            res.append((st, end))
    return res
