import dgl
import torch
import numpy as np

class MOSNeighborSamplerInductive(dgl.dataloading.Sampler):
    def __init__(self, node_budget, num_layers, g, train_mask, bulk_decomp=1):
        super().__init__()
        # direct arguments
        self.node_budget = node_budget
        self.num_layers = num_layers
        self.g = g
        self.train_mask = train_mask
        self.bulk_decomp = bulk_decomp

        # infered arguments
        assert self.train_mask is None
        self.total_nodes = self.g.num_nodes()
        self.num_batches = self.total_nodes // self.node_budget + 1

        # utilities
        self.N = 20 * self.num_batches * self.bulk_decomp # num of small blocks
        self.start_idxs = np.random.choice(range(self.g.num_nodes()),size=(self.N,))
        self.start_idxs = torch.from_numpy(self.start_idxs)

    def __iter__(self):
        self.n = 0
        return self
    
    def __len__(self):
        return self.num_batches

    def __next__(self):
        """
        return three elements: final_ranges, input_nodes, subg
        final_ranges means the ranges of all sampled nodes, in Inductive also seeds,
        it means input_nodes, which we use to induce the subg
        """
        if self.n < self.num_batches:

            # first get the node_ranges of the composed subgraph
            st_list = self.start_idxs[self.n*self.bulk_decomp:(self.n+1)*self.bulk_decomp]
            range_list = []
            for st in st_list:
                end = st+self.node_budget//self.bulk_decomp
                if end > self.total_nodes:
                    range_list.append((st, self.total_nodes))
                    range_list.append((0, end-self.total_nodes))
                else:
                    range_list.append((st, end))
            final_ranges = combine_range(range_list)

            # then prepare the node ids and get subgraph
            input_nodes = torch.cat([torch.arange(st,end) for st,end in final_ranges])
            subg = self.g.subgraph(input_nodes)

            self.n += 1
        else:
            rp = torch.randperm(len(self.start_idxs))
            self.start_idxs = self.start_idxs[rp]
            raise StopIteration()

        return final_ranges, None, subg


class MOSNeighborSamplerTransductive(dgl.dataloading.Sampler):
    """
    given batch_size=B and model_depth=K 
    for the first K-1 layer, induce the subgraph and perform message
    passing on it; for the last layer, identify all the seeds and
    construct a block using FullNeighborSampler 
    """
    def __init__(self, node_budget, num_layers, g, train_mask, bulk_decomp=1):
        super().__init__()
        # direct arguments
        self.node_budget = node_budget
        self.num_layers = num_layers
        self.g = g
        self.train_mask = train_mask
        self.bulk_decomp = bulk_decomp

        # infered arguments
        assert self.train_mask is not None
        self.total_nodes = self.g.num_nodes()
        self.num_batches = self.total_nodes // self.node_budget + 1

        # utilities
        self.N = 10 * self.num_batches * self.bulk_decomp # num of small blocks
        self.start_idxs = np.random.choice(range(self.g.num_nodes()),size=(self.N,))
        self.start_idxs = torch.from_numpy(self.start_idxs)
        self.full_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)

    def __iter__(self):
        self.n = 0
        return self
    
    def __len__(self):
        return self.num_batches

    def __next__(self):
        """
        return three elements: final_ranges, seeds, subgs
        final_ranges include all sampled nodes, transform with 
        seeds are all sampled training nodes
        subgs = [induced_subgraph, block], where block's dst nodes are seeds
        """
        if self.n < self.num_batches:
            # first get the node_ranges of the composed subgraph
            st_list = self.start_idxs[self.n*self.bulk_decomp:(self.n+1)*self.bulk_decomp]
            range_list = []
            for st in st_list:
                end = st+self.node_budget//self.bulk_decomp
                if end > self.total_nodes:
                    range_list.append((st, self.total_nodes))
                    range_list.append((0, end-self.total_nodes))
                else:
                    range_list.append((st, end))
            final_ranges = combine_range(range_list)

            # then prepare the node ids and get subgraph
            input_nodes = torch.cat([torch.arange(st,end) for st,end in final_ranges])
            subg = self.g.subgraph(input_nodes)

            # next get all seeds using train_mask
            cur_train_mask = torch.cat([self.train_mask[st:end] for st,end in final_ranges])
            cur_train_nids_in_sg = cur_train_mask.nonzero().reshape(-1)
            seeds = input_nodes[cur_train_mask]

            # finally prepare a block with dst=seeds
            block = self.full_sampler.sample_blocks(subg, cur_train_nids_in_sg)[-1][0]

            self.n += 1
        else:
            rp = torch.randperm(len(self.start_idxs))
            self.start_idxs = self.start_idxs[rp]
            raise StopIteration()

        return final_ranges, seeds, [subg, block]

def combine_range(lis):
    """
    given several ranges in the format of [(st1, end1), (st2, end2)...]
    return combined ranges
    """
    lis.sort(key=lambda x:x[0])
    res = [lis[0]]
    for st, end in lis[1:]:
        if st < res[-1][1]:
            # cur_start < previous end, overlap
            if end <= res[-1][1]:
                # cur_range covered by prev_range
                continue
            else:
                prev = list(res.pop())
                prev[1] = end
                res.append(tuple(prev))
        elif st == res[-1][1]:
            # concat
            prev = list(res.pop())
            prev[1] = end
            res.append(tuple(prev))
        else:
            res.append((st, end))
    return res
