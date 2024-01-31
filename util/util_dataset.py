import copy
import torch
import numpy as np
from util.util_aug import DagAug
from torch.utils.data import Dataset
from torch_geometric.data import Data
from util.util_data import nasbench2graph_101,BatchCollator,generate_101_base_info,generate_201_base_info,generate_darts_base_info,\
    load_data_from_pkl


def padding_zero_in_matrix(module_adjacency, module_operations, search_space):
    adj=copy.deepcopy(module_adjacency)
    ops=copy.deepcopy(module_operations)
    len_operations = len(ops)
    assert len_operations == len(adj)
    vertices_num = search_space.get_vertices_num()
    for i in range(len_operations, vertices_num):
        ops.insert(i - 1, 'null')
    padding_matrix = np.insert(adj, len_operations - 1,np.zeros([vertices_num - len_operations, len_operations]), axis=0)
    adj = np.insert(padding_matrix, [len_operations - 1], np.zeros([vertices_num, vertices_num - len_operations]), axis=1)
    return adj, ops


class NASBench101Dataset(Dataset):
    def __init__(self,data,search_space,nasbench,aug,optional=1,is_padding=False,mode='use',dataset='101'):
        super(NASBench101Dataset, self).__init__()
        self.data=data
        self.len=len(self.data)
        self.search_space=search_space
        self.nasbench=nasbench
        self.aug_name=aug
        self.is_padding=is_padding
        self.mode=mode
        self.dataset=dataset
        self.dag_aug=DagAug(self.aug_name,self.search_space,self.nasbench,optional=optional,mode='edit')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.dataset=='101':
            base_data,_=self.data[index]
        elif self.dataset=='201':
            base_data, _ ,_,_= self.data[index]
        elif self.dataset=='darts':
            base_data=self.data[index]
        base_adjacency, base_operations=base_data['module_adjacency'],base_data['module_operations']
        adjacency_aug1,operations_aug1=self.dag_aug.aug(base_adjacency,base_operations)
        adjacency_aug2, operations_aug2 = self.dag_aug.aug(base_adjacency, base_operations)
        if self.is_padding:
            adjacency_aug1, operations_aug1 = padding_zero_in_matrix(adjacency_aug1, operations_aug1, self.search_space)
            adjacency_aug2, operations_aug2 = padding_zero_in_matrix(adjacency_aug2, operations_aug2, self.search_space)
        edge_idx_1, node_feature_1=nasbench2graph_101([adjacency_aug1,operations_aug1],self.search_space.get_ops_dict())
        edge_idx_2, node_feature_2 = nasbench2graph_101([adjacency_aug2, operations_aug2],self.search_space.get_ops_dict())
        data_aug1=Data(x=node_feature_1,edge_index=edge_idx_1,dtype=torch.long)
        data_aug2 = Data(x=node_feature_2, edge_index=edge_idx_2,dtype=torch.long)
        return data_aug1, data_aug2


class ReNASBench101Dataset(Dataset):
    def __init__(self, data, search_space):
        super(ReNASBench101Dataset,self).__init__()
        self.data=data
        self.len = len(self.data)
        self.search_space=search_space

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        base_data, acc = self.data[index]
        base_adjacency, base_operations = base_data['module_adjacency'], base_data['module_operations']
        # base_adjacency, base_operations = padding_zero_in_matrix(base_adjacency, base_operations, self.search_space)
        edge_idx, node_feature = nasbench2graph_101([base_adjacency, base_operations],
                                                        self.search_space.get_ops_dict())
        graph=Data(x=node_feature, edge_index=edge_idx,y=acc,dtype=torch.long)
        return graph


class ReNASBench201Dataset(Dataset):
    def __init__(self, data, search_space,dataset='cifar10'):
        super(ReNASBench201Dataset,self).__init__()
        self.data=data
        self.len = len(self.data)
        self.search_space=search_space
        self.dataset=dataset

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.dataset=='cifar10':
            base_data, acc,_,_= self.data[index]
        elif self.dataset=='cifar100':
            base_data,_, acc,_= self.data[index]
        elif self.dataset=='imagenet':
            base_data,_,_,acc= self.data[index]
        else:
            base_data, acc,_,_ = self.data[index]
        base_adjacency, base_operations = base_data['module_adjacency'], base_data['module_operations']
        edge_idx, node_feature = nasbench2graph_101([base_adjacency, base_operations],
                                                        self.search_space.get_ops_dict())
        graph=Data(x=node_feature, edge_index=edge_idx,y=acc,dtype=torch.long)
        return graph


class ReDartsDataset(Dataset):
    def __init__(self, data, search_space):
        super(ReDartsDataset,self).__init__()
        self.data=data
        self.len = len(self.data)
        self.search_space=search_space

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        base_data, acc = self.data[index]
        base_adjacency, base_operations = base_data['module_adjacency'], base_data['module_operations']
        edge_idx, node_feature = nasbench2graph_101([base_adjacency, base_operations],
                                                        self.search_space.get_ops_dict())
        graph=Data(x=node_feature, edge_index=edge_idx,y=acc,dtype=torch.long)
        return graph

