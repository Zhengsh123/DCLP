import os
import copy
import torch
import random
import pickle
import numpy as np
from nasbench import api
import scipy.sparse as sp
import torch.nn.functional as F
from sklearn import preprocessing
from collections import namedtuple
from nas_201_api import NASBench201API as API201


def generate_101_to_pkl(nasbench101_path, write_path, data_ratio=1.):
    if not os.path.exists(write_path):
        nasbench = api.NASBench(nasbench101_path)
        all_data = []
        num = int(423624 * data_ratio)
        for unique_hash in nasbench.hash_iterator():
            fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
            final_test_accuracy_list = []
            for i in range(3):
                final_test_accuracy_list.append(computed_metrics[108][i]['final_test_accuracy'])
            final_test_accuracy = np.mean(final_test_accuracy_list)
            all_data.append((fixed_metrics, final_test_accuracy))
        with open(write_path, 'wb') as file:
            pickle.dump(all_data[:num], file)


def generate_201_to_pkl(nasbench201_path, write_path, data_ratio=1.):
    all_data=[]
    if not os.path.exists(write_path):
        nasbench201=API201(nasbench201_path)
        range_num=int(data_ratio*len(nasbench201.evaluated_indexes))
        for index in range(range_num):
            info = nasbench201.query_meta_info_by_index(index, '12')
            arch_str = info.arch_str
            info = nasbench201.query_meta_info_by_index(index, '200')
            cifar10_acc = info.get_metrics('cifar10', 'ori-test')['accuracy']
            cifar100_acc=info.get_metrics('cifar100', 'ori-test')['accuracy']
            ImageNet_acc=info.get_metrics('ImageNet16-120', 'ori-test')['accuracy']
            ops_list=save_arch_str2op_list(arch_str)
            pruned_matrix, pruned_op = delete_useless_node(ops_list)
            if pruned_matrix is None:
                continue
            fix_matrix={'module_adjacency':pruned_matrix,'module_operations':pruned_op}
            all_data.append((fix_matrix,cifar10_acc,cifar100_acc,ImageNet_acc))
        with open(write_path, 'wb')as file:
            pickle.dump(all_data, file)
    return all_data


def generate_darts_to_pkl(write_path,data_num):
    available_ops = ['input1', 'input2', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5',
                     'dil_conv_3x3', 'dil_conv_5x5', 'output']
    archs = []
    if not os.path.exists(write_path):
        for _ in range(data_num):
            genotype = sample_darts_arch(available_ops[2:-1])
            arch = darts_to_nasbench101(genotype)
            archs.append(arch)
        with open(write_path, 'wb')as file:
            pickle.dump(archs, file)
    return archs


def sample_darts_arch(available_ops):
    geno = []
    for _ in range(2):
        cell = []
        for i in range(4):
            ops_normal = np.random.choice(available_ops, 2)
            nodes_in_normal = sorted(np.random.choice(range(i+2), 2, replace=False))
            cell.extend([(ops_normal[0], nodes_in_normal[0]), (ops_normal[1], nodes_in_normal[1])])
        geno.append(cell)
    genotype = Genotype(normal=geno[0], normal_concat=[2, 3, 4, 5], reduce=geno[1], reduce_concat=[2, 3, 4, 5])
    return genotype


def darts_to_nasbench101(genotype):
    arch = []
    for arch_list, concat in [(genotype.normal, genotype.normal_concat), (genotype.reduce, genotype.reduce_concat)]:
        num_ops = len(arch_list) + 3
        adj = np.zeros((num_ops, num_ops), dtype=np.uint8)
        ops = ['input1', 'input2', 'output']
        node_lists = [[0], [1] , [2, 3], [4, 5], [6, 7], [8, 9], [10]]
        for node in arch_list:
            node_idx = len(ops) - 1
            adj[node_lists[node[1]], node_idx] = 1
            ops.insert(-1, node[0])
        adj[[x for c in concat for x in node_lists[c]], -1] = 1
        cell = {'module_adjacency': adj,
                'module_operations': ops,}
        arch.append(cell)
    adj = np.zeros((num_ops*2, num_ops*2), dtype=np.uint8)
    adj[:num_ops, :num_ops] = arch[0]['module_adjacency']
    adj[num_ops:, num_ops:] = arch[1]['module_adjacency']
    ops = arch[0]['module_operations'] + arch[1]['module_operations']
    arch = {'module_adjacency': adj,
            'module_operations': ops,}
    return arch


def save_arch_str2op_list(save_arch_str):
    op_list = []
    save_arch_str_list = API201.str2lists(save_arch_str)
    op_list.append(save_arch_str_list[0][0][0])
    op_list.append(save_arch_str_list[1][0][0])
    op_list.append(save_arch_str_list[1][1][0])
    op_list.append(save_arch_str_list[2][0][0])
    op_list.append(save_arch_str_list[2][1][0])
    op_list.append(save_arch_str_list[2][2][0])
    return op_list


def delete_useless_node(ops):
    matrix = copy.deepcopy(BASIC_MATRIX)
    for i, op in enumerate(ops, start=1):
        m = []
        n = []
        if op == 'skip_connect':
            for m_index in range(8):
                ele = matrix[m_index][i]
                if ele == 1:
                    matrix[m_index][i] = 0
                    m.append(m_index)
            for n_index in range(8):
                ele = matrix[i][n_index]
                if ele == 1:
                    matrix[i][n_index] = 0
                    n.append(n_index)
            for m_index in m:
                for n_index in n:
                    matrix[m_index][n_index] = 1
        elif op == 'none':
            for m_index in range(8):
                matrix[m_index][i] = 0
            for n_index in range(8):
                matrix[i][n_index] = 0
    ops_copy = copy.deepcopy(ops)
    ops_copy.insert(0, 'input')
    ops_copy.append('output')
    model_spec = api.ModelSpec(matrix=matrix, ops=ops_copy)
    return model_spec.matrix, model_spec.ops


def get_data_index_from_101(num,pkl_path,type='train',train_num=500):
    if not os.path.isdir('../pkl'):
        os.makedirs('../pkl')
    if type == 'fixed_test':
        save_path = os.path.join(pkl_path, 'fixed_test_data{:0>6d}_{:0>6d}.pkl'.format(train_num, num))
    else:
        save_path = os.path.join(pkl_path, 'train_data{:0>6d}.pkl'.format(num))
    if os.path.isfile(save_path):
        with open(save_path, 'rb') as file:
            random_list = pickle.load(file)
        print('Exist {:s}_data.pkl, loading...'.format(type))
    else:
        max_number=423624
        list_remove_train = list(range(0, max_number))
        if type == 'fixed_test':
            train_data_index_path = os.path.join(pkl_path, 'train_data{:0>6d}.pkl'.format(train_num))
            print('Removing train list (len: {:}) to sample...'.format(train_num))
            with open(train_data_index_path, 'rb') as file:
                train_list = pickle.load(file)
            for i in range(train_num):
                list_remove_train.remove(train_list[i])
        print('left: {:}'.format(len(list_remove_train)))
        random_list = random.sample(list_remove_train, num)
        random_list.sort()
        with open(save_path, 'wb') as file:
            pickle.dump(random_list, file)
        print('Run for the first time! Create new {:s}_data.pkl'.format(type))
    return random_list


def get_corresponding_metrics_by_index(index_list,pkl_path,data_path, type='train'):
    save_path = os.path.join(pkl_path, '{:s}_metrics{:0>6d}_{:0>6d}.pkl'.format(type, len(index_list), index_list[0]))
    print('\nGetting the corresponding metrics by index.')
    if not os.path.isfile(save_path):
        f=open(data_path,'rb')
        data = pickle.load(f)
        all_data = []
        for iter_num in range(len(data)):
            if iter_num in index_list:
                fixed_metrics,final_test_accuracy=data[iter_num]
                all_data.append((fixed_metrics, final_test_accuracy))
        if type in ['train', 'fixed_test']:
            with open(save_path, 'wb') as file:
                pickle.dump(all_data, file)
    else:
        with open(save_path, 'rb') as file:
            all_data = pickle.load(file)
        print('Loading: {:}'.format(save_path))
    return all_data


def get_data_index_from_201(num,pkl_path,type='train',train_num=500):
    if not os.path.isdir('../pkl/nasbench_201'):
        os.makedirs('../pkl/nasbench_201')
    if type == 'fixed_test':
        save_path = os.path.join(pkl_path, 'fixed_test_data{:0>6d}_{:0>6d}_201.pkl'.format(train_num, num))
    else:
        save_path = os.path.join(pkl_path, 'train_data{:0>6d}_201.pkl'.format(num))
    if os.path.isfile(save_path):
        with open(save_path, 'rb') as file:
            random_list = pickle.load(file)
        print('Exist {:s}_data.pkl, loading...'.format(type))
    else:
        max_number=15284
        list_remove_train = list(range(0, max_number))
        if type == 'fixed_test':
            train_data_index_path = os.path.join(pkl_path, 'train_data{:0>6d}_201.pkl'.format(train_num))
            print('Removing train list (len: {:}) to sample...'.format(train_num))
            with open(train_data_index_path, 'rb') as file:
                train_list = pickle.load(file)
            for i in range(train_num):
                list_remove_train.remove(train_list[i])
        print('left: {:}'.format(len(list_remove_train)))
        random_list = random.sample(list_remove_train, num)
        random_list.sort()
        with open(save_path, 'wb') as file:
            pickle.dump(random_list, file)
        print('Run for the first time! Create new {:s}_data_201.pkl'.format(type))
    return random_list


def get_corresponding_metrics_by_index_201(index_list,pkl_path,data_path, type='train'):
    save_path = os.path.join(pkl_path, '{:s}_metrics{:0>6d}_{:0>6d}_201.pkl'.format(type, len(index_list), index_list[0]))
    print('\nGetting the corresponding metrics by index.')
    if not os.path.isfile(save_path):
        f = open(data_path, 'rb')
        data = pickle.load(f)
        all_data = []
        for iter_num in range(len(data)):
            if iter_num in index_list:
                fixed_metrics, cifar10_acc,cifar100_acc,imagenet_acc = data[iter_num]
                all_data.append((fixed_metrics, cifar10_acc,cifar100_acc,imagenet_acc))
        if type in ['train', 'fixed_test']:
            with open(save_path, 'wb') as file:
                pickle.dump(all_data, file)
    else:
        with open(save_path, 'rb') as file:
            all_data = pickle.load(file)
        print('Loading: {:}'.format(save_path))
    return all_data


def get_finetune_data_darts(pkl_path,data_path,train_num):
    save_path = os.path.join(pkl_path,'train_metrics{:0>6d}_darts.pkl'.format(train_num))
    if not os.path.isfile(save_path):
        f = open(data_path, 'rb')
        data = pickle.load(f)
        all_data = []
        for iter_num in range(train_num):
                fixed_metrics, acc = data[iter_num]
                all_data.append((fixed_metrics, acc))
        with open(save_path, 'wb') as file:
            pickle.dump(all_data, file)
    else:
        with open(save_path, 'rb') as file:
            all_data = pickle.load(file)
        print('Loading: {:}'.format(save_path))
    return all_data


class SearchSpace():
    def __init__(self, num_vertices, ops_dict, max_len, space_name='nasbench101'):
        self.num_vertices = num_vertices
        self.ops_dict = ops_dict
        self.max_len = max_len
        self.space_name = space_name

    def get_vertices_num(self):
        return self.num_vertices

    def get_ops_dict(self):
        return self.ops_dict

    def get_max_len(self):
        return self.max_len

    def get_ops_num(self):
        return len(self.ops_dict.keys())


def generate_101_base_info():
    vertice_num = 7
    ops_dict = {
        'input': 0,
        'conv3x3-bn-relu': 1,
        'conv1x1-bn-relu': 2,
        'maxpool3x3': 3,
        'output': 4,
        'null': 5
    }
    max_len = 423624
    search_space = SearchSpace(vertice_num, ops_dict, max_len)
    return search_space


def generate_201_base_info():
    vertice_num = 8
    ops_dict = {
        'input': 0,
        'nor_conv_1x1': 1,
        'nor_conv_3x3': 2,
        'avg_pool_3x3': 3,
        'output': 4,
        'null': 5
    }
    max_len = 15625
    search_space = SearchSpace(vertice_num, ops_dict, max_len)
    return search_space


def generate_darts_base_info():
    vertice_num = 12
    ops_dict = {
        'input1': 0,
        'input2': 1,
        'max_pool_3x3': 2,
        'avg_pool_3x3': 3,
        'skip_connect': 4,
        'sep_conv_3x3': 5,
        'sep_conv_5x5':6,
        'dil_conv_3x3':7,
        'dil_conv_5x5':8,
        'output':9
    }
    max_len = 100000
    search_space = SearchSpace(vertice_num, ops_dict, max_len)
    return search_space


def load_data_from_pkl(path):
    f = open(path, 'rb')
    data = pickle.load(f)
    return data


def nasbench101_init(nasbench_path):
    return api.NASBench(nasbench_path)


def nasbench2graph_101(data, ops_dict, is_idx=False, reverse=False):
    matrix, ops = data[0], data[1]
    vertice_num=len(ops)
    if reverse:
        matrix = matrix.T
    if is_idx:
        node_feature=torch.tensor([int(ops[i].item()) for i in range(vertice_num)])
    else:
        node_feature = torch.tensor([ops_dict[ops[i]] for i in range(vertice_num)])
    node_feature=F.one_hot(node_feature,len(ops_dict.keys())).float()
    tmp_coo = sp.coo_matrix(matrix)
    indices = np.vstack((tmp_coo.row, tmp_coo.col))
    edge_idx = torch.tensor(indices,dtype=torch.long)
    return edge_idx, node_feature


class BatchCollator(object):
    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        aug_list_1 = transposed_batch[0]
        aug_list_2 = transposed_batch[1]
        return aug_list_1, aug_list_2


def data_norm(data):
    acc_list=[]
    for temp_data in data:
        acc_list.append(temp_data[1])
    acc_arr=np.array(acc_list).reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    target = min_max_scaler.fit_transform(acc_arr).reshape(-1)
    temp=[]
    for index in range(len(data)):
        temp.append((data[index][0],target[index]))
    data=temp
    return data


def data_scale(data):
    acc_num=len(data[0])-1
    data_num=len(data)
    acc_arr=np.zeros((data_num,acc_num))
    for i in range(data_num):
        for j in range(acc_num):
            acc_arr[i][j]=data[i][j+1]
    target = preprocessing.scale(acc_arr,axis=0)
    temp = []
    for index in range(data_num):
        temp_temp=[data[index][0]]
        for j in range(acc_num):
            temp_temp.append(target[index][j])
        temp.append(temp_temp)
    data = temp
    return data


BASIC_MATRIX = [[0, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0]]

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


