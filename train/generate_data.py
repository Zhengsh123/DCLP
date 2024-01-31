import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from util.util_data import generate_101_to_pkl,generate_201_to_pkl,generate_darts_to_pkl,\
    generate_darts_base_info,sample_darts_arch,darts_to_nasbench101
from darts.cnn.train_search import Train
import argparse
import json
from collections import namedtuple
from util.util_data import darts_to_nasbench101
import os
import pickle
import numpy as np

parser = argparse.ArgumentParser(description='Generate pretrain and fine-tuning data')
parser.add_argument('--search_space',default='101',help='name of search space,101,201,darts')
parser.add_argument('--data_num',default=100000,type=int,help='only darts need')
args = parser.parse_args()


def generate_darts_train_data_from_log():
    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    filePath = './dataset/darts/'
    names = os.listdir(filePath)
    all_data = []
    for name in names:
        path = filePath + name
        with open(path, 'r', encoding='utf8') as fp:
            json_data = json.load(fp)
            acc = json_data['test_accuracy']
            normal = []
            reduce = []
            for item in json_data['optimized_hyperparamater_config'].keys():
                if "edge_normal" in item:
                    temp_index = int(item.split(':')[2].split('_')[2])
                    ops = json_data['optimized_hyperparamater_config'][item]
                    normal.append((temp_index, ops))
                if "edge_reduce" in item:
                    temp_index = int(item.split(':')[2].split('_')[2])
                    ops = json_data['optimized_hyperparamater_config'][item]
                    reduce.append((temp_index, ops))
            gene_normal = []
            gene_reduce = []
            normal_concat = [2, 3, 4, 5]
            reduce_concat = [2, 3, 4, 5]
            n = [0, 2, 5, 9]
            for i in range(4):
                gene_normal.append((normal[2 * i][1], normal[2 * i][0] - n[i]))
                gene_reduce.append((reduce[2 * i][1], reduce[2 * i][0] - n[i]))
                gene_normal.append((normal[2 * i + 1][1], normal[2 * i + 1][0] - n[i]))
                gene_reduce.append((reduce[2 * i + 1][1], reduce[2 * i + 1][0] - n[i]))
            gene = Genotype(normal=gene_normal, normal_concat=normal_concat, reduce=gene_reduce,
                            reduce_concat=reduce_concat)
            res = darts_to_nasbench101(gene)
            all_data.append((res, acc))
            fp.close()
    with open('./pkl/darts_train_data.pkl', 'wb') as file:
        pickle.dump(all_data, file)


def generate_darts_train_data_from_scratch(data_num):
    available_ops = ['input1', 'input2', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5',
                     'dil_conv_3x3', 'dil_conv_5x5', 'output']
    archs = []
    darts_train=Train()
    for _ in range(data_num):
        genotype = sample_darts_arch(available_ops[2:-1])
        val_accs, test_accs = darts_train.main(42, genotype)
        val_acc = np.mean(list(zip(*val_accs))[1])
        test_acc = np.mean(list(zip(*test_accs))[1])
        arch = darts_to_nasbench101(genotype)
        archs.append((arch,test_acc))
    with open('./darts_train_data.pkl', 'wb') as file:
        pickle.dump(archs, file)


if __name__=="__main__":
    if args.search_space=='101':
        generate_101_to_pkl('./dataset/nasbench_only108.tfrecord', './pkl/nasbench101_all_data.pkl', 1)
    elif  args.search_space=='201':
        generate_201_to_pkl('./dataset/NAS-Bench-201-v1_1-096897.pth', './pkl/nasbench201_all_data.pkl', 1)
    elif args.search_space == 'darts':
        generate_darts_to_pkl('./pkl/darts_data.pkl', args.data_num)
        generate_darts_train_data_from_log()
        #If you want to start training from scratch to get the dataset,
        # you can use the following function
        # generate_darts_train_data_from_scratch(500)
    print('generate data is done')

