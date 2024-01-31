import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import torch
import pickle
import argparse
import numpy as np
from encoder.gin import EncoderGIN
from darts.cnn.train_search import Train
from util.util_data import nasbench2graph_101
from torch_geometric.data import Data
from torch_geometric.data import Batch
from util.util_data import generate_101_base_info,generate_201_base_info,generate_darts_base_info,\
    sample_darts_arch,darts_to_nasbench101
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


parser = argparse.ArgumentParser(description='DCLP search')
parser.add_argument('--search_space',default='101',help='name of search space,101,201,darts')
parser.add_argument('--save_path', default='./nasbench_101.txt',  help='save path')
parser.add_argument('--train_path',default='./pkl/nasbench101_all_data.pkl', metavar='DIR',help='path to dataset')
parser.add_argument('--predictor_path',default='./res/predictor_101_0030.pt',help='name of search space,101,201,darts')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

parser.add_argument('--k', default=1, type=int,help='architecture selected every iteration')
parser.add_argument('--T', default=50, type=int,help='iteration num')
parser.add_argument('--data_size', default=500, type=int,help='architectures given every iteration')
args = parser.parse_args()


def random_generate_data(data_num):
    if args.search_space=='101' or args.search_space=='201':
        all_data_path=args.train_path
        f = open(all_data_path, 'rb')
        all_data = pickle.load(f)
        if args.search_space=='101':
            search_space = generate_101_base_info()
        else:
            search_space = generate_201_base_info()
        index_list = np.random.randint(len(all_data) - 1, size=data_num)
        arch_info=[]
        data=[]
        all_acc=[]
        for index in index_list:
            temp_data=all_data[index]
            adj=temp_data[0]['module_adjacency']
            ops=temp_data[0]['module_operations']
            arch_info.append((adj,ops))
            acc=temp_data[1]
            all_acc.append(acc)
            edge_idx, node_feature = nasbench2graph_101([adj, ops],search_space.get_ops_dict())
            data.append(Data(x=node_feature, edge_index=edge_idx, dtype=torch.long))
        return data,all_acc,arch_info
    elif args.search_space=='darts':
        available_ops = ['input1', 'input2', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3',
                         'sep_conv_5x5',
                         'dil_conv_3x3', 'dil_conv_5x5', 'output']
        search_space=generate_darts_base_info()
        arch_info= []
        data=[]
        for _ in range(data_num):
            genotype = sample_darts_arch(available_ops[2:-1])
            arch = darts_to_nasbench101(genotype)
            adj = arch['module_adjacency']
            ops = arch['module_operations']
            edge_idx, node_feature = nasbench2graph_101([adj, ops], search_space.get_ops_dict())
            data.append(Data(x=node_feature, edge_index=edge_idx, dtype=torch.long))
            arch_info.append(genotype)
        return data,[],arch_info
    else:
        return [],[],[]


def search():
    if args.search_space == '101':
        search_space = generate_101_base_info()
    elif args.search_space == '201':
        search_space = generate_201_base_info()
    elif args.search_space == 'darts':
        search_space = generate_darts_base_info()
    else:
        search_space = generate_101_base_info()
    model=EncoderGIN(input_dim=search_space.get_ops_num(),output_dim=1,re_train=True)
    loaded_paras = torch.load(args.predictor_path)
    model.load_state_dict(loaded_paras)
    model.eval()
    best_arch=None
    max_acc=0
    if args.search_space == '101' or args.search_space=='201':
        for i in range(args.T):
            data,all_acc,arch_info=random_generate_data(args.data_size)
            batch=Batch.from_data_list(data)
            with torch.no_grad():
                predic_acc=model(batch).numpy().reshape(-1)
                max_index=np.argmax(predic_acc)
                if all_acc[max_index]>max_acc:
                    max_acc=all_acc[max_index]
                    best_arch=arch_info[max_index]

        with open(args.save_path,'w')as fw:
            fw.write(str(best_arch[0]))
            fw.write('\n')
            fw.write(str(best_arch[1]))
            fw.write('\n')
            fw.write(str(max_acc))
        fw.close()
    elif args.search_space == 'darts':
        darts_train=Train()
        pool=[]
        for i in range(args.T):
            data,_,arch_info=random_generate_data(args.data_size)
            batch = Batch.from_data_list(data)
            with torch.no_grad():
                predic_acc = model(batch).numpy().reshape(-1)
                max_index = np.argmax(predic_acc)
                pool.append(arch_info[max_index])
        with open(args.save_path, 'wb')as file:
            pickle.dump(pool, file)
        acc=[]
        for arch in pool:
            val_accs, test_accs = darts_train.main(42, arch)
            val_acc = np.mean(list(zip(*val_accs))[1])
            test_acc = np.mean(list(zip(*test_accs))[1])
            acc.append(test_acc)
        max_index=acc.index(max(acc))
        best_arch=pool[max_index]
        with open('./darts.txt','w')as fw:
            fw.write(str(best_arch))
            fw.write('\n')
            fw.write(str(acc[max_index]))
        fw.close()


if __name__=="__main__":
    search()