import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import argparse
import warnings
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from scipy.stats import kendalltau
from encoder.gin import EncoderGIN
from util.util_data import get_data_index_from_101,get_corresponding_metrics_by_index,\
    get_data_index_from_201,get_corresponding_metrics_by_index_201,get_finetune_data_darts,data_scale
from util.util_dataset import NASBench101Dataset,load_data_from_pkl,generate_101_base_info,BatchCollator,\
    generate_201_base_info,generate_darts_base_info,ReNASBench101Dataset,ReNASBench201Dataset,ReDartsDataset
from torch_geometric.data import DataLoader


parser = argparse.ArgumentParser(description='PyTorch NAS predictor fine-tuning')
parser.add_argument('--search_space',default='101',help='name of search space,101,201,darts')
parser.add_argument('--train_path',default='./pkl/nasbench101_all_data.pkl', metavar='DIR',help='path to dataset')
parser.add_argument('--epochs', default=200, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--pretrain_path',default='./checkpoint/checkpoint_101_0061.pth.tar',help='name of search space,101,201,darts')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--seed', default=2023, type=int,help='seed for initializing training.')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--dataset', default='cifar10', help='NASBench-201 can select from cifar10,cifar100,imagenet')

parser.add_argument('--train_num', default=30, type=int,help='labeled data used in training')
parser.add_argument('--batch_size', default=10, type=int,help='batch_size of fine-tuning')
parser.add_argument('--test_num', default=3000, type=int,help='labeled data used in test')
args = parser.parse_args()


def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    train(args.gpu, args)


if args.search_space=='101':
    search_space=generate_101_base_info()
elif args.search_space=='201':
    search_space = generate_201_base_info()
elif args.search_space=='darts':
    search_space = generate_darts_base_info()
else:
    search_space = generate_101_base_info()
model = EncoderGIN(input_dim=search_space.get_ops_num(),output_dim=1, re_train=True)
if args.pretrain_path is not None:
    checkpoint = torch.load(args.pretrain_path, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
            state_dict[k[len("encoder_q."):]] = state_dict[k]
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
model.cuda()
model.train()
criterion = torch.nn.MSELoss()
parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = torch.optim.Adam(parameters,lr=args.lr)


def train(gpu,args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.search_space=='101':
        search_space=generate_101_base_info()
        train_list = get_data_index_from_101(args.train_num, './pkl')
        test_list = get_data_index_from_101(args.test_num, './pkl', 'fixed_test', args.train_num)
        train_data = get_corresponding_metrics_by_index(train_list, './pkl',args.train_path)
        test_data = get_corresponding_metrics_by_index(test_list, './pkl',args.train_path, 'fixed_test')
        train_data = data_scale(train_data)
        train_dataset = ReNASBench101Dataset(train_data, search_space)
        test_dataset = ReNASBench101Dataset(test_data, search_space)
        model_save_path='./res/predictor_101_{:04d}.pt'.format(args.train_num)
    elif args.search_space=='201':
        search_space = generate_201_base_info()
        train_list = get_data_index_from_201(args.train_num, './pkl')
        test_list = get_data_index_from_201(args.test_num, './pkl', 'fixed_test', args.train_num)
        train_data = get_corresponding_metrics_by_index_201(train_list, './pkl', args.train_path)
        test_data = get_corresponding_metrics_by_index_201(test_list, './pkl', args.train_path, 'fixed_test')
        train_data = data_scale(train_data)
        train_dataset = ReNASBench201Dataset(train_data, search_space,dataset=args.dataset)
        test_dataset = ReNASBench201Dataset(test_data, search_space,dataset=args.dataset)
        model_save_path = './res/predictor_201_{:04d}.pt'.format(args.train_num)
    elif args.search_space=='darts':
        search_space = generate_darts_base_info()
        train_data=get_finetune_data_darts('./pkl',args.train_path,args.train_num)
        train_data = data_scale(train_data)
        train_dataset = ReDartsDataset(train_data, search_space)
        model_save_path = './res/predictor_darts_{:04d}.pt'.format(args.train_num)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader):
            target = data.y.float().view(args.batch_size, -1).cuda()
            data.cuda()
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(str(epoch)+'_loss:',loss.item())

    torch.save(model.state_dict(), model_save_path)
    if args.search_space=='101' or args.search_space=='201':
        test_loader = DataLoader(test_dataset, batch_size=args.test_num, shuffle=False)
        Ktau = test(test_loader)
        print(Ktau)


def test(test_loader):
    model.eval()
    for i, data in enumerate(test_loader):
        data.cuda()
        target = data.y.cpu().float()
        output = model(data).reshape(args.test_num)
        result_arg = np.argsort(output.cpu().detach().numpy())
        y_test_arg = np.argsort(np.array(target))
        result_rank = np.zeros(len(y_test_arg))
        y_test_rank = np.zeros(len(y_test_arg))
        for i in range(len(y_test_arg)):
            result_rank[result_arg[i]] = i
            y_test_rank[y_test_arg[i]] = i
        KTau,_ = kendalltau(result_rank, y_test_rank)
        model.train()
        return KTau


def list_mle(y_pred, y_true, k=None):
    if k is not None:
        sublist_indices = (y_pred.shape[1] * torch.rand(size=k)).long()
        y_pred = y_pred[:, sublist_indices]
        y_true = y_true[:, sublist_indices]
    _, indices = y_true.sort(descending=True, dim=-1)
    pred_sorted_by_true = y_pred.gather(dim=1, index=indices)
    cumsums = pred_sorted_by_true.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
    listmle_loss = torch.log(cumsums + 1e-10) - pred_sorted_by_true
    return listmle_loss.sum(dim=1).mean()


if __name__=="__main__":
    main()