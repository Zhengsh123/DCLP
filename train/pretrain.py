import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import argparse
import math
import os
import sys
import time
import torch
import random
import shutil
import warnings
import torch.optim
import torch.utils.data
from tqdm import tqdm
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from torch_geometric.data import Batch
from encoder.gin import EncoderGIN
from util.util_dataset import NASBench101Dataset,load_data_from_pkl,generate_101_base_info,BatchCollator,\
    generate_201_base_info,generate_darts_base_info
from train.builder import MoCo


parser = argparse.ArgumentParser(description='PyTorch NAS predictor Training')
parser.add_argument('--search_space',default='101',help='name of search space,101,201,darts')
parser.add_argument('--train_path',default='./pkl/nasbench101_all_data.pkl', metavar='DIR',help='path to dataset')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2048, type=int,
                    metavar='N',
                    help='mini-batch size (default: 2048), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.015, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=131072, type=int,
                    help='queue size; number of negative keys (default: 131072)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


def main(base_encoder=EncoderGIN):
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args.gpu,args,base_encoder=base_encoder)


def main_worker(gpu,args,base_encoder=EncoderGIN):
    if args.search_space=='101':
        search_space=generate_101_base_info()
        base_filename='./checkpoint/checkpoint_101'
    elif args.search_space=='201':
        search_space = generate_201_base_info()
        base_filename = './checkpoint/checkpoint_201'
    elif args.search_space=='darts':
        search_space = generate_darts_base_info()
        base_filename = './checkpoint/checkpoint_darts'
    else:
        search_space = generate_101_base_info()
        base_filename = './checkpoint/checkpoint_101'
    args.gpu = gpu
    input_dim=search_space.get_ops_num()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    model=MoCo(base_encoder,input_dim,args.moco_dim,args.moco_k,args.moco_m,args.moco_t,args.mlp)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    criterion=nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer=torch.optim.SGD(model.parameters(),args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True
    data=load_data_from_pkl(args.train_path)
    nasbench=None
    train_dataset = NASBench101Dataset(data, search_space, nasbench, aug='edit_node',optional=1,dataset=args.search_space)
    collator = BatchCollator()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               drop_last=True,collate_fn=collator)
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        if epoch == 32:
            train_dataset = NASBench101Dataset(data, search_space, nasbench, aug='empty',dataset=args.search_space)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                       drop_last=True, collate_fn=collator)
        if epoch == 42:
            train_dataset = NASBench101Dataset(data, search_space, nasbench, optional=1,aug='edit_node',dataset=args.search_space)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                       drop_last=True, collate_fn=collator)
        adjust_learning_rate(optimizer, epoch, args)
        train(train_loader,model,criterion,optimizer,epoch,args)
        if epoch%10==1:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename=base_filename+'_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        aug_1 = list(data[0])
        aug_2 = list(data[1])
        batch_1 = Batch.from_data_list(aug_1)
        batch_2 = Batch.from_data_list(aug_2)
        if args.gpu is not None:
            batch_1 = batch_1.cuda(args.gpu, non_blocking=True)
            batch_2 = batch_2.cuda(args.gpu, non_blocking=True)
        output, target = model(im_q=batch_1, im_k=batch_2)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), batch_1.size(0))
        top1.update(acc1[0], batch_1.size(0))
        top5.update(acc5[0], batch_1.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if args.cos:
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__=="__main__":
    main()
