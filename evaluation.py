import os, sys, time, argparse
import math
import random
from easydict import EasyDict as edict
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from pathlib import Path
lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from datasets import get_datasets, get_nas_search_loaders
from procedures import prepare_seed, prepare_logger
from procedures import Linear_Region_Collector, get_ntk_n
from utils import get_model_infos
from log_utils import time_string
from models import get_cell_based_tiny_net, get_search_spaces  # , nas_super_nets
from nas_201_api import NASBench201API as API
from pdb import set_trace as bp
import torchsummary
from enum import Enum
import torch.optim as optim
from util import *
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler


def train(train_loader, model, criterion, optimizer, epoch, device, args, scaler):
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()

    # switch to train mode
    model.train()  

    print("Start training")
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with autocast():
            output = model(images)[1]
            loss = criterion(output, target)

        # output = model(images)
        # loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    return top1.avg, top5.avg


def test(valid_loader, network, criterion, args, device):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(valid_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    base_progress = 0
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(valid_loader):
            i = base_progress + i
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = network(images)[1]
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i + 1)
    return top1.avg, top5.avg

def main(args):

    assert args.timestamp != 'none', 'please give the timestamp to load the expreiment result'
        
    train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_path, -1)

    ##### config & logging #####
    config = edict()
    config.class_num = class_num
    config.xshape = xshape
    config.batch_size = args.batch_size
    config.NTK_batch_size = args.NTK_batch_size
    args.save_dir = args.save_dir + \
        "/repeat%d-prunNum%d-prec%d-%s-batch%d"%(
                args.repeat, args.prune_number, args.precision, args.init, config["NTK_batch_size"]) + \
        "/{:}/seed{:}".format(args.timestamp, args.rand_seed)
    config.save_dir = args.save_dir
    ###############

    if args.dataset != 'imagenet-1k':
        search_loader, train_loader, valid_loader = get_nas_search_loaders(train_data, valid_data, args.dataset, 'configs/', config.batch_size, args.workers)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    search_space = get_search_spaces('cell', args.search_space_name)
    if args.search_space_name == 'nas-bench-201':
        model_config = edict({'name': 'DARTS-V1',
                              'C': 3, 'N': 1, 'depth': -1, 'use_stem': True,
                              'max_nodes': args.max_nodes, 'num_classes': class_num,
                              'space': search_space,
                              'affine': True, 'track_running_stats': bool(args.track_running_stats),
                             })
        model_config_thin = edict({'name': 'DARTS-V1',
                                   'C': 1, 'N': 1, 'depth': 1, 'use_stem': False,
                                   'max_nodes': args.max_nodes, 'num_classes': class_num,
                                   'space': search_space,
                                   'affine': True, 'track_running_stats': bool(args.track_running_stats),
                                  })
    elif args.search_space_name == 'darts':
        model_config = edict({'name': 'DARTS-V1',
                              'C': 1, 'N': 1, 'depth': 2, 'use_stem': True, 'stem_multiplier': 1,
                              'num_classes': class_num,
                              'space': search_space,
                              'affine': True, 'track_running_stats': bool(args.track_running_stats),
                              'super_type': args.super_type,
                              'steps': 4,
                              'multiplier': 4,
                             })
        model_config_thin = edict({'name': 'DARTS-V1',
                                   'C': 1, 'N': 1, 'depth': 2, 'use_stem': False, 'stem_multiplier': 1,
                                   'max_nodes': args.max_nodes, 'num_classes': class_num,
                                   'space': search_space,
                                   'affine': True, 'track_running_stats': bool(args.track_running_stats),
                                   'super_type': args.super_type,
                                   'steps': 4,
                                   'multiplier': 4,
                                  })
    ## my test space                              
    elif args.search_space_name == 'Ye':
        model_config = edict({'name': 'DARTS-V1',
                              'C': 3, 'N': 1, 'depth': -1, 'use_stem': True,
                              'max_nodes': args.max_nodes, 'num_classes': class_num,
                              'space': search_space,
                              'affine': True, 'track_running_stats': bool(args.track_running_stats),
                             })
        model_config_thin = edict({'name': 'DARTS-V1',
                                   'C': 1, 'N': 1, 'depth': 1, 'use_stem': False,
                                   'max_nodes': args.max_nodes, 'num_classes': class_num,
                                   'space': search_space,
                                   'affine': True, 'track_running_stats': bool(args.track_running_stats),
                                  })

    arch_parameters = torch.from_numpy(np.load(args.save_dir + '/arch_parameters_history.npy')[-1]).cuda()

    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    
    # create logger for ploting
    file_complete = creating_path("Evaluation_Result", args.timestamp, "logs",
                                  file_name = str(args.batch_size)+ '_' + str(args.lr), extension='log')
    logger_complete = create_logger("complete", file_complete)

    save_path = creating_path("Evaluation_Result", args.timestamp, "checkpoint", 
                                  file_name = str(args.batch_size)+ '_' + str(args.lr))


    # Rebuilt the final searched network
    network = get_cell_based_tiny_net(model_config)
    network = network.to(device)
    network.set_alphas(arch_parameters)
    print(network)
    # torchsummary(network, (3, 32, 32))

    criterion = nn.CrossEntropyLoss().to(device)
          
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    epoch_num = args.epochs

    best_acc1 = 0

    scaler = GradScaler()

    # training loop
    for epoch in range(epoch_num):

        # train for one epoch
        train_acc1, train_acc5 = train(train_loader, network, criterion, optimizer, epoch, device, args, scaler)

        # evaluate on validation set
        test_acc1, test_acc5 = test(valid_loader, network, criterion, args, device)
        
        scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = test_acc1 > best_acc1
        best_acc1 = max(test_acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': network.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }, is_best, save_path)

        cur_lr = optimizer.param_groups[0]['lr']
        msg = ('Epoch: [{0}]\t'
                'LR:[{1}]\t'
                'Train_acc_1 {2}\t'
                'Train_acc_5 {3}\t'
                'Test_acc_1 {4}\t'
                'Test_acc_5 {5}\t'
                )
        logger_complete.info(msg.format(epoch+1, cur_lr, train_acc1, train_acc5, test_acc1, test_acc5))

    closer_logger(logger_complete)
    print('Finish training')

if __name__ == '__main__':
    data_paths = {
    "cifar10": "./data.cifar10",
    "cifar100": "/ssd1/cifar.python",
    "ImageNet16-120": "/ssd1/ImageNet16",
    "imagenet-1k": "/ssd2/chenwy/imagenet_final",
    }
    
    parser = argparse.ArgumentParser("TENAS")
    parser.add_argument('--data_path', type=str, default=data_paths['cifar10'], help='Path to dataset')
    parser.add_argument('--dataset', type=str, default= 'cifar10', choices=['cifar10', 'cifar100', 'ImageNet16-120', 'imagenet-1k'], help='Choose between cifar10/100/ImageNet16-120/imagenet-1k')
    parser.add_argument('--search_space_name', type=str, default='nas-bench-201',  help='space of operator candidates: nas-bench-201 or darts.')
    parser.add_argument('--NTK_batch_size', type=int, default=72, help='batch size for ntk')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('--save_dir', type=str, default="./output/prune-{space}/{dataset}".format(space="nas-bench-201", dataset='cifar10') ,help='Folder to save checkpoints and log.')
    parser.add_argument('--super_type', type=str, default='basic',  help='type of supernet: basic or nasnet-super')
    parser.add_argument('--rand_seed', type=int, default=0, help='manual seed')
    parser.add_argument('--precision', type=int, default=3, help='precision for % of changes of cond(NTK) and #Regions')
    parser.add_argument('--prune_number', type=int, default=1, help='number of operator to prune on each edge per round')
    parser.add_argument('--repeat', type=int, default=3, help='repeat calculation of NTK and Regions')
    parser.add_argument('--timestamp', default='none', type=str, help='timestamp for logging naming')
    parser.add_argument('--init', default='kaiming_normal', help='use gaussian init')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 0)')
    parser.add_argument('--max_nodes', type=int, default=4, help='The maximum number of nodes.')
    parser.add_argument('--track_running_stats', type=int, default=1, choices=[0, 1], help='Whether use track_running_stats or not in the BN layer.')
    
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
    args = parser.parse_args()
    main(args)
