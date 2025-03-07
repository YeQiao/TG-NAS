import os
import sys
import numpy as np
import time
import torch
import utils
from tqdm import tqdm
import glob
import logging
import argparse
import torch.nn as nn
import genotypes as genotypes
import torch.utils
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from thop import profile
from model import NetworkCIFAR as Network
from torchsummary import summary

parser = argparse.ArgumentParser("training cifar-10")
parser.add_argument('--workers', type=int, default=16, help='number of workers to load dataset')
parser.add_argument('--data', type=str, default='datapath', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DrNAS_cifar10', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

# set up logs
args.save = './experiments/{}/eval-{}-{}-{}-{}'.format(
    args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"), args.arch, args.seed)
if args.cutout:
    args.save += '-cutout-' + str(args.cutout_length)
if args.auxiliary:
    args.save += '-auxiliary-' + str(args.auxiliary_weight)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save)

CIFAR_CLASSES = 10

if args.dataset=='cifar100':
    CIFAR_CLASSES = 100


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------')
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model.drop_path_prob = 0
    macs, params = profile(model, inputs=(torch.randn(1, 3, 32, 32), ), verbose=False)
    logging.info("param = %f, flops = %f", params, macs)
    model = model.cuda()
    summary(model, (3,32,32))

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.dataset == 'cifar100':
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    else:
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_acc = train_acc = valid_acc = 0.0
    epoch_bar = tqdm(range(args.epochs), position=0, leave=True)
    for epoch in epoch_bar:
        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        description = 'Epoch [{}/{}] | LR:{} | Train:{} | Validation:{} | Best: {}'.format(epoch+1, args.epochs, scheduler.get_last_lr()[0], train_acc, valid_acc, best_acc)

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)
        description = 'Epoch [{}/{}] | LR:{} | Train:{} | Validation:{} | Best: {}'.format(epoch+1, args.epochs, scheduler.get_last_lr()[0], train_acc, valid_acc, best_acc)
        epoch_bar.set_description(description)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        if valid_acc > best_acc:
            best_acc = valid_acc
        logging.info('valid_acc %f, best_acc %f', valid_acc, best_acc)
        description = 'Epoch [{}/{}] | LR:{} | Train:{} | Validation:{} | Best: {}'.format(epoch+1, args.epochs, scheduler.get_last_lr()[0], train_acc, valid_acc, best_acc)
        epoch_bar.set_description(description)

        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/valid_best", best_acc, epoch)
        writer.add_scalar("acc/valid", valid_acc, epoch)

        scheduler.step()
        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
