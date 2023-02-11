import os
from statistics import mean
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from math import pi, cos, log, floor
from torch.optim.lr_scheduler import _LRScheduler
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pathlib
import dill
from enum import Enum
import re
import math
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score, f1_score
from collections import defaultdict
from scipy import stats 
from torch.utils.tensorboard import SummaryWriter
#lighting data augmentation
imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

#label smooth
class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss

def loss_fn_kd(outputs, labels, teacher_outputs, temp = 6, alpha = 1):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    T = temp
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
                             F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)

        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
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
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def Cifar10DataLoader(batch_size = 256):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    }
    image_datasets = {}
    image_datasets['train'] = dset.CIFAR10(root='./data.cifar10', train=True, download=True, transform=data_transforms['train'])
    image_datasets['val'] = dset.CIFAR10(root='./data.cifar10', train=False, download=True, transform=data_transforms['val'])

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True if x == 'train' else False,
                    num_workers=1, pin_memory=True) for x in ['train', 'val']}
    
    return dataloders

def Cifar100DataLoader(batch_size = 256):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    }
    image_datasets = {}
    image_datasets['train'] = dset.CIFAR100(root='./data.cifar100', train=True, download=True, transform=data_transforms['train'])
    image_datasets['val'] = dset.CIFAR100(root='./data.cifar100', train=False, download=True, transform=data_transforms['val'])

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True if x == 'train' else False,
                    num_workers=1, pin_memory=True) for x in ['train', 'val']}
    
    return dataloders

class CosineWarmupLR(_LRScheduler):
    '''
    Cosine lr decay function with warmup.
    Ref: https://github.com/PistonY/torch-toolbox/blob/master/torchtoolbox/optimizer/lr_scheduler.py
         https://github.com/Randl/MobileNetV3-pytorch/blob/master/cosine_with_warmup.py
    Lr warmup is proposed by 
        `Accurate, Large Minibatch SGD:Training ImageNet in 1 Hour`
        `https://arxiv.org/pdf/1706.02677.pdf`
    Cosine decay is proposed by 
        `Stochastic Gradient Descent with Warm Restarts`
        `https://arxiv.org/abs/1608.03983`
    Args:
        optimizer (Optimizer): optimizer of a model.
        iter_in_one_epoch (int): number of iterations in one epoch.
        epochs (int): number of epochs to train.
        lr_min (float): minimum(final) lr.
        warmup_epochs (int): warmup epochs before cosine decay.
        last_epoch (int): init iteration. In truth, this is last_iter
    Attributes:
        niters (int): number of iterations of all epochs.
        warmup_iters (int): number of iterations of all warmup epochs.
        cosine_iters (int): number of iterations of all cosine epochs.
    '''

    def __init__(self, optimizer, epochs, iter_in_one_epoch, lr_min=0, warmup_epochs=0, last_epoch=-1):
        self.lr_min = lr_min
        self.niters = epochs * iter_in_one_epoch
        self.warmup_iters = iter_in_one_epoch * warmup_epochs
        self.cosine_iters = iter_in_one_epoch * (epochs - warmup_epochs)
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return [(self.lr_min + (base_lr - self.lr_min) * self.last_epoch / self.warmup_iters) for base_lr in self.base_lrs]
        else:
            return [(self.lr_min + (base_lr - self.lr_min) * (1 + cos(pi * (self.last_epoch - self.warmup_iters) / self.cosine_iters)) / 2) for base_lr in self.base_lrs]

class CosineAnnealingWarmRestarts(_LRScheduler):
    '''
    copied from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingWarmRestarts
    Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{i}}\pi))
    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0`(after restart), set :math:`\eta_t=\eta_{max}`.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    '''

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, warmup_epochs=0, decay_rate=0.5):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if warmup_epochs < 0 or not isinstance(warmup_epochs, int):
            raise ValueError("Expected positive integer warmup_epochs, but got {}".format(warmup_epochs))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.decay_power = 0
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
        self.T_cur = self.last_epoch

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [(self.eta_min + (base_lr - self.eta_min) * self.T_cur / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr * (self.decay_rate**self.decay_power) - self.eta_min) * (1 + cos(pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        '''Step could be called after every batch update
        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         scheduler.step(epoch + i / iters)
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
        This function can be called in an interleaved way.
        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        '''
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch < self.warmup_epochs:
                self.T_cur = epoch
            else:
                epoch_cur = epoch - self.warmup_epochs
                if epoch_cur >= self.T_0:
                    if self.T_mult == 1:
                        self.T_cur = epoch_cur % self.T_0
                        self.decay_power = epoch_cur // self.T_0
                    else:
                        n = int(log((epoch_cur / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                        self.T_cur = epoch_cur - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                        self.T_i = self.T_0 * self.T_mult ** (n)
                        self.decay_power = n
                else:
                    self.T_i = self.T_0
                    self.T_cur = epoch_cur
        self.last_epoch = floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr



def save_checkpoint(state, is_best, save):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# todo use os package to cheak folder and create one if it is not exist
def model_save(model, test_accuracy_1, model_name, datasets):
    acc1 = re.sub('[^0-9.]', "", str(test_accuracy_1))[:-1]
    file_path_model = creating_path("pub","Ye","Resutls", file_name=model_name + "_" + datasets,
                                    extension='pth')

    try:
        torch.save(model, file_path_model)
    except AttributeError:
        torch.save(model, file_path_model, pickle_module=dill)


def params_save(model, epoch, optimizer, train_accuracy_1, test_accuracy_1, model_name, datasets):
    acc1 = re.sub('[^0-9.]', "", str(test_accuracy_1))[:-1]
    file_path_params = creating_path("pub","Ye","Resutls", file_name=model_name + "_",
                                     extension='pth.tar')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_stat_dict': optimizer.state_dict(),
        'top1_accuracy_train': train_accuracy_1,
        'top1_accuracy_test': test_accuracy_1,

    }, file_path_params)
    
def create_logger(name,file):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logging.basicConfig(filemode="w")
    logger.propagate = False
    return logger

def closer_logger(arg_logger):
    FLAG_SHUTDOWN = 1
    if FLAG_SHUTDOWN:
        logging.shutdown()
    else:
        handlers = arg_logger.handlers[:]
        for handler in handlers:
            handler.close()
            arg_logger.removeHandler(handler)

def creating_path(*folders_name: str, file_name:str = None, extension:str=None)->str:
    """
    This function is taken folder name, and file with desired extension for the file
    the goal to have poxis path in order to write or read from it
    it is more likily to use the function inside the code not for other things
    :return path_file
    """
    folders_name = list(folders_name)
    folders_path = "/".join([str(folder) for folder in folders_name])
    ""
    path = str(pathlib.Path.cwd()) + '/' + folders_path
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if file_name and extension is not None:
        file_path = str(path) + "/" + file_name + "." + extension
    else:
        file_path = str(path)
    return file_path
    
def parse_train(path_log_file):
    """
    Mohammed Alneamri
    the idea of this function is to prpaer for ploting
    so we have to plot the log file we generated using logging
    so it is a little bit of hacking I have to change little bit
    maybe using the index or rindex to have the epoch, loss , acc1, acc5
    using defaultdict becuae of having dict in dict
    the only difference here the log has data, time whcih has the index 1,2
    :param path_log_file: path of testing log file
    :return: list epochs, loss, acc1,acc5
    """

    loss = defaultdict(list)
    acc1 = defaultdict(list)
    acc5 = defaultdict(list)
    with open(path_log_file) as file:
        for line in file:
            items = line.split("\t")
            epoch = re.sub("[^0-9]", "", items[0][8:11])
            loss[int(epoch)].append(float(re.sub("[^0-9,.]", "",items[3][13:19])))
            acc1[int(epoch)].append(float(re.sub("[^0-9,.]", "",items[4][15:21])))
            acc5[int(epoch)].append(float(re.sub("[^0-9,.]", "",items[5][15:21])))
    loss_dict = {}
    acc1_dict = {}
    acc5_dict = {}
    for epoch, loss in loss.items():
        loss_dict[epoch] = round(sum(loss) / len(loss), 3)
    for epoch, acc1 in acc1.items():
        acc1_dict[epoch] = round(sum(acc1) / len(acc1), 3)
    for epoch, acc5 in acc5.items():
        acc5_dict[epoch] = round(sum(acc5) / len(acc5), 3)

    epochs = list(acc1_dict.keys())
    loss = list(loss_dict.values())
    acc1 = list(acc1_dict.values())
    acc5 = list(acc5_dict.values())

    return epochs, loss, acc1, acc5

def parse_test(path_log_file):
    """
    Mohammed Alneamri
    the idea of this function is to prpaer for ploting
    so we have to plot the log file we generated using logging
    so it is a little bit of hacking I have to change little bit
    maybe using the index or rindex to have the epoch, loss , acc1, acc5
    using defaultdict becuae of having dict in dict
    :param path_log_file: path of testing log file
    :return: list epochs, loss, acc1,acc5
    """
    loss = defaultdict(list)
    acc1 = defaultdict(list)
    acc5 = defaultdict(list)
    with open(path_log_file) as file:
        for line in file:
            items = line.split("\t")
            epoch = re.sub("[^0-9]", "", items[0][8:11])
            loss[int(epoch)].append(float(re.sub("[^0-9,.]", "",items[1][13:19])))
            acc1[int(epoch)].append(float(re.sub("[^0-9,.]", "",items[2][15:21])))
            acc5[int(epoch)].append(float(re.sub("[^0-9,.]", "",items[3][15:21])))
    loss_dict = {}
    acc1_dict = {}
    acc5_dict = {}
    for epoch, loss in loss.items():
        loss_dict[epoch] = round(sum(loss) / len(loss), 3)
    for epoch, acc1 in acc1.items():
        acc1_dict[epoch] = round(sum(acc1) / len(acc1), 3)
    for epoch, acc5 in acc5.items():
        acc5_dict[epoch] = round(sum(acc5) / len(acc5), 3)

    epochs = list(acc1_dict.keys())
    loss = list(loss_dict.values())
    acc1 = list(acc1_dict.values())
    acc5 = list(acc5_dict.values())
    
    return epochs, loss, acc1, acc5
    
def pretty_plot(model_name, dataset):
    """
    Written by Mohammed
    this is will plot the figures for you with accuracy anmd on the graph
    very nice and bea
    todo this function is not general can only paint four lines 2 for train and 2 fro test for one model
    todo I Need to make work with list of the patth and can plot differtrnt model in the same figures
    """
    # model_name = model.__class__.__name__
    path_train_log = 'pub/Ye/logs/'+model_name+'/'+dataset+'/train_logger/'+'__'+model_name+'__run___training.log'
    path_test_log = 'pub/Ye/logs/'+model_name+'/'+dataset+'/test_logger/'+'__'+model_name+'__run___test.log'

    fig, ax = plt.subplots()
    epochs, loss_train, acc1_train, acc5_train = parse_train(path_train_log)
    epochs, loss_test, acc1_test, acc5_test = parse_test(path_test_log)

    acc1_p_tr, = ax.plot(epochs, acc1_train, label='acc1_train', linestyle='--', color='g', marker='D', markersize=5,
                         linewidth=2)
    acc5_p_tr, = ax.plot(epochs, acc5_train, label='acc5_train', linestyle='--', color='b', marker='D', markersize=5,
                         linewidth=2)

    acc1_p_ts, = ax.plot(epochs, acc1_test, label='acc1_test', linestyle='--', color='r', marker='D', markersize=5,
                         linewidth=2)
    acc5_p_ts, = ax.plot(epochs, acc5_test, label='acc5_test', linestyle='--', color='c', marker='D', markersize=5,
                         linewidth=2)

    ax.set_title(model_name)
    # ax.se
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epochs')
    ax.axis('tight')
    plt.annotate('Train Acc1: {}'.format(max(acc1_train)),
                 xy=(acc1_train.index(max(acc1_train)), max(acc1_train)), xycoords='data',
                 xytext=(+10, +15), textcoords='offset points', fontsize=10,
                 bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(None)),
                 arrowprops=dict(arrowstyle="wedge,tail_width=1.",
                                 fc=(1.0, 0.7, 0.7), ec=(1., .5, .5),
                                 patchA=None,
                                 patchB=None,
                                 relpos=(0.2, 0.8),
                                 connectionstyle="arc3,rad=-0.1"))

    plt.annotate('Train Acc5: {}'.format(max(acc5_train)),
                 xy=(acc5_train.index(max(acc5_train)), max(acc5_train)), xycoords='data',
                 xytext=(-37, -30), textcoords='offset points', fontsize=10,
                 bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(None)),
                 arrowprops=dict(arrowstyle="wedge,tail_width=1.",
                                 fc=(1.0, 0.7, 0.7), ec=(1., .5, .5),
                                 patchA=None,
                                 patchB=None,
                                 relpos=(0.2, 0.8),
                                 connectionstyle="arc3,rad=-0.1"))

    plt.annotate('Test Acc1: {}'.format(max(acc1_test)),
                 xy=(acc1_test.index(max(acc1_test)), max(acc1_test)), xycoords='data',
                 xytext=(+10, +15), textcoords='offset points', fontsize=10,
                 bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(None)),
                 arrowprops=dict(arrowstyle="wedge,tail_width=1.",
                                 fc=(1.0, 0.7, 0.7), ec=(1., .5, .5),
                                 patchA=None,
                                 patchB=None,
                                 relpos=(0.2, 0.8),
                                 connectionstyle="arc3,rad=-0.1"))

    plt.annotate('Test Acc5: {}'.format(max(acc5_test)),
                 xy=(acc5_test.index(max(acc5_test)), max(acc5_test)), xycoords='data',
                 xytext=(+10, -15), textcoords='offset points', fontsize=10,
                 bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(None)),
                 arrowprops=dict(arrowstyle="wedge,tail_width=1.",
                                 fc=(1.0, 0.7, 0.7), ec=(1., .5, .5),
                                 patchA=None,
                                 patchB=None,
                                 relpos=(0.2, 0.8),
                                 connectionstyle="arc3,rad=-0.1"))

    ax.legend(handles=[acc1_p_tr, acc5_p_tr, acc1_p_ts, acc5_p_ts])
    ax.grid()
    figurename = creating_path("reports","accuracy","figures_pretty", model_name, "png")
    plt.savefig(figurename, dpi=300)
    plt.show()

def classic_plot(model_name,dataset):
    """
    this smilar to the pretty one, the only difference is not having rad buubles
    """
    # model_name = model.__class__.__name__
    path_train_log = 'pub/Ye/logs/'+model_name+'/'+dataset+'/train_logger/'+'__'+model_name+'__run___training.log'
    path_test_log = 'pub/Ye/logs/'+model_name+'/'+dataset+'/test_logger/'+'__'+model_name+'__run___test.log'
    
    fig, ax = plt.subplots()
    epochs, loss_train, acc1_train, acc5_train = parse_train(path_train_log)
    epochs, loss_test, acc1_test, acc5_test = parse_test(path_test_log)

    acc1_p_tr, = ax.plot(epochs, acc1_train, label='acc1_train', linestyle='--', color='g', marker='D',
                         markersize=5,
                         linewidth=2)
    acc5_p_tr, = ax.plot(epochs, acc5_train, label='acc5_train', linestyle='--', color='b', marker='D',
                         markersize=5,
                         linewidth=2)

    acc1_p_ts, = ax.plot(epochs, acc1_test, label='acc1_test', linestyle='--', color='r', marker='D', markersize=5,
                         linewidth=2)
    acc5_p_ts, = ax.plot(epochs, acc5_test, label='acc5_test', linestyle='--', color='c', marker='D', markersize=5,
                         linewidth=2)

    ax.set_title(model_name)
    # ax.se
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epochs')
    ax.axis('tight')
    plt.annotate('Train Acc1: {}'.format(max(acc1_train)),
                 xy=(acc1_train.index(max(acc1_train)), max(acc1_train)), xycoords='data',
                 xytext=(+10, +15), textcoords='offset points', fontsize=10,
                 arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=45"))

    plt.annotate('Train Acc5: {}'.format(max(acc5_train)),
                 xy=(acc5_train.index(max(acc5_train)), max(acc5_train)), xycoords='data',
                 xytext=(-37, -30), textcoords='offset points', fontsize=10,
                 arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=45"))

    plt.annotate('Test Acc1: {}'.format(max(acc1_test)),
                 xy=(acc1_test.index(max(acc1_test)), max(acc1_test)), xycoords='data',
                 xytext=(+10, +15), textcoords='offset points', fontsize=10,
                 arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=45"))

    plt.annotate('Test Acc5: {}'.format(max(acc5_test)),
                 xy=(acc5_test.index(max(acc5_test)), max(acc5_test)), xycoords='data',
                 xytext=(+10, -15), textcoords='offset points', fontsize=10,
                 arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=45"))

    ax.legend(handles=[acc1_p_tr, acc5_p_tr, acc1_p_ts, acc5_p_ts])
    ax.grid()
    figurename = creating_path("reports","accuracy","figures_classic", model_name, "png")
    plt.savefig(figurename, dpi=300)
    plt.show()

def plot_multi(*args, title, same_figure=True, save_fig=True):
    if same_figure == True:
        logs = list(args)
        models_names = []
        logs_type = []
        fig, ax = plt.subplots()
        plt.rc('lines', linewidth=1)
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
                                   cycler('linestyle', ['-', '--', ':', '-.'])))
        for log in logs:
            model_name, log_type = log.split("__")[1], log.split("__")[3]
            if log_type == "_training.log":
                epochs, loss_train, acc1_train, acc5_train = parse_train(log)
                acc1_p_tr, = ax.plot(epochs, acc1_train, label='acc1_train_' + str(model_name))
                acc5_p_tr, = ax.plot(epochs, acc5_train, label='acc5_train_' + str(model_name))
            if log_type == "_test.log":
                epochs, loss_test, acc1_test, acc5_test = parse_test(log)
                acc1_p_ts, = ax.plot(epochs, acc1_test, label='acc1_test_' + str(model_name))
                acc5_p_ts, = ax.plot(epochs, acc5_test, label='acc5_test_' + str(model_name))

                ax.set_title(title)
                ax.set_ylabel('Accuracy')
                ax.set_xlabel('Epochs')
                ax.axis('tight')

                ax.legend()
                ax.grid()
    if same_figure == False:
        logs = list(args)
        models_names = []
        logs_type = []
        fig, (ax0, ax1) = plt.subplots(2, 1)
        plt.rc('lines', linewidth=1)
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
                                   cycler('linestyle', ['-', '--', ':', '-.'])))
        for log in logs:
            model_name, log_type = log.split("__")[1], log.split("__")[3]
            if log_type == "_training.log":
                epochs, loss_train, acc1_train, acc5_train = parse_train(log)
                acc1_p_tr, = ax0.plot(epochs, acc1_train, label='acc1_train_' + str(model_name))
                acc5_p_tr, = ax0.plot(epochs, acc5_train, label='acc5_train_' + str(model_name))
            if log_type == "_test.log":
                epochs, loss_test, acc1_test, acc5_test = parse_test(log)
                acc1_p_ts, = ax1.plot(epochs, acc1_test, label='acc1_test_' + str(model_name))
                acc5_p_ts, = ax1.plot(epochs, acc5_test, label='acc5_test_' + str(model_name))

                ax0.set_title(title)
                ax0.set_ylabel('Accuracy')
                ax0.set_xlabel('Epochs')
                ax0.axis('tight')

                ax0.legend()
                ax0.grid()
                ax1.set_title(title)
                ax1.set_ylabel('Accuracy')
                ax1.set_xlabel('Epochs')
                ax1.axis('tight')

                ax1.legend()
                ax1.grid()

            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=1.2,
                                wspace=0.4,
                                hspace=0.4)

        if save_fig == True:
            figurename = creating_path("figures", title, "png")
            plt.savefig(figurename, dpi=300)
            plt.show()
        else:
            plt.show()


from pathlib import Path


def plot_multix(*models, datasets, train_acc1=True, train_acc5=True, test_acc1=True, test_acc5=True, title,
               same_figure=True, save_fig=True):
    models = list(models)
    fig, ax = plt.subplots()
    plt.rc('lines', linewidth=1)
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
                               cycler('linestyle', ['-', '--', ':', '-.'])))
    for model in models:
        model_name = model.__class__.__name__
        path_folder = './logs/' + model_name + "/" + datasets
        folder = Path(path_folder)
        for file in folder.iterdir():
            if train_acc1 == True or train_acc5 == True:
                if str(file).split('/')[-1] == 'train_logger':
                    for log in Path(file).iterdir():
                        epochs, loss_train, acc1_train, acc5_train = parse_train(str(log))
                        if train_acc1 == True and train_acc5 == False:
                            acc1_p_tr, = ax.plot(epochs, acc1_train, label='acc1_train_' + str(model_name))
                        if train_acc1 == False and train_acc5 == True:
                            acc5_p_tr, = ax.plot(epochs, acc5_train, label='acc5_train_' + str(model_name))
                        if train_acc1 == True and train_acc5 == True:
                            acc1_p_tr, = ax.plot(epochs, acc1_train, label='acc1_train_' + str(model_name))
                            acc5_p_tr, = ax.plot(epochs, acc5_train, label='acc5_train_' + str(model_name))
            if test_acc1 == True or test_acc5 == True:
                if str(file).split('/')[-1] == 'test_logger':
                    for log in Path(file).iterdir():
                        epochs, loss_test, acc1_test, acc5_test = parse_test(str(log))
                        if test_acc1 == True and test_acc5 == False:
                            acc1_p_ts, = ax.plot(epochs, acc1_test, label='acc1_test_' + str(model_name))
                        if test_acc1 == False and test_acc5 == True:
                            acc5_p_ts, = ax.plot(epochs, acc5_test, label='acc5_test_' + str(model_name))
                        if test_acc1 == True and test_acc5 == True:
                            acc1_p_ts, = ax.plot(epochs, acc1_test, label='acc1_test_' + str(model_name))
                            acc5_p_ts, = ax.plot(epochs, acc5_test, label='acc5_test_' + str(model_name))

            ax.set_title(title)
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Epochs')
            ax.axis('tight')

            ax.legend()
            ax.grid()

def plot_hist_conv_linear(model,save_fig=False,plt_show=True,model_name=None):
    print('start to plot')
    layers = {}
    weights = {}
    counter = 0
    if model_name == None:
        model_name = model.__class__.__name__
    else:
        model_name= model_name
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            layers[layer.__class__.__name__ + "_" + str(counter)] = "x".join(map(str, layer.weight.shape))
            weights[layer.__class__.__name__ + "_" + str(counter)] = layer.weight.cpu().detach().numpy().flatten()

        if isinstance(layer, torch.nn.Linear):
            layers[layer.__class__.__name__ + "_" + str(counter)] = "x".join(map(str, layer.weight.shape))
            weights[layer.__class__.__name__ + "_" + str(counter)] = layer.weight.cpu().detach().numpy().flatten()
        if isinstance(layer,torch.nn.Conv2d) or isinstance(layer,torch.nn.Conv2d):
            counter += 1
    for idx, params in weights.items():
        (mean_fitted, std_fitted) = stats.norm.fit(params)
        print('mean of layer No. ' + idx + ' ', mean_fitted)
        print('std of layer No. ' + idx + ' ', std_fitted)
        x = np.linspace(min(params), max(params), 600)
        weight_guass_fit = stats.norm.pdf(x, loc=mean_fitted, scale=std_fitted)
        n, bins, patchers = plt.hist(params, histtype='stepfilled',
                                     cumulative=False, bins=600, density=True)

        # plt.plot(x, weight_guass_fit, label='guess')
        plt.title(idx + " : " + layers[idx])
        # plt.legend()
        if save_fig == True:
            figure_name = creating_path("reports","filters","distrbutions",model_name,file_name=idx + "__" + layers[idx],extension='png')
            plt.savefig(figure_name, dpi=150, bbox_inches='tight')
        if plt_show == True:
            plt.show()