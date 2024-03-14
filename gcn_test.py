import torch
import numpy as np
import scipy.stats as stats
from scipy.stats import spearmanr, kendalltau
import logging
import argparse
from GNN_proxy_tool.models import GCN
from GNN_proxy_tool.dataloader import Nas_101_Dataset,Nas_201_Dataset
from util import *
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
OPS =  ["input", "conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3", "output"]


def chunk(indices, size):
    return torch.split(torch.tensor(indices), size)

class MyBatchSampler(torch.utils.data.Sampler):
    def __init__(self, a_indices, b_indices, batch_size): 
        self.a_indices = a_indices
        self.b_indices = b_indices
        self.batch_size = batch_size
    def __len__(self):
        return (len(self.a_indices) + len(self.b_indices)) // self.batch_size
    def __iter__(self):
        random.shuffle(self.a_indices)
        random.shuffle(self.b_indices)
        a_batches  = chunk(self.a_indices, self.batch_size)
        b_batches = chunk(self.b_indices, self.batch_size)
        all_batches = list(a_batches + b_batches)
        all_batches = [batch.tolist() for batch in all_batches]
        random.shuffle(all_batches)
        return iter(all_batches)


def train(model, optimizer, loss, train_loader, epoch):
    logging.info("training gcn ... ")
    total_loss_train = 0
    count = 0
    total_difference = 0
    predicted = []
    ground_truth = []
    model.train()
    for i_batch, sample_batched in enumerate(train_loader):
        adjs, features, accuracys = sample_batched['adjacency_matrix'], sample_batched['operations'], \
                                    sample_batched['accuracy'].view(-1, 1)
        adjs, features, accuracys = adjs.cuda(), features.cuda(), accuracys.cuda()
        optimizer.zero_grad()
        outputs = model(features, adjs)
        loss_train = loss(outputs, accuracys)
        loss_train.backward()
        optimizer.step()
        count += 1
        difference = torch.mean(torch.abs(outputs - accuracys), 0)
        total_difference += difference.item()
        total_loss_train += loss_train.item()
        vx = outputs.cpu().detach().numpy().flatten()
        vy = accuracys.cpu().detach().numpy().flatten()
        predicted.append(vx)
        ground_truth.append(vy)
    predicted = np.hstack(predicted)
    ground_truth = np.hstack(ground_truth)
    corr, p = spearmanr(predicted, ground_truth)
    kt, pc = kendalltau(predicted, ground_truth)
    # logging.info("epoch {:d}".format(epoch + 1) + " train results:" + "train loss= {:.6f}".format(
    #     total_loss_train / count) + "abs_error:{:.6f}".format(total_difference / count) + "spearmanr_corr:{:.6f}".format(
    #     corr) + "kendalltau_corr:{:.6f}".format(
    #     kt))
    print("epoch {:d}".format(epoch + 1) + " train results:" + "train loss= {:.6f}".format(
        total_loss_train / count) + "abs_error:{:.6f}".format(total_difference / count) + "spearmanr_corr:{:.6f}".format(
        corr) + "kendalltau_corr:{:.6f}".format(
        kt))
    return kt, corr

def validate(model, loss, validation_loader, logging=None):
    loss_val = 0
    overall_difference = 0
    count = 0
    predicted = []
    ground_truth = []
    model.eval()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(validation_loader):
            adjs, features, accuracys = sample_batched['adjacency_matrix'], sample_batched['operations'], \
                                        sample_batched['accuracy'].view(-1, 1)
            adjs, features, accuracys = adjs.cuda(), features.cuda(), accuracys.cuda()
            outputs = model(features, adjs)
            loss_train = loss(outputs, accuracys)
            count += 1
            difference = torch.mean(torch.abs(outputs - accuracys), 0)
            overall_difference += difference.item()
            loss_val += loss_train.item()
            vx = outputs.cpu().detach().numpy().flatten()
            vy = accuracys.cpu().detach().numpy().flatten()
            predicted.append(vx)
            ground_truth.append(vy)
        predicted = np.hstack(predicted)
        ground_truth = np.hstack(ground_truth)
        # print("pridicted:", predicted)
        # print("ground_truth:", ground_truth)
        corr, p = spearmanr(predicted, ground_truth)
        kt, pc = kendalltau(predicted, ground_truth)
    if logging == None:
        print("test result " + " loss= {:.6f}".format(loss_val / count) + " abs_error:{:.6f}".format(
            overall_difference / count) + " corr:{:.6f}".format(corr) + " kendalltau_corr:{:.6f}".format(
        kt) + "\n")
    else:    
        logging.info("test result " + " loss= {:.6f}".format(loss_val / count) + " abs_error:{:.6f}".format(
            overall_difference / count) + " corr:{:.6f}".format(corr) + "kendalltau_corr:{:.6f}".format(
        kt))
    return kt, corr

def fit(args, lr, num_epoch, selected_loss, ifsigmoid, batch_size, logger_complete, save_path):
    if args.onehot:
        nfeat = len(OPS) + 1
    else:
        nfeat = args.embedding_size + 1

    gcn = GCN(
        # nfeat=len(self.__dataset[0]['operations'][0]) + 1,
        # nfeat=len(OPS) + 1,
        nfeat=nfeat,
        ifsigmoid=ifsigmoid
    )

    gcn = gcn.to(args.device)
    optimizer = torch.optim.AdamW(gcn.parameters(),lr=lr)
    # optimizer = torch.optim.Adam(gcn.parameters(),lr=lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

    dataset = Nas_101_Dataset(pickle_file=args.data_path)
    train_set, val_set = torch.utils.data.random_split(dataset, [args.train_split/100, 1-args.train_split/100])

    train_loader = DataLoader(train_set, batch_size=batch_size,
                                                shuffle=True)
    if args.use_201_to_train:
        dataset_201 = Nas_201_Dataset(pickle_file=args.data_path_201)
        # train_set_all = ConcatDataset([train_set, dataset_201])
        # train_loader = DataLoader(train_set_all, batch_size=batch_size,
        #                                         shuffle=True)
        
        new_dataset = ConcatDataset((train_set, dataset_201))
        a_len = train_set.__len__()
        ab_len = a_len + dataset_201.__len__()
        a_indices = list(range(a_len))
        b_indices = list(range(a_len, ab_len))

        batch_sampler = MyBatchSampler(a_indices, b_indices, batch_size)

        train_loader = DataLoader(new_dataset,  batch_sampler=batch_sampler)

    if args.use_201_to_tune:
        dataset_201 = Nas_201_Dataset(pickle_file=args.data_path_201)
        # train_set_all = ConcatDataset([train_set, dataset_201])

        train_loader = DataLoader(dataset_201, batch_size=batch_size,
                                                shuffle=True)
        checkpoint = torch.load(args.model_path, map_location=torch.device(args.device))
        gcn.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # print(checkpoint['optimizer'])
        # print(optimizer)
        scheduler.load_state_dict(checkpoint['scheduler'])

        validation_loader = DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True)
        print('pretuning test:\n')
        print('101 test:')
        validate(model=gcn, loss=selected_loss, validation_loader=validation_loader)
        print('201 test:')
        validate(model=gcn, loss=selected_loss, validation_loader=train_loader)

    # if args.use_201_to_test:
    #     dataset_201 = Nas_201_Dataset(pickle_file=args.data_path_201)
    #     validation_loader = DataLoader(dataset_201, batch_size=batch_size,
    #                                             shuffle=True)
    # else:
    #     validation_loader = DataLoader(val_set, batch_size=batch_size,
    #                                             shuffle=True)

    loss = selected_loss

    best_tau_test = 0
    best_rou_test = 0

    print('pretuning test')
    for epoch in range(num_epoch):
        tau_train, rou_train = train(model=gcn, optimizer=optimizer, loss=loss, train_loader=train_loader, epoch=epoch)
        tau_test, rou_test = validate(model=gcn, loss=loss, validation_loader=validation_loader)
        scheduler.step()
        
        is_best = tau_test > best_tau_test
        best_tau_test = max(tau_test, best_tau_test)
        best_rou_test = max(rou_test, best_rou_test)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': gcn.state_dict(),
            'best_tau_test': best_tau_test,
            'best_rou_test': best_rou_test,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }, is_best, save_path)

        cur_lr = optimizer.param_groups[0]['lr']
        msg = ('Epoch: [{0}]\t'
                'LR:[{1}]\t'
                'tau_train {2}\t'
                'rou_train {3}\t'
                'tau_test {4}\t'
                'rou_test {5}\t'
                )
        logger_complete.info(msg.format(epoch+1, cur_lr, tau_train, rou_train, tau_test, rou_test))

    closer_logger(logger_complete)


def main(args):
    torch.manual_seed(0)
    loss = torch.nn.MSELoss()

    if args.onehot:
        embedding_name = 'onehot'
    else:
        embedding_name = 'sentence_transformer'

    # create logger and save file
    file_complete = creating_path("new_gnn_result/GNN_Evaluation_Result_" + args.data_path[9:-4] + '_' + str(args.embedding_size),  "logs",
                                  file_name = str(args.batch_size)+ '_' + str(args.learning_rate) + '_' + str(args.train_split) + '_' + args.comment, extension='log')
    logger_complete = create_logger("complete", file_complete)

    save_path = creating_path("new_gnn_result/GNN_Evaluation_Result_" + args.data_path[9:-4] + '_' + str(args.embedding_size), "checkpoint" + str(args.batch_size)+ '_' + str(args.learning_rate) + str(args.train_split) + '_' + args.comment, 
                                  file_name = str(args.batch_size)+ '_' + str(args.learning_rate) + str(args.train_split) + '_' + args.comment)
    
    fit(args, args.learning_rate, args.epochs, loss, False, args.batch_size, logger_complete, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/yeq6/Research_project/MicroNAS/nasbench_101_dataset.pkl', help='location of the data')
    parser.add_argument('--data_path_201', type=str, default='/home/yeq6/Research_project/MicroNAS/nasbench_201_dataset_selected_1.pkl', help='location of the 201 data')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--train_split', type=float, default=90, help='percentage of data used for training')
    parser.add_argument('--embedding_size', type=int, default=768, help='embedding size')
    parser.add_argument('--onehot', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_201_to_test', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_201_to_train', action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_201_to_tune', action=argparse.BooleanOptionalAction)
    parser.add_argument('--model_path', type=str, default='/home/yeq6/Research_project/MicroNAS/GNN_Evaluation_Result/checkpoint/checkpoint.pth.tar', help='location of the GNN model')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
    parser.add_argument('--comment', type=str, default='', help='device')
    
    args = parser.parse_args()

    main(args)