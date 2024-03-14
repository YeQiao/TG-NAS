from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import scipy.stats as stats
from scipy.stats import spearmanr, kendalltau
import logging
import argparse
from GNN_proxy_tool.models import GCN
from GNN_proxy_tool.dataloader import Nas_101_Dataset, Nas_201_Dataset
from util import *
from torch.optim.lr_scheduler import StepLR
OPS_101 =  ["input", "conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3", "output"]
OPS_201 =  ["input", "skip_connect" ,"conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3", "output"]

def validate(args, model):

    loss = torch.nn.MSELoss()
    if args.test_201:
        dataset = Nas_201_Dataset(pickle_file=args.data_path_201)
    else:    
        dataset = Nas_101_Dataset(pickle_file=args.data_path_101)
    # train_set, val_set = torch.utils.data.random_split(dataset, [args.train_split/100, 1-args.train_split/100])
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                shuffle=False)
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
            # print(adjs)
            # break
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
        print("pridicted dimension:", predicted.shape)
        print("ground_truth dimension:", ground_truth.shape)

        scaler = MinMaxScaler(feature_range=(np.min(ground_truth), np.max(ground_truth)))
        normalized_arr = scaler.fit_transform(predicted.reshape(-1, 1)).flatten()
        
        combined = np.column_stack((ground_truth, normalized_arr))
        print("combined dimension:", combined.shape)
        np.savetxt('test'+'_'+args.comment+'.txt', combined, fmt='%.4f')
        np.save('201_file'+'_'+args.comment, combined)
        
        corr, p = spearmanr(predicted, ground_truth)
        kt, pc = kendalltau(predicted, ground_truth)
        print("test result " + " loss= {:.6f}".format(loss_val / count) + " abs_error:{:.6f}".format(
            overall_difference / count) + " corr:{:.6f}".format(corr) + " kendalltau_corr:{:.6f}".format(
        kt) + "\n")

def main(args):
    torch.manual_seed(0)
    
    gcn = GCN(
        # nfeat=len(self.__dataset[0]['operations'][0]) + 1,
        # nfeat=len(OPS_101) + 1,
        nfeat=args.embedding_size + 1, # need some think ?
        ifsigmoid=False
    )
    gcn.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device))['state_dict']) 
    gcn = gcn.to(args.device)
    validate(args, gcn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_101', type=str, default='/home/yeq6/Research_project/MicroNAS/nasbench_101_dataset.pkl', help='location of the 101 data')
    parser.add_argument('--data_path_201', type=str, default='/home/yeq6/Research_project/MicroNAS/nasbench_201_dataset_selected_1.pkl', help='location of the 201 data')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--model_path', type=str, default='/home/yeq6/Research_project/MicroNAS/GNN_Evaluation_Result/checkpoint/checkpoint.pth.tar', help='location of the GNN model')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--embedding_size', type=int, default=768, help='embedding size')
    parser.add_argument('--comment', type=str, default='', help='comment')
    parser.add_argument('--test_201', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    main(args)