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
from torchsummary import summary
from util import *
from util import pytorch2onnx
from GNN_proxy_tool.models import GCN
from GNN_proxy_tool.utils.gcn_utils import padzero, add_global_node
from sentence_transformers import SentenceTransformer
INF = 1000  # used to mark prunned operators
darts_OPS = ['input', 'none', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'avg_pool_3x3', 'max_pool_3x3', 'output']

 # LUT for latency modeling
latency_table_nasbench = {
    'none' : 0, 
    'skip_connect' : 1.43, 
    'nor_conv_1x1' : 23.49, 
    'nor_conv_3x3' : 80.26,
    'avg_pool_3x3' : 24.02
}

latency_table_darts = {
    'none' : 0, 
    'skip_connect' : 1.963, 
    'sep_conv_3x3' : 180.838, 
    'sep_conv_5x5' : 223.902, 
    'dil_conv_3x3' : 183.645, 
    'dil_conv_5x5' : 238.948, 
    'avg_pool_3x3' : 15.672, 
    'max_pool_3x3' : 15.054
}

def ops_embedding(args):
    model = SentenceTransformer(args.embedding_model)
    print('embedding model loaded')
    # Our sentences we like to encode
    sentences = ['Input',
                
                'None',

                'Residual connection',
                
                'depth-wise separable convolution 3x3',

                'depth-wise separable convolution 5x5',
                
                'dilated convolution 3x3',

                'dilated convolution 5x5',
                
                'Average pooling 3x3',

                'Max pooling 3x3',

                'Output'
                ]
    embeddings = model.encode(sentences, device = 'cuda:0')
    return embeddings

def transform_matrix_embedding(genotype, embeddings):
    # print(genotype)
    normal = genotype['normal']
    node_num = len(normal)+3
    adj = np.zeros((node_num, node_num))
    ops = []
    ops.append(embeddings[0])
    ops.append(embeddings[0]) # question remaining, is it aproprate to treat two input node the same? 
    for i in range(len(normal)):
        op, connect = normal[i]
        if connect == 0 or connect==1:
            adj[connect][i+2] = 1
        else:
            adj[(connect-2)*2+2][i+2] = 1
            adj[(connect-2)*2+3][i+2] = 1
        # ops[i+2][darts_OPS.index(op)] = 1

        op_embedding = embeddings[darts_OPS.index(op)]
        ops.append(op_embedding)
    adj[2:-1, -1] = 1
    # ops[0:2, 0] = 1
    # ops[-1][-1] = 1
    ops.append(embeddings[-1])
    return adj, ops

def get_gnn_pridiction(genotype, args, embeddings):
    gcn = GCN(
        # nfeat=len(self.__dataset[0]['operations'][0]) + 1,
        # nfeat=len(OPS_101) + 1,
        nfeat=args.embedding_size + 1, # need some think ?
        ifsigmoid=False
        )
    gcn.load_state_dict(torch.load(args.gnn_model_path, map_location=torch.device(args.device))['state_dict']) 
    gcn = gcn.to(args.device)
    gcn.eval()
    adj, ops = transform_matrix_embedding(genotype, embeddings)

    adjacency_matrix = padzero(np.array(adj,dtype=np.float32), ifAdj=True, maxsize=7)
    adjacency_matrix = add_global_node(adjacency_matrix, ifAdj = True)
    operations = padzero(np.array(ops,dtype=np.float32), ifAdj=False, maxsize=7)
    operations = add_global_node(operations, ifAdj = False)
    adjacency_matrix, operations = torch.from_numpy(adjacency_matrix).unsqueeze(dim=0).to(args.device), torch.from_numpy(operations).unsqueeze(dim=0).to(args.device)
    output = gcn(operations, adjacency_matrix)
    output = output.cpu().detach().numpy().flatten()[0]
    return output

def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def kaiming_normal_fanout_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    elif method == 'kaiming_norm_fanout':
        model.apply(kaiming_normal_fanout_init)
    return model


def round_to(number, precision, eps=1e-8):
    # round to significant figure
    dtype = type(number)
    if number == 0:
        return number
    sign = number / abs(number)
    number = abs(number) + eps
    power = math.floor(math.log(number, 10)) + 1
    if dtype == int:
        return int(sign * round(number*10**(-power), precision) * 10**(power))
    else:
        return sign * round(number*10**(-power), precision) * 10**(power)


def prune_func_rank(xargs, arch_parameters, model_config, model_config_thin, train_loader, lrc_model, search_space, precision=10, prune_number=1, embeddings = None):
    # arch_parameters now has three dim: cell_type, edge, op
    network_origin = get_cell_based_tiny_net(model_config).cuda().train()
    # init_model(network_origin, xargs.init)
    network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda().train()
    # init_model(network_thin_origin, xargs.init)

    for alpha in arch_parameters:
        alpha[:, 0] = -INF
    network_origin.set_alphas(arch_parameters)
    network_thin_origin.set_alphas(arch_parameters)

    GNN_proxy_originl = get_gnn_pridiction(network_origin.genotype(ifdic = True), xargs, embeddings)
    print('original_output: ', GNN_proxy_originl)

    alpha_active = [(nn.functional.softmax(alpha, 1) > 0.01).float() for alpha in arch_parameters]
    prune_number = min(prune_number, alpha_active[0][0].sum()-1)  # adjust prune_number based on current remaining ops on each edge
    ntk_all = []  # (ntk, (edge_idx, op_idx))
    regions_all = []  # (regions, (edge_idx, op_idx))
    GNN_proxy_all = []  # (regions, (edge_idx, op_idx))
    choice2regions = {}  # (edge_idx, op_idx): regions
    pbar = tqdm(total=int(sum(alpha.sum() for alpha in alpha_active)), position=0, leave=True)
    for idx_ct in range(len(arch_parameters)):
        # cell type (ct): normal or reduce
        for idx_edge in range(len(arch_parameters[idx_ct])):
            # edge
            if alpha_active[idx_ct][idx_edge].sum() == 1:
                # only one op remaining
                continue
            for idx_op in range(len(arch_parameters[idx_ct][idx_edge])):
                # op
                if alpha_active[idx_ct][idx_edge, idx_op] > 0:
                    # this edge-op not pruned yet
                    _arch_param = [alpha.detach().clone() for alpha in arch_parameters]
                    _arch_param[idx_ct][idx_edge, idx_op] = -INF
                    # ##### get ntk (score) ########
                    network = get_cell_based_tiny_net(model_config).cuda().train()
                    network.set_alphas(_arch_param)
                    # ntk_delta = []
                    # repeat = xargs.repeat
                    # for _ in range(repeat):
                    #     # random reinit
                    #     init_model(network_origin, xargs.init+"_fanout" if xargs.init.startswith('kaiming') else xargs.init)  # for backward
                    #     # make sure network_origin and network are identical
                    #     for param_ori, param in zip(network_origin.parameters(), network.parameters()):
                    #         param.data.copy_(param_ori.data)
                    #     network.set_alphas(_arch_param)
                    #     # NTK cond TODO #########
                    #     ntk_origin, ntk = get_ntk_n(loader, [network_origin, network], recalbn=0, train_mode=True, num_batch=1)
                    #     # ####################
                    #     ntk_delta.append(round((ntk_origin - ntk) / ntk_origin, precision))  # higher the more likely to be prunned
                    # ntk_all.append([np.mean(ntk_delta), (idx_ct, idx_edge, idx_op)])  # change of ntk
                    # network.zero_grad()
                    # network_origin.zero_grad()
                    # #############################
                    # network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda()
                    # network_thin_origin.set_alphas(arch_parameters)
                    # network_thin_origin.train()
                    # network_thin = get_cell_based_tiny_net(model_config_thin).cuda()
                    # network_thin.set_alphas(_arch_param)
                    # network_thin.train()
                    # with torch.no_grad():
                    #     _linear_regions = []
                    #     repeat = xargs.repeat
                    #     for _ in range(repeat):
                    #         # random reinit
                    #         init_model(network_thin_origin, xargs.init+"_fanin" if xargs.init.startswith('kaiming') else xargs.init)  # for forward
                    #         # make sure network_thin and network_thin_origin are identical
                    #         for param_ori, param in zip(network_thin_origin.parameters(), network_thin.parameters()):
                    #             param.data.copy_(param_ori.data)
                    #         network_thin.set_alphas(_arch_param)
                    #         #####
                    #         lrc_model.reinit(models=[network_thin_origin, network_thin], seed=xargs.rand_seed)
                    #         _lr, _lr_2 = lrc_model.forward_batch_sample()
                    #         _linear_regions.append(round((_lr - _lr_2) / _lr, precision))  # change of #Regions, lower the more likely to be prunned
                    #         lrc_model.clear()
                    #     linear_regions = np.mean(_linear_regions)
                    #     regions_all.append([linear_regions, (idx_ct, idx_edge, idx_op)])
                    #     choice2regions[(idx_ct, idx_edge, idx_op)] = linear_regions
                    #############################

                    network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda()
                    network_thin_origin.set_alphas(arch_parameters)
                    # network_thin_origin.train()
                    network_thin = get_cell_based_tiny_net(model_config_thin).cuda()
                    network_thin.set_alphas(_arch_param)
                    # network_thin.train()


                    GNN_proxy = get_gnn_pridiction(network.genotype(ifdic = True), xargs, embeddings)
                    print('GNN_proxy_output: ', GNN_proxy)
                    delta_gnn_proxy = round((GNN_proxy_originl - GNN_proxy) / GNN_proxy_originl, precision)
                    GNN_proxy_all.append([delta_gnn_proxy, (idx_ct, idx_edge, idx_op)])

                    torch.cuda.empty_cache()
                    del network_thin
                    del network_thin_origin
                    pbar.update(1)
    # ntk_all = sorted(ntk_all, key=lambda tup: round_to(tup[0], precision), reverse=True)  # descending: we want to prune op to decrease ntk, i.e. to make ntk_origin > ntk
    # print("NTK conds:", ntk_all)
    rankings = {}  # dict of (cell_idx, edge_idx, op_idx): [ntk_rank, regions_rank]

    GNN_proxy_all = sorted(GNN_proxy_all, key=lambda tup: round_to(tup[0], precision), reverse=False)  # ascending: we want to prune op to increase lr, i.e. to make lr < lr_2
    # print('GNN_proxy_all: ', GNN_proxy_all)    
    for idx, data in enumerate(GNN_proxy_all):
        if idx == 0:
            rankings[data[1]] = [idx]
        else:
            if data[0] == GNN_proxy_all[idx-1][0]:
                # same ntk as previous
                rankings[data[1]] = [ rankings[GNN_proxy_all[idx-1][1]][0] ]
            else:
                rankings[data[1]] = [ rankings[GNN_proxy_all[idx-1][1]][0] + 1 ]
    # for idx, data in enumerate(ntk_all):
    #     if idx == 0:
    #         rankings[data[1]] = [idx]
    #     else:
    #         if data[0] == ntk_all[idx-1][0]:
    #             # same ntk as previous
    #             rankings[data[1]] = [ rankings[ntk_all[idx-1][1]][0] ]
    #         else:
    #             rankings[data[1]] = [ rankings[ntk_all[idx-1][1]][0] + 1 ]
    # regions_all = sorted(regions_all, key=lambda tup: round_to(tup[0], precision), reverse=False)  # ascending: we want to prune op to increase lr, i.e. to make lr < lr_2
    # # print("#Regions:", regions_all)
    # for idx, data in enumerate(regions_all):
    #     if idx == 0:
    #         rankings[data[1]].append(idx)
    #     else:
    #         if data[0] == regions_all[idx-1][0]:
    #             # same #Regions as previous
    #             rankings[data[1]].append(rankings[regions_all[idx-1][1]][1])
    #         else:
    #             rankings[data[1]].append(rankings[regions_all[idx-1][1]][1]+1)
    # rankings_list = [[k, v] for k, v in rankings.items()]  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
    # # ascending by sum of two rankings
    # rankings_sum = sorted(rankings_list, key=lambda tup: sum(tup[1]), reverse=False)  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
    
    rankings_list = [[k, v] for k, v in rankings.items()]  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
    # ascending by sum of two rankings
    rankings_sum = sorted(rankings_list, key=lambda tup: tup[0], reverse=False)  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
    
    edge2choice = {}  # (cell_idx, edge_idx): list of (cell_idx, edge_idx, op_idx) of length prune_number
    for (cell_idx, edge_idx, op_idx), [GNN_rank] in rankings_sum:
        if (cell_idx, edge_idx) not in edge2choice:
            edge2choice[(cell_idx, edge_idx)] = [(cell_idx, edge_idx, op_idx)]
        elif len(edge2choice[(cell_idx, edge_idx)]) < prune_number:
            edge2choice[(cell_idx, edge_idx)].append((cell_idx, edge_idx, op_idx))
    choices_edges = list(edge2choice.values())
    print("Final Ranking:", rankings_sum)
    # print("Pruning Choices:", choices_edges)
    for choices in choices_edges:
        for (cell_idx, edge_idx, op_idx) in choices:
            arch_parameters[cell_idx].data[edge_idx, op_idx] = -INF

    return arch_parameters, choices_edges


def prune_func_rank_group(xargs, arch_parameters, model_config, model_config_thin, train_loader = None, lrc_model = None, search_space = "darts", edge_groups=[(0, 2), (2, 5), (5, 9), (9, 14)], num_per_group=2, precision=10 ,embeddings = None):
    # arch_parameters now has three dim: cell_type, edge, op
    # embeddings = ops_embedding(xargs)
    network_origin = get_cell_based_tiny_net(model_config).cuda().train()
    # init_model(network_origin, xargs.init)
    network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda().train()
    # init_model(network_thin_origin, xargs.init)

    for alpha in arch_parameters:
        alpha[:, 0] = -INF
    network_origin.set_alphas(arch_parameters)
    network_thin_origin.set_alphas(arch_parameters)
    GNN_proxy_originl = get_gnn_pridiction(network_origin.genotype(ifdic = True), xargs, embeddings)

    alpha_active = [(nn.functional.softmax(alpha, 1) > 0.01).float() for alpha in arch_parameters]
    ntk_all = []  # (ntk, (edge_idx, op_idx))
    regions_all = []  # (regions, (edge_idx, op_idx))
    GNN_proxy_all = []  # (regions, (edge_idx, op_idx))
    choice2regions = {}  # (edge_idx, op_idx): regions
    pbar = tqdm(total=int(sum(alpha.sum() for alpha in alpha_active)), position=0, leave=True)
    assert edge_groups[-1][1] == len(arch_parameters[0])
    for idx_ct in range(len(arch_parameters)):
        # cell type (ct): normal or reduce
        for idx_group in range(len(edge_groups)):
            edge_group = edge_groups[idx_group]
            # print("Pruning cell %s group %s.........."%("normal" if idx_ct == 0 else "reduction", str(edge_group)))
            if edge_group[1] - edge_group[0] <= num_per_group:
                # this group already meets the num_per_group requirement
                pbar.update(1)
                continue
            for idx_edge in range(edge_group[0], edge_group[1]):
                # edge
                for idx_op in range(len(arch_parameters[idx_ct][idx_edge])):
                    # op
                    if alpha_active[idx_ct][idx_edge, idx_op] > 0:
                        # this edge-op not pruned yet
                        _arch_param = [alpha.detach().clone() for alpha in arch_parameters]
                        _arch_param[idx_ct][idx_edge, idx_op] = -INF
                        # ##### get ntk (score) ########
                        network = get_cell_based_tiny_net(model_config).cuda().train()
                        network.set_alphas(_arch_param)
                        # ntk_delta = []
                        # repeat = xargs.repeat
                        # for _ in range(repeat):
                        #     # random reinit
                        #     init_model(network_origin, xargs.init+"_fanout" if xargs.init.startswith('kaiming') else xargs.init)  # for backward
                        #     # make sure network_origin and network are identical
                        #     for param_ori, param in zip(network_origin.parameters(), network.parameters()):
                        #         param.data.copy_(param_ori.data)
                        #     network.set_alphas(_arch_param)
                        #     # NTK cond TODO #########
                        #     ntk_origin, ntk = get_ntk_n(loader, [network_origin, network], recalbn=0, train_mode=True, num_batch=1)
                        #     # ####################
                        #     ntk_delta.append(round((ntk_origin - ntk) / ntk_origin, precision))
                        # ntk_all.append([np.mean(ntk_delta), (idx_ct, idx_edge, idx_op)])  # change of ntk
                        # network.zero_grad()
                        # network_origin.zero_grad()
                        #############################
                        network_thin_origin = get_cell_based_tiny_net(model_config_thin).cuda()
                        network_thin_origin.set_alphas(arch_parameters)
                        # network_thin_origin.train()
                        network_thin = get_cell_based_tiny_net(model_config_thin).cuda()
                        network_thin.set_alphas(_arch_param)
                        # network_thin.train()




                        GNN_proxy = get_gnn_pridiction(network.genotype(ifdic = True), xargs, embeddings)
                        delta_gnn_proxy = round((GNN_proxy_originl - GNN_proxy) / GNN_proxy_originl, precision)
                        GNN_proxy_all.append([delta_gnn_proxy, (idx_ct, idx_edge, idx_op)])




                        # with torch.no_grad():
                        #     _linear_regions = []
                        #     repeat = xargs.repeat
                        #     for _ in range(repeat):
                        #         # random reinit
                        #         init_model(network_thin_origin, xargs.init+"_fanin" if xargs.init.startswith('kaiming') else xargs.init)  # for forward
                        #         # make sure network_thin and network_thin_origin are identical
                        #         for param_ori, param in zip(network_thin_origin.parameters(), network_thin.parameters()):
                        #             param.data.copy_(param_ori.data)
                        #         network_thin.set_alphas(_arch_param)
                        #         #####
                        #         lrc_model.reinit(models=[network_thin_origin, network_thin], seed=xargs.rand_seed)
                        #         _lr, _lr_2 = lrc_model.forward_batch_sample()
                        #         _linear_regions.append(round((_lr - _lr_2) / _lr, precision))  # change of #Regions
                        #         lrc_model.clear()
                        #     linear_regions = np.mean(_linear_regions)
                        #     regions_all.append([linear_regions, (idx_ct, idx_edge, idx_op)])
                        #     choice2regions[(idx_ct, idx_edge, idx_op)] = linear_regions
                        #############################
                        torch.cuda.empty_cache()
                        del network_thin
                        del network_thin_origin
                        pbar.update(1)


            # stop and prune this edge group
            # ntk_all = sorted(ntk_all, key=lambda tup: round_to(tup[0], precision), reverse=True)  # descending: we want to prune op to decrease ntk, i.e. to make ntk_origin > ntk
            # print("NTK conds:", ntk_all)\
                        
            rankings = {}  # dict of (cell_idx, edge_idx, op_idx): [ntk_rank, regions_rank]
            # for idx, data in enumerate(ntk_all):
            #     if idx == 0:
            #         rankings[data[1]] = [idx]
            #     else:
            #         if data[0] == ntk_all[idx-1][0]:
            #             # same ntk as previous
            #             rankings[data[1]] = [ rankings[ntk_all[idx-1][1]][0] ]
            #         else:
            #             rankings[data[1]] = [ rankings[ntk_all[idx-1][1]][0] + 1 ]
                        
            GNN_proxy_all = sorted(GNN_proxy_all, key=lambda tup: round_to(tup[0], precision), reverse=False)  # ascending: we want to prune op to increase lr, i.e. to make lr < lr_2
            
            for idx, data in enumerate(GNN_proxy_all):
                if idx == 0:
                    rankings[data[1]] = [idx]
                else:
                    if data[0] == GNN_proxy_all[idx-1][0]:
                        # same ntk as previous
                        rankings[data[1]] = [ rankings[GNN_proxy_all[idx-1][1]][0] ]
                    else:
                        rankings[data[1]] = [ rankings[GNN_proxy_all[idx-1][1]][0] + 1 ]
            
            
            # regions_all = sorted(regions_all, key=lambda tup: round_to(tup[0], precision), reverse=False)  # ascending: we want to prune op to increase lr, i.e. to make lr < lr_2
            # # print("#Regions:", regions_all)
            # for idx, data in enumerate(regions_all):
            #     if idx == 0:
            #         rankings[data[1]].append(idx)
            #     else:
            #         if data[0] == regions_all[idx-1][0]:
            #             # same #Regions as previous
            #             rankings[data[1]].append(rankings[regions_all[idx-1][1]][1])
            #         else:
            #             rankings[data[1]].append(rankings[regions_all[idx-1][1]][1]+1)
            rankings_list = [[k, v] for k, v in rankings.items()]  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
            # ascending by sum of two rankings
            rankings_sum = sorted(rankings_list, key=lambda tup: tup[0], reverse=False)  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]
            choices = [item[0] for item in rankings_sum[:-num_per_group]]
            # print("Final Ranking:", rankings_sum)
            # print("Pruning Choices:", choices)
            for (cell_idx, edge_idx, op_idx) in choices:
                arch_parameters[cell_idx].data[edge_idx, op_idx] = -INF
            # reinit
            ntk_all = []  # (ntk, (edge_idx, op_idx))
            GNN_proxy_all = []
            regions_all = []  # (regions, (edge_idx, op_idx))
            choice2regions = {}  # (edge_idx, op_idx): regions

    return arch_parameters



def is_single_path(network):
    arch_parameters = network.get_alphas()
    edge_active = torch.cat([(nn.functional.softmax(alpha, 1) > 0.01).float().sum(1) for alpha in arch_parameters], dim=0)
    for edge in edge_active:
        assert edge > 0
        if edge > 1:
            return False
    return True


def main(xargs):
    PID = os.getpid()
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepare_seed(xargs.rand_seed)

    if xargs.timestamp == 'none':
        xargs.timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))

    train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)

    ##### config & logging #####
    config = edict()
    config.class_num = class_num
    config.xshape = xshape
    config.batch_size = xargs.batch_size
    xargs.save_dir = xargs.save_dir + \
        "/repeat%d-prunNum%d-prec%d-%s-batch%d"%(
                xargs.repeat, xargs.prune_number, xargs.precision, xargs.init, config["batch_size"]) + \
        "/{:}/seed{:}".format(xargs.timestamp, xargs.rand_seed)
    config.save_dir = xargs.save_dir
    logger = prepare_logger(xargs)
    ###############

    # if xargs.dataset != 'imagenet-1k':
    #     search_loader, train_loader, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/', config.batch_size, xargs.workers)
    # else:
    #     train_loader = torch.utils.data.DataLoader(train_data, batch_size=xargs.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    # logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(train_loader), config.batch_size))
    logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

    # logger.log('||||||| {:10s} ||||||| {:}'.format("rank weights (ntk, lr, flops, latency)", [(1-xargs.flops_weight-xargs.latency_weight)/2, (1-xargs.flops_weight-xargs.latency_weight)/2, xargs.flops_weight, xargs.latency_weight]))

    search_space = get_search_spaces('cell', xargs.search_space_name)
    if xargs.search_space_name == 'nas-bench-201':
        model_config = edict({'name': 'DARTS-V1',
                              'C': 3, 'N': 1, 'depth': -1, 'use_stem': True,
                              'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                              'space': search_space,
                              'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                             })
        model_config_thin = edict({'name': 'DARTS-V1',
                                   'C': 1, 'N': 1, 'depth': 1, 'use_stem': False,
                                   'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                   'space': search_space,
                                   'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                                  })
    elif xargs.search_space_name == 'darts':
        model_config = edict({'name': 'DARTS-V1',
                              'C': 1, 'N': 1, 'depth': 2, 'use_stem': True, 'stem_multiplier': 1,
                              'num_classes': class_num,
                              'space': search_space,
                              'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                              'super_type': xargs.super_type,
                              'steps': 4,
                              'multiplier': 4,
                             })
        model_config_thin = edict({'name': 'DARTS-V1',
                                   'C': 1, 'N': 1, 'depth': 2, 'use_stem': False, 'stem_multiplier': 1,
                                   'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                   'space': search_space,
                                   'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                                   'super_type': xargs.super_type,
                                   'steps': 4,
                                   'multiplier': 4,
                                  })
    ## my test space                              
    # elif xargs.search_space_name == 'Ye':
    #     model_config = edict({'name': 'DARTS-V1',
    #                           'C': 3, 'N': 1, 'depth': -1, 'use_stem': True,
    #                           'max_nodes': xargs.max_nodes, 'num_classes': class_num,
    #                           'space': search_space,
    #                           'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
    #                          })
    #     model_config_thin = edict({'name': 'DARTS-V1',
    #                                'C': 1, 'N': 1, 'depth': 1, 'use_stem': False,
    #                                'max_nodes': xargs.max_nodes, 'num_classes': class_num,
    #                                'space': search_space,
    #                                'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
    #                               })
    
    network = get_cell_based_tiny_net(model_config)
    logger.log('model-config : {:}'.format(model_config))
    arch_parameters = [alpha.detach().clone() for alpha in network.get_alphas()]
    for alpha in arch_parameters:
        alpha[:, :] = 0

    # TODO Linear_Region_Collector
    # lrc_model = Linear_Region_Collector(input_size=(1000, 1, 3, 3), sample_batch=3, dataset=xargs.dataset, data_path=xargs.data_path, seed=xargs.rand_seed)

    # ### all params trainable (except train_bn) #########################
    flop, param = get_model_infos(network, xshape)
    logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
    logger.log('search-space [{:} ops] : {:}'.format(len(search_space), search_space))
    if xargs.arch_nas_dataset is None or xargs.search_space_name == 'darts'or xargs.search_space_name == 'Ye':
        api = None
    else:
        api = API(xargs.arch_nas_dataset)
    logger.log('{:} create API = {:} done'.format(time_string(), api))

    network = network.cuda()

    genotypes = {}; genotypes['arch'] = {-1: network.genotype()}

    arch_parameters_history = []
    arch_parameters_history_npy = []
    start_time = time.time()
    epoch = -1

    # if args.resume_search:
    #     arch_parameters = torch.from_numpy(np.load("arch_parameters_history_imagenet.npy")[-1]).cuda() 

    embeddings = ops_embedding(xargs)

    for alpha in arch_parameters:
        alpha[:, 0] = -INF
    arch_parameters_history.append([alpha.detach().clone() for alpha in arch_parameters])
    arch_parameters_history_npy.append([alpha.detach().clone().cpu().numpy() for alpha in arch_parameters])
    np.save(os.path.join(xargs.save_dir, "arch_parameters_history.npy"), arch_parameters_history_npy)
    while not is_single_path(network):
        epoch += 1
        torch.cuda.empty_cache()
        print("<< ============== JOB (PID = %d) %s ============== >>"%(PID, '/'.join(xargs.save_dir.split("/")[-6:])))

        arch_parameters, op_pruned = prune_func_rank(xargs, arch_parameters, model_config, model_config_thin, train_loader = None, lrc_model = None, search_space = search_space,
                                                     precision=xargs.precision,
                                                     prune_number=xargs.prune_number,
                                                     embeddings = embeddings)
        # rebuild supernet
        network = get_cell_based_tiny_net(model_config)
        network = network.cuda()
        network.set_alphas(arch_parameters)
        
        # torchsummary(network, (3, 32, 32))

        arch_parameters_history.append([alpha.detach().clone() for alpha in arch_parameters])
        arch_parameters_history_npy.append([alpha.detach().clone().cpu().numpy() for alpha in arch_parameters])
        np.save(os.path.join(xargs.save_dir, "arch_parameters_history.npy"), arch_parameters_history_npy)
        genotypes['arch'][epoch] = network.genotype()

        logger.log('operators remaining (1s) and prunned (0s)\n{:}'.format('\n'.join([str((alpha > -INF).int()) for alpha in network.get_alphas()])))
        # logger.log("Latency estimation for MCU for current iteration is : {:.2f}".format(get_estimated_mcu_latency(network, xargs)))
    if xargs.search_space_name == 'darts':
        print("===>>> Prune Edge Groups...")
        arch_parameters = prune_func_rank_group(xargs, arch_parameters, model_config, model_config_thin, train_loader = None, lrc_model = None, search_space = search_space,
                                                edge_groups=[(0, 2), (2, 5), (5, 9), (9, 14)], num_per_group=2,
                                                precision=xargs.precision,
                                                embeddings = embeddings
                                                )
        network = get_cell_based_tiny_net(model_config)
        network = network.cuda()
        network.set_alphas(arch_parameters)
        arch_parameters_history.append([alpha.detach().clone() for alpha in arch_parameters])
        arch_parameters_history_npy.append([alpha.detach().clone().cpu().numpy() for alpha in arch_parameters])
        np.save(os.path.join(xargs.save_dir, "arch_parameters_history.npy"), arch_parameters_history_npy)

    logger.log('<<<--->>> End: {:}'.format(network.genotype()))
    logger.log('operators remaining (1s) and prunned (0s)\n{:}'.format('\n'.join([str((alpha > -INF).int()) for alpha in network.get_alphas()])))

    end_time = time.time()
    logger.log('\n' + '-'*100)
    logger.log("Time spent: %d s"%(end_time - start_time))
    # check the performance from the architecture dataset
    if api is not None:
        logger.log('{:}'.format(api.query_by_arch(genotypes['arch'][epoch])))

    # logger.log("Flops of searched model: \n{:}".format(get_model_infos(network, xshape)))
    # final_latency = get_estimated_mcu_latency(network, xargs)
    # logger.log("Latency estimation for MCU: {:.2f}".format(final_latency))
    # logger.log('||||||| {:10s} ||||||| {:}'.format("rank weights (ntk, lr, flops, latency)", [(1-xargs.flops_weight-xargs.latency_weight)/2, (1-xargs.flops_weight-xargs.latency_weight)/2, xargs.flops_weight, xargs.latency_weight]))
    logger.close()
    # torch.save(network, config.save_dir + 'saved_model.pkl')
    # pytorch2onnx(config.save_dir + 'saved_model.pkl', config.save_dir + 'saved_onnx_model.pkl', [3,32,32])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("TENAS")
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'ImageNet16-120', 'imagenet-1k'], help='Choose between cifar10/100/ImageNet16-120/imagenet-1k')
    parser.add_argument('--search_space_name', type=str, default='nas-bench-201',  help='space of operator candidates: nas-bench-201 or darts.')
    parser.add_argument('--max_nodes', type=int, help='The maximum number of nodes.')
    parser.add_argument('--track_running_stats', type=int, choices=[0, 1], help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for ntk')
    parser.add_argument('--save_dir', type=str, help='Folder to save checkpoints and log.')
    parser.add_argument('--arch_nas_dataset', type=str, help='The path to load the nas-bench-201 architecture dataset (tiny-nas-benchmark).')
    parser.add_argument('--rand_seed', type=int, help='manual seed')
    parser.add_argument('--precision', type=int, default=3, help='precision for % of changes of cond(NTK) and #Regions')
    parser.add_argument('--prune_number', type=int, default=1, help='number of operator to prune on each edge per round')
    parser.add_argument('--repeat', type=int, default=3, help='repeat calculation of NTK and Regions')
    parser.add_argument('--timestamp', default='none', type=str, help='timestamp for logging naming')
    parser.add_argument('--init', default='kaiming_uniform', help='use gaussian init')
    parser.add_argument('--super_type', type=str, default='basic',  help='type of supernet: basic or nasnet-super')
    parser.add_argument('--flops_weight', type=float, default=0, help='weight of flops in the ranking system, range from 0 to 1')
    parser.add_argument('--latency_weight', type=float, default=0, help='weight of latency in the ranking system, range from 0 to 1')
    parser.add_argument('--resume_search', action=argparse.BooleanOptionalAction)
    parser.add_argument('--embedding_model', type=str, default='all-mpnet-base-v2', help='sentence transformer model to use')
    parser.add_argument('--comment', type=str, default='', help='optional comment')
    # parser.add_argument('--gnn_model_path', type=str, default='new_gnn_result/GNN_Evaluation_Result_nasbench_101_dataset_sentance_transformer_all-mpnet-base-v2_short_embedding_768/checkpoint128_0.001100.0_used_201_to_finetune_and_101_to_test_adamW/checkpoint.pth.tar', help='location of the GNN model')
    parser.add_argument('--gnn_model_path', type=str, default='GNN_Evaluation_Result_nasbench_101_dataset_sentance_transformer_all-MiniLM-L6-v2_short_embedding_384/checkpoint128_0.001100.0_used_201_to_test_short_embdding_with_adamW/checkpoint.pth.tar', help='location of the GNN model')
    parser.add_argument('--embedding_size', type=int, default=384, help='embedding size')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    main(args)
