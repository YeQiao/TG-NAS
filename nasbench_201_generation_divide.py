import pkgutil
if not hasattr(pkgutil, 'ImpImporter'):
    pkgutil.ImpImporter = pkgutil.zipimporter

import importlib.machinery
if not hasattr(importlib.machinery.FileFinder, 'find_module'):
    # Patch FileFinder to provide a find_module method that uses find_spec.
    def find_module(self, fullname):
        spec = self.find_spec(fullname)
        return spec.loader if spec is not None else None
    importlib.machinery.FileFinder.find_module = find_module
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from sklearn import preprocessing
from sentence_transformers import SentenceTransformer

OPS = ['input', 'none', 'skip_connect', 'nor_conv_3x3', 'nor_conv_1x1',  'avg_pool_3x3', 'output']

def transform_matrix(genotype):
    normal = genotype
    node_num = len(normal)+2
    adj = np.zeros((node_num, node_num))
    ops = np.zeros((node_num, len(OPS)))
    for i in range(len(normal)):
        op, connect = normal[i]
        if connect == 0 or connect==1:
            adj[connect][i+1] = 1
        else:
            adj[(connect-2)*2+2][i+1] = 1
            adj[(connect-2)*2+3][i+1] = 1
        ops[i+1][OPS.index(op)] = 1
    adj[4:-1, -1] = 1
    ops[0:1, 0] = 1
    ops[-1][-1] = 1
    return adj, ops

def transform_matrix_selected(genotype, isseleted):
    normal = genotype
    node_num = len(normal)+2
    adj = np.zeros((node_num, node_num))
    ops = np.zeros((node_num, len(OPS)))
    for i in range(len(normal)):
        op, connect = normal[i]
        if connect == 0 or connect==1:
            adj[connect][i+1] = 1
        else:
            adj[(connect-2)*2+2][i+1] = 1
            adj[(connect-2)*2+3][i+1] = 1
        ops[i+1][OPS.index(op)] = 1
    adj[4:-1, -1] = 1
    ops[0:1, 0] = 1
    ops[-1][-1] = 1
    if isseleted:
        ops = ops[:,[0,3,4,5,6]]
    return adj, ops

def transform_matrix_embedding(genotype, embeddings):
    normal = genotype
    node_num = len(normal)+2
    adj = np.zeros((node_num, node_num))
    # ops = np.zeros((node_num, len(OPS)))
    ops = []
    ops.append(embeddings[0])
    for i in range(len(normal)):
        op, connect = normal[i]
        if connect == 0 or connect==1:
            adj[connect][i+1] = 1
        else:
            adj[(connect-2)*2+2][i+1] = 1
            adj[(connect-2)*2+3][i+1] = 1
        # ops[i+1][OPS.index(op)] = 1
        op_embedding = embeddings[OPS.index(op)]
        ops.append(op_embedding)
    adj[4:-1, -1] = 1
    ops.append(embeddings[-1])
    # ops[0:1, 0] = 1
    # ops[-1][-1] = 1
    return adj, ops

def main(args):
        
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # model = SentenceTransformer('all-mpnet-base-v2')
    model = SentenceTransformer(args.model_path)
    print('model loaded')
    # Our sentences we like to encode
    sentences = ['Input operator',
                
                'A none operator that does nothing',

                'A residual connection operator that adding identity mapping to the next layer',
                
                """A two-dimensional convolutional operator 
                with a kernel size of 3 by 3 is applied,
                succeeded by a batch normalization layer,
                and followed by a rectified linear layer""",
                
                """A two-dimensional convolutional operator
                with a kernel size of 1 by 1 is applied,
                succeeded by a batch normalization layer,
                and followed by a rectified linear layer""",
                
                'A average pooling operator with a kernel size 3 by 3',

                'Output operator'
                ]
    if args.sentence_length == "long":
        sentences = ['Input operator',
                
                'A none operator that does nothing',

                'A residual connection operator that adding identity mapping to the next layer',
                
                """A two-dimensional convolutional operator 
                with a kernel size of 3 by 3 is applied,
                succeeded by a batch normalization layer,
                and followed by a rectified linear layer""",
                
                """A two-dimensional convolutional operator
                with a kernel size of 1 by 1 is applied,
                succeeded by a batch normalization layer,
                and followed by a rectified linear layer""",
                
                'A average pooling operator with a kernel size 3 by 3',

                'Output operator'
                ]
  
    else:
        raise ValueError("Invalid sentence type. Expected one of: [short, medium, long]")

    embeddings = model.encode(sentences, device = 'cuda:0') 
    print('embedding loaded')

    print('\nIterating over unique models in the dataset.')

    # save the acc in correct order for later uasge
    acc_cifar10, acc_cifar100, acc_imagenet = [], [], []
    with open('all_model.txt') as f:
        for line in f:
            line = line.rstrip()
            # print(line[0:7])
            if line[0:22] == 'cifar10        train :':
                acc = float(line[-7:-2])
                acc_cifar10.append(acc)
            elif line[0:22] == 'cifar100       train :':
                acc = float(line[-7:-2])
                acc_cifar100.append(acc)
            elif line[0:22] == 'ImageNet16-120 train :':
                acc = float(line[-7:-2])
                acc_imagenet.append(acc)

    # construct the dataset

    i = -1
    data_all_list_cifar10 = []
    data_all_list_cifar100 = []
    data_all_list_imagenet = []
    data_selected_list = []
    with open('all_model.txt') as f:
        for line in f:
            if line.strip() == "":
                i+=1
            line = line.rstrip()
            if line[0:7] == 'arch : ':
                select = True
                geno = line[7:]
                geno_list = geno.split('+')
                genotype = []
                for a in geno_list:
                    a = a[1:-1]
                    b = a.split('|')
                    for c in b:
                        ops = c[0:-2]
                        connect = c[-1:]
                        genotype.append((ops, int(connect)))
                        if ops == 'skip_connect' or ops == 'none':
                            select = False
                # print(genotype)
                # print(i)
                if select and args.select:
                    # adj, ops = transform_matrix_selected(genotype, select)
                    # data_point = {'adjacency_matrix': adj, "operations": ops,
                    #             "metrics": acc_all[i]}
                    # data_selected_list.append(data_point)
                    return
                elif args.onehot: 
                    # adj, ops = transform_matrix(genotype)
                    # data_point = {'adjacency_matrix': adj, "operations": ops,
                    #                 "metrics": acc_all[i]}
                    # data_all_list.append(data_point)
                    return
                else:
                    adj, ops = transform_matrix_embedding(genotype, embeddings)
                    data_point_cifar10 = {'adjacency_matrix': adj, "operations": ops,
                                    "metrics": acc_cifar10[i]}
                    data_all_list_cifar10.append(data_point_cifar10)
                    data_point_cifar100 = {'adjacency_matrix': adj, "operations": ops,
                                    "metrics": acc_cifar100[i]}
                    data_all_list_cifar100.append(data_point_cifar100)
                    data_point_imagenet = {'adjacency_matrix': adj, "operations": ops,
                                    "metrics": acc_imagenet[i]}
                    data_all_list_imagenet.append(data_point_imagenet)

    # save the samples as dataset
    if args.onehot:
        embedding_name = 'onehot_'
    else:
        embedding_name = 'sentence_transformer_'

    # save the selected samples if needed
    if args.select:
        with open("nasbench_201_dataset_selected_" + embedding_name  + args.model_path + '_' + args.comment + ".pkl", 'wb') as f:
            pickle.dump(data_selected_list, f)
    with open("/home/jingchl6/.local/TG-NAS/nasbench_201_dataset_cifar10" + embedding_name + args.comment + ".pkl", 'wb') as f:
        pickle.dump(data_all_list_cifar10, f)
    with open("/home/jingchl6/.local/TG-NAS/nasbench_201_dataset_cifar100" + embedding_name + args.comment + ".pkl", 'wb') as f:
        pickle.dump(data_all_list_cifar100, f)
    with open("/home/jingchl6/.local/TG-NAS/nasbench_201_dataset_Imagenet" + embedding_name + args.comment + ".pkl", 'wb') as f:
        pickle.dump(data_all_list_imagenet, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--select', action=argparse.BooleanOptionalAction)
    parser.add_argument('--onehot', action=argparse.BooleanOptionalAction)
    parser.add_argument('--sentence_length', type=str, default="long")
    # parser.add_argument('--model_path', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='sentence transformer model to use')
    parser.add_argument('--model_path', type=str, default="/home/jingchl6/.local/sentencedata/fine_tuned_sentence_transformer", help='sentence transformer model to use')
    parser.add_argument('--comment', type=str, default='', help='optional comment')
    args = parser.parse_args()
    main(args)

