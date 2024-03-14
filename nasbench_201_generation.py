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
    # sentences = ['Input',
                
    #             'None',

    #             'Residual connection',
                
    #             'Convolution 3x3',
                
    #             'Convolution 1x1',
                
    #             'Average pooling 3x3',

    #             'Output operator'
    #             ]
    # Sentences are encoded by calling model.encode()
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
    elif args.sentence_length == "medium":
        sentences = ['Input',
                     
                'Doing nothing',
                
                'Identity mapping to the next layer',
        
                'Convolution 3 by 3 kernel, Batchnorm, ReLU',
                
                'Convolution 1 by 1 kernel, Batchnorm, ReLU',
                
                'Max pooling 3 by 3 kernel',

                'Output'
                ]
    elif args.sentence_length == "short":   
        sentences = ['Input',
                
                'None',

                'Residual connection',
                
                'Convolution 3 by 3',
                
                'Convolution 1 by 1',
                
                'Average pooling 3 by 3',

                'Output'
                ]
  
    else:
        raise ValueError("Invalid sentence type. Expected one of: [short, medium, long]")

    embeddings = model.encode(sentences, device = 'cuda:0')
    print('embedding loaded')

    print('\nIterating over unique models in the dataset.')

    # save the acc in correct order for later uasge
    acc_all = []
    with open('all_model.txt') as f:
        for line in f:
            line = line.rstrip()
            # print(line[0:7])
            if line[0:22] == 'cifar10        train :':
                acc = float(line[-7:-2])
                acc_all.append(acc)

    # construct the dataset
    i = -1
    data_all_list = []
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
                    adj, ops = transform_matrix_selected(genotype, select)
                    data_point = {'adjacency_matrix': adj, "operations": ops,
                                "metrics": acc_all[i]}
                    data_selected_list.append(data_point)
                    return
                elif args.onehot: 
                    adj, ops = transform_matrix(genotype)
                    data_point = {'adjacency_matrix': adj, "operations": ops,
                                    "metrics": acc_all[i]}
                    data_all_list.append(data_point)
                    return
                else:
                    adj, ops = transform_matrix_embedding(genotype, embeddings)
                    data_point = {'adjacency_matrix': adj, "operations": ops,
                                    "metrics": acc_all[i]}
                    data_all_list.append(data_point)

    # save the samples as dataset
    if args.onehot:
        embedding_name = 'onehot_'
    else:
        embedding_name = 'sentence_transformer_'

    # save the selected samples if needed
    if args.select:
        with open("nasbench_201_dataset_selected_" + embedding_name  + args.model_path + '_' + args.comment + ".pkl", 'wb') as f:
            pickle.dump(data_selected_list, f)
    with open("new_data/nasbench_201_dataset_all_" + embedding_name  + args.model_path + '_' + args.comment + ".pkl", 'wb') as f:
        pickle.dump(data_all_list, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--select', action=argparse.BooleanOptionalAction)
    parser.add_argument('--onehot', action=argparse.BooleanOptionalAction)
    parser.add_argument('--sentence_length', type=str, default="short")
    parser.add_argument('--model_path', type=str, default='models/my-64_dim-model', help='sentence transformer model to use')
    parser.add_argument('--comment', type=str, default='', help='optional comment')
    args = parser.parse_args()
    main(args)

