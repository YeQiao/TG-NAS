from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from absl import app
from nasbench import api
import pickle
from sentence_transformers import SentenceTransformer
# import os
# import gzip
# import csv
# import random
import numpy as np
# import torch
import argparse
# Replace this string with the path to the downloaded nasbench.tfrecord before
# executing.
NASBENCH_TFRECORD = '/home/yeq6/Research_project/MicroNAS/nasbench_only108.tfrecord'

# Iterate through unique models in the dataset. Models are unqiuely identified
  # by a hash.

OPS =  ["input", "conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3", "output"]

def encode(module_operations, embeddings):
    ops = []
    for op in module_operations:
        op_list = embeddings[OPS.index(op)]
        ops.append(op_list)
        # print(op)
        # print(op_list)
        # break
    return ops

def main(args):
        
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer(args.model_path)
    print('model loaded')
    #Our sentences we like to encode
    if args.sentence_length == "long":
        sentences = ['Input operator',
            
                    """A two-dimensional convolutional operator 
                    with a kernel size of 3 by 3 is applied,
                    succeeded by a batch normalization layer,
                    and followed by a rectified linear layer""",
                    
                    """A two-dimensional convolutional operator
                    with a kernel size of 1 by 1 is applied,
                    succeeded by a batch normalization layer,
                    and followed by a rectified linear layer""",
                    
                    'A max pooling operator with a kernel size 3 by 3',

                    'Output operator'
                    ]
    elif args.sentence_length == "medium":
        sentences = ['Input',
        
                'Convolution 3 by 3 kernal, Batchnorm, ReLU',
                
                'Convolution 1 by 1 kernal, Batchnorm, ReLU',
                
                'Max pooling 3 by 3 kernal',

                'Output'
                ]
    elif args.sentence_length == "short":   
        sentences = ['Input',
        
                'Convolution 3 by 3',
                
                'Convolution 1 by 1',
                
                'Max pooling 3 by 3',

                'Output'
                ]
    else:
        raise ValueError("Invalid sentence type. Expected one of: [short, medium, long]")

    #Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences, device = 'cuda:0')
    print('embedding loaded')
    #Print the embeddings
    # for sentence, embedding in zip(sentences, embeddings):
    #     print("Sentence:", sentence)
    #     print("Embedding:", embedding)
    #     print("Embedding lenght:", len(embedding))
    #     print("finished")
    # return
    print('\nIterating over unique models in the dataset.')
    nasbench = api.NASBench(NASBENCH_TFRECORD)

    data_list = []
    for unique_hash in nasbench.hash_iterator():
        fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
        # print(fixed_metrics)
        model_spec = api.ModelSpec(
        # Adjacency matrix of the module
        matrix= fixed_metrics['module_adjacency'],
        # Operations at the vertices of the module, matches order of matrix
        ops=fixed_metrics['module_operations'])
        data = nasbench.query(model_spec)
        data_saved = {"adjacency_matrix" : data['module_adjacency'].tolist(),
                "operations" : encode(data['module_operations'], embeddings),
                "metrics" : data['test_accuracy']}
        data_list.append(data_saved)
    
    dataset_name = "new_data/nasbench_101_dataset_sentance_transformer_" + \
        str(args.model_path)[str(args.model_path).rfind('/'):] + '_' + str(args.sentence_length) + "embedding.pkl"

    with open(dataset_name, 'wb') as f:
        pickle.dump(data_list, f)
    print(dataset_name, "generated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--select', action=argparse.BooleanOptionalAction)
    parser.add_argument('--onehot', action=argparse.BooleanOptionalAction)
    parser.add_argument('--sentence_length', type=str, default="short")
    parser.add_argument('--model_path', type=str, default='all-MiniLM-L6-v2', help='sentence transformer model to use')
    parser.add_argument('--comment', type=str, default='', help='optional comment')
    args = parser.parse_args()
    main(args)

