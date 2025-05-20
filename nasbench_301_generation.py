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
import os
import json
import pickle
import argparse
import re
from collections import namedtuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Define the list of possible operations.
OPS = ['input', 'none', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5',
       'dil_conv_3x3', 'dil_conv_5x5', 'avg_pool_3x3', 'max_pool_3x3', 'output']

# Define the Genotype namedtuple.
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def ops_embedding(args):
    model = SentenceTransformer(args.embedding_model)
    print('Embedding model loaded')
    sentences = [
        'Input operator',
        'A none operator that does nothing',
        'A residual connection operator that adds identity mapping to the next layer',
        "A depthwise separable convolution is applied with a kernel size of 3 by 3",
        "A depthwise separable convolution is applied with a kernel size of 5 by 5",
        "A dilated convolution is applied with a kernel size of 3 by 3",
        "A dilated convolution is applied with a kernel size of 5 by 5",
        'An average pooling operator with a kernel size 3 by 3',
        'A max pooling operator with a kernel size 3 by 3',
        'Output operator'
    ]
    embeddings = model.encode(sentences, device='cuda:0')
    return embeddings

def encode_custom(module_operations, embeddings):
    return [embeddings[OPS.index(op)] for op in module_operations]

def genotype_normal_from_config(config):
    """
    Extract the normal-cell genotype from the JSON configuration.
    For each key "NetworkSelectorDatasetInfo:darts:inputs_node_normal_X" (X is a node index),
    split the string (e.g., "0_1") to get the source nodes.
    Then, assign a sequential edge index to look up the corresponding operator from keys 
    "NetworkSelectorDatasetInfo:darts:edge_normal_<edge_index>".
    
    If any incoming edge is "skip_connect", choose that; otherwise choose the first.
    Use the first input as the chosen input.
    
    Returns a list of tuples (chosen_op, chosen_input) for each intermediate node.
    """
    genotype = []
    #find all keys for normal cell inputs.
    input_keys = [k for k in config.keys() if k.startswith("NetworkSelectorDatasetInfo:darts:inputs_node_normal_")]
    #sort keys by node index 
    input_keys = sorted(input_keys, key=lambda k: int(k.split('_')[-1]))
    edge_counter = 0
    for key in input_keys:
        src_nodes = config[key].split('_')
        incoming_ops = []
        for input_pos, _ in enumerate(src_nodes):
            edge_key = f"NetworkSelectorDatasetInfo:darts:edge_normal_{edge_counter}"
            op_edge = config.get(edge_key, "none")
            incoming_ops.append(op_edge)
            edge_counter += 1
        chosen_op = "skip_connect" if "skip_connect" in incoming_ops else incoming_ops[0] #prior skip_connect, or choose the first one
        chosen_input = int(src_nodes[0])
        genotype.append((chosen_op, chosen_input))
    return genotype


def transform_matrix_embedding(genotype_dict, embeddings):
    """
    Given a genotype dictionary with key 'normal' (a list of (op, connect) tuples),
    generate the cell’s adjacency matrix and a list of operator embeddings.
    Assumes:
      - 2 input nodes,
      - len(normal) intermediate nodes,
      - 1 output node.
    """
    normal = genotype_dict['normal']
    node_num = len(normal) + 3  # 2 inputs + intermediate nodes + 1 output.
    adj = np.zeros((node_num, node_num))
    ops = []
    # For input nodes, use the same embedding.
    ops.append(embeddings[0])
    ops.append(embeddings[0])
    for i in range(len(normal)):
        op, connect = normal[i]
        if connect == 0 or connect == 1:
            adj[connect][i + 2] = 1
        else:
            adj[connect][i + 2] = 1
        op_embedding = embeddings[OPS.index(op)]
        ops.append(op_embedding)
    # Connect each intermediate node to the output node.
    for node in range(2, node_num - 1):
        adj[node][node_num - 1] = 1
    ops.append(embeddings[-1])
    return adj, ops

def main(args):
    embeddings = ops_embedding(args)
    print("SentenceTransformer embeddings loaded.")

    data_list = []
    
    # Process JSON files directly in the base_dir
    for file in os.listdir(args.base_dir):
        file_path = os.path.join(args.base_dir, file)
        if os.path.isfile(file_path) and file.endswith(".json"):
            try:
                with open(file_path) as f:
                    entry = json.load(f)
                config_orig = entry["optimized_hyperparamater_config"]
                test_acc = entry["test_accuracy"]
                # Extract the normal-cell genotype from the JSON configuration.
                genotype_normal = genotype_normal_from_config(config_orig)
                # Ensure exactly 4 intermediate nodes (pad with ('none', 0) if needed).
                while len(genotype_normal) < 4:
                    genotype_normal.append(('none', 0))
                genotype_normal = genotype_normal[:4]
                # Build a Genotype namedtuple for the normal cell.
                genotype = Genotype(
                    normal = genotype_normal,
                    normal_concat = list(range(3, 3 + len(genotype_normal))),
                    reduce = [],
                    reduce_concat = []
                )
                # Create a dictionary with key 'normal' for transform_matrix_embedding.
                genotype_dict = {"normal": genotype.normal}
                matrix_normal, ops_normal = transform_matrix_embedding(genotype_dict, embeddings)
                data_list.append({
                    "adjacency_matrix": matrix_normal,
                    "operations": ops_normal,
                    "metrics": test_acc
                })
            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")
    
    # Process JSON files in any subdirectories of base_dir
    for subdir in os.listdir(args.base_dir):
        sub_path = os.path.join(args.base_dir, subdir)
        if os.path.isdir(sub_path):
            for file in os.listdir(sub_path):
                if file.endswith(".json"):
                    file_path = os.path.join(sub_path, file)
                    try:
                        with open(file_path) as f:
                            entry = json.load(f)
                        config_orig = entry["optimized_hyperparamater_config"]
                        test_acc = entry["test_accuracy"]
                        genotype_normal = genotype_normal_from_config(config_orig)
                        while len(genotype_normal) < 4:
                            genotype_normal.append(('none', 0))
                        genotype_normal = genotype_normal[:4]
                        genotype = Genotype(
                            normal = genotype_normal,
                            normal_concat = list(range(3, 3 + len(genotype_normal))),
                            reduce = [],
                            reduce_concat = []
                        )
                        genotype_dict = {"normal": genotype.normal}
                        matrix_normal, ops_normal = transform_matrix_embedding(genotype_dict, embeddings)
                        data_list.append({
                            "adjacency_matrix": matrix_normal,
                            "operations": ops_normal,
                            "metrics": test_acc
                        })
                    except Exception as e:
                        print(f"Skipping {file_path} due to error: {e}")

    if len(data_list) == 0:
        print("No JSON files were found!")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"nasbench_301_normal_{os.path.basename(args.model_path)}_{args.sentence_length}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(data_list, f)
        print(f"\n✅ Dataset saved: {output_file} with {len(data_list)} architectures.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str,
                        default="/shared/jingchl6/TG-NAS/home/siemsj/projects/nasbench_201_2/analysis/nb_301_v13",  # adjust this path accordingly
                        help='Base directory containing JSON files or subdirectories with JSON files')
    parser.add_argument('--output_dir', type=str, default="new_data",
                        help='Directory where the output .pkl file will be saved')
    parser.add_argument('--sentence_length', type=str, default="long",
                        help="(Used only for naming in this script)")
    parser.add_argument('--model_path', type=str,
                        default="/home/jingchl6/.local/sentencedata/fine_tuned_sentence_transformer",
                        help='Sentence transformer model path')
    parser.add_argument('--embedding_model', type=str,
                        default="/home/jingchl6/.local/sentencedata/fine_tuned_sentence_transformer",
                        help='Embedding model for ops_embedding')
    args = parser.parse_args()
    main(args)