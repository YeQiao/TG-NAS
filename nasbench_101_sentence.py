from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sentence_transformers import SentenceTransformer
from absl import app
from nasbench import api
import pickle  

NASBENCH_TFRECORD = '/shared/jingchl6/TG-NAS/nasbench_only108.tfrecord'

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

fine_tuned_path = "/home/jingchl6/.local/sentencedata/fine_tuned_sentence_transformer"
fine_tuned_model = SentenceTransformer(fine_tuned_path)

OPS =  ["input", "conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3", "output"]

def encode_ops(module_operations):
    embeddings = []
    for op in module_operations:
        emb = fine_tuned_model.encode(op)
        embeddings.append(emb.tolist())  # or simply emb if you prefer numpy arrays
    return embeddings

def main():

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
                "operations" : encode_ops(data['module_operations']),
                "metrics" : data['test_accuracy']}
        data_list.append(data_saved)

    with open("nasbench_101_dataset_sentence.pkl", 'wb') as f:
        pickle.dump(data_list, f)
    print("nasbench_101 dataset generated with sentence transformer")

if __name__ == "__main__":
    main()