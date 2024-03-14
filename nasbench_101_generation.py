
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from nasbench import api
import pickle  
# Replace this string with the path to the downloaded nasbench.tfrecord before
# executing.
NASBENCH_TFRECORD = '/home/yeq6/MicroNAS/nasbench_only108.tfrecord'

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
# Iterate through unique models in the dataset. Models are unqiuely identified
  # by a hash.

OPS =  ["input", "conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3", "output"]

def onehot(module_operations):
    ops = []
    for op in module_operations:
        op_list = len(OPS) * [0]
        op_list[OPS.index(op)] = 1
        ops.append(op_list)
    return ops

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
                "operations" : onehot(data['module_operations']),
                "metrics" : data['test_accuracy']}
        data_list.append(data_saved)

    with open("nasbench_101_dataset.pkl", 'wb') as f:
        pickle.dump(data_list, f)
    print("nasbench_101 dataset generated")

if __name__ == "__main__":
    main()

