from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    # 'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)

AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

flops_07_darts_cifar10 = Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('max_pool_3x3', 1)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('avg_pool_3x3', 1), ('skip_connect', 3), ('max_pool_3x3', 1), ('skip_connect', 3)], reduce_concat=[2, 3, 4, 5])
Latency_07_darts_cifar10 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('max_pool_3x3', 2)], reduce_concat=[2, 3, 4, 5])

My_TENAS_cifar10 = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 1), ('sep_conv_3x3', 3), ('dil_conv_3x3', 1), ('sep_conv_5x5', 2)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('max_pool_3x3', 4)], reduce_concat=[2, 3, 4, 5])
DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

TENAS_cifar10 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 4)], reduce_concat=[2, 3, 4, 5])
TENAS_imagenet = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0), ('dil_conv_3x3', 4)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])

TEGNAS_RL = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
TEGNAS_Evolution = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('sep_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])
TEGNAS_ProbNAS = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 3)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 4)], reduce_concat=[2, 3, 4, 5])

TE_DAG_imagenet_0 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 2), ('sep_conv_3x3', 4)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_1 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 1), ('sep_conv_3x3', 3), ('dil_conv_3x3', 0), ('sep_conv_3x3', 3)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_2 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 4)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_3 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 3)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_4 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 3), ('sep_conv_3x3', 4)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_5 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_6 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('sep_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_7 = Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 4)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_8 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_9 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 2), ('avg_pool_3x3', 3), ('sep_conv_3x3', 3), ('sep_conv_3x3', 4)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 4)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_10 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 3)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_11 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 3), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_12 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_13 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 2), ('dil_conv_3x3', 4)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('dil_conv_5x5', 0), ('sep_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_14 = Genotype(normal=[('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_15 = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 4)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_16 = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2)], reduce_concat=[2, 3, 4, 5])
TE_DAG_imagenet_17 = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 3)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])

WOT_imagenet_4 = Genotype(normal=[['sep_conv_5x5', 1], ['sep_conv_3x3', 0], ['max_pool_3x3', 1], ['max_pool_3x3', 1], ['none', 0], ['sep_conv_5x5', 1], ['dil_conv_5x5', 2], ['none', 2]], normal_concat=[2, 3, 4, 5], reduce=[['none', 0], ['sep_conv_5x5', 0], ['sep_conv_5x5', 0], ['max_pool_3x3', 1], ['none', 3], ['max_pool_3x3', 0], ['sep_conv_3x3', 0], ['dil_conv_5x5', 3]], reduce_concat=[2, 3, 4, 5])
WOT_DAG_imagenet_4 = Genotype(normal=[['none', 0], ['sep_conv_5x5', 1], ['sep_conv_5x5', 2], ['sep_conv_5x5', 2], ['dil_conv_3x3', 1], ['dil_conv_5x5', 3], ['max_pool_3x3', 4], ['max_pool_3x3', 0]], normal_concat=[2, 3, 4, 5], reduce=[['sep_conv_3x3', 1], ['dil_conv_3x3', 0], ['max_pool_3x3', 0], ['sep_conv_5x5', 0], ['sep_conv_3x3', 0], ['dil_conv_3x3', 3], ['skip_connect', 4], ['skip_connect', 0]], reduce_concat=[2, 3, 4, 5])
WOT_DAG_imagenet_7 = Genotype(normal=[['sep_conv_3x3', 0], ['sep_conv_3x3', 1], ['dil_conv_5x5', 0], ['dil_conv_5x5', 2], ['dil_conv_3x3', 3], ['avg_pool_3x3', 1], ['dil_conv_3x3', 2], ['avg_pool_3x3', 1]], normal_concat=[2, 3, 4, 5], reduce=[['max_pool_3x3', 0], ['none', 0], ['dil_conv_5x5', 0], ['none', 1], ['sep_conv_3x3', 1], ['sep_conv_3x3', 0], ['none', 1], ['dil_conv_5x5', 3]], reduce_concat=[2, 3, 4, 5])
WOT_DAG_imagenet_8 = Genotype(normal=[['sep_conv_3x3', 0], ['sep_conv_5x5', 1], ['dil_conv_5x5', 0], ['none', 0], ['avg_pool_3x3', 1], ['sep_conv_3x3', 0], ['sep_conv_5x5', 1], ['dil_conv_5x5', 1]], normal_concat=[2, 3, 4, 5], reduce=[['dil_conv_3x3', 1], ['skip_connect', 0], ['sep_conv_3x3', 1], ['avg_pool_3x3', 2], ['max_pool_3x3', 3], ['dil_conv_3x3', 2], ['dil_conv_3x3', 2], ['max_pool_3x3', 0]], reduce_concat=[2, 3, 4, 5])
