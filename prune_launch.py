import os
import time
import argparse

# TODO please configure TORCH_HOME and data_paths before running
# TORCH_HOME = "/ssd1/chenwy"  # Path that contains the nas-bench-201 database. If you only want to run on NASNET (i.e. DARTS) search space, then just leave it empty
TORCH_HOME = ""
data_paths = {
    "cifar10": "./data.cifar10",
    "cifar100": "/ssd1/cifar.python",
    "ImageNet16-120": "/ssd1/ImageNet16",
    "imagenet-1k": "/home/data/share/Dataset/imagenet",
}


parser = argparse.ArgumentParser("TENAS_launch")
parser.add_argument('--gpu', default=0, type=int, help='use gpu with cuda number')
parser.add_argument('--space', default='nas-bench-201', type=str, choices=['nas-bench-201', 'darts', 'Ye'], help='which nas search space to use')
parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar10', 'cifar100', 'ImageNet16-120', 'imagenet-1k'], help='Choose between cifar10/100/ImageNet16-120/imagenet-1k')
parser.add_argument('--seed', default=0, type=int, help='manual seed')
parser.add_argument('--max_node', default=4, type=int, help='The maximum number of nodes.')
parser.add_argument('--flops_weight', type=float, default=0, help='weight of flops in the ranking system, range from 0 to 1')
parser.add_argument('--latency_weight', type=float, default=0, help='weight of latency in the ranking system, range from 0 to 1')
# parser.add_argument('--resume_search', action=argparse.BooleanOptionalAction)
parser.add_argument('--embedding_model', type=str, default='all-mpnet-base-v2', help='sentence transformer model to use')
parser.add_argument('--embedding_size', type=int, default=384, help='embedding size')
args = parser.parse_args()


##### Basic Settings
precision = 5
# init = 'normal'
# init = 'kaiming_uniform'
init = 'kaiming_normal'


if args.space == "nas-bench-201":
    prune_number = 1
    batch_size = 128
    space = "nas-bench-201"  # different spaces of operator candidates, not structure of supernet
    super_type = "basic"  # type of supernet structure
elif args.space == "Ye":
    prune_number = 1
    batch_size = 128
    space = "Ye"  # different spaces of operator candidates, not structure of supernet
    super_type = "basic"  # type of supernet structure
elif args.space == "darts":
    space = "darts"
    super_type = "nasnet-super"
    if args.dataset == "cifar10":
        prune_number = 3
        batch_size = 14
        # batch_size = 6
    elif args.dataset == "imagenet-1k":
        prune_number = 2
        batch_size = 24


timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))

# --embedding_size {embedding_size}\

core_cmd = "CUDA_VISIBLE_DEVICES={gpuid} OMP_NUM_THREADS=4 python ./prune_tenas_gnn_darts.py \
--save_dir {save_dir} --max_nodes {max_nodes} \
--dataset {dataset} \
--data_path {data_path} \
--search_space_name {space} \
--super_type {super_type} \
--arch_nas_dataset {TORCH_HOME}NAS-Bench-201-v1_0-e61699.pth \
--track_running_stats 1 \
--workers 0 --rand_seed {seed} \
--timestamp {timestamp} \
--precision {precision} \
--init {init} \
--repeat 3 \
--batch_size {batch_size} \
--prune_number {prune_number} \
--flops_weight {flops_weight} \
--latency_weight {latency_weight} \
--resume_search \
--embedding_model {embedding_model}\
".format(
    gpuid=args.gpu,
    save_dir="./output/prune-{space}/{dataset}".format(space=space, dataset=args.dataset),
    max_nodes=args.max_node,
    data_path=data_paths[args.dataset],
    dataset=args.dataset,
    TORCH_HOME=TORCH_HOME,
    space=space,
    super_type=super_type,
    seed=args.seed,
    timestamp=timestamp,
    precision=precision,
    init=init,
    batch_size=batch_size,
    prune_number=prune_number,
    flops_weight=args.flops_weight,
    latency_weight=args.latency_weight,
    embedding_model = args.embedding_model,
    # embedding_size = args.embedding_size,
)

os.system(core_cmd)
