#!/bin/bash
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0 --latency_weight 0.1
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0 --latency_weight 0.2
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0 --latency_weight 0.3
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0 --latency_weight 0.4
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0 --latency_weight 0.5
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0 --latency_weight 0.6
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0 --latency_weight 0.7
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0 --latency_weight 0.8
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0 --latency_weight 0.9

python prune_launch.py --space darts --dataset imagenet-1k  --max_node 4 --flops_weight 0.7 --latency_weight 0


python DARTS_evaluation/train_imagenet_dist.py --arch flops_07_darts_cifar10 --save flops_07_darts_cifar10 --gpu 0