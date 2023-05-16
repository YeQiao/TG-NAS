#!/bin/bash
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0.1 --latency_weight 0
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0.2 --latency_weight 0
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0.3 --latency_weight 0
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0.4 --latency_weight 0
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0.5 --latency_weight 0
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0.6 --latency_weight 0
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0.7 --latency_weight 0
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0.8 --latency_weight 0
python prune_launch.py --space nas-bench-201 --dataset cifar10  --max_node 4 --flops_weight 0.9 --latency_weight 0