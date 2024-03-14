#!/bin/bash


python nasbench_101_generation_1.py --model_path all-mpnet-base-v2 --sentence_length short --comment short_embedding
python nasbench_101_generation_1.py --model_path all-mpnet-base-v2 --sentence_length medium --comment medium_embedding
python nasbench_101_generation_1.py --model_path all-mpnet-base-v2 --sentence_length long --comment long_embedding

python nasbench_101_generation_1.py --model_path all-MiniLM-L6-v2 --sentence_length short --comment short_embedding
python nasbench_101_generation_1.py --model_path all-MiniLM-L6-v2 --sentence_length medium --comment medium_embedding
python nasbench_101_generation_1.py --model_path all-MiniLM-L6-v2 --sentence_length long --comment long_embedding


python nasbench_201_generation.py --model_path all-mpnet-base-v2 --sentence_length short --comment short_embedding
python nasbench_201_generation.py --model_path all-mpnet-base-v2 --sentence_length medium --comment medium_embedding
python nasbench_201_generation.py --model_path all-mpnet-base-v2 --sentence_length long --comment long_embedding

python nasbench_201_generation.py --model_path all-MiniLM-L6-v2 --sentence_length short --comment short_embedding
python nasbench_201_generation.py --model_path all-MiniLM-L6-v2 --sentence_length medium --comment medium_embedding
python nasbench_201_generation.py --model_path all-MiniLM-L6-v2 --sentence_length long --comment long_embedding