#!/bin/bash

for i in {1..9}
do
x=$(( $i*10 ))
echo "The training data ratio is $x" 
python gcn_test.py --train_split $x > gcn_proxy_result_$x.txt
done
echo All done


python gcn_test.py --train_split 100  --embedding_size 384 \
--data_path nasbench_101_dataset_sentance_transformer_all-MiniLM-L6-v2.pkl \
--data_path_201 nasbench_201_dataset_all_sentence_transformer_all-MiniLM-L6-v2.pkl \
--use_201_to_test


python gcn_test.py --train_split 100  --embedding_size 64 \
--data_path nasbench_101_dataset_sentance_transformer_64.pkl \
--data_path_201 nasbench_201_dataset_all_sentence_transformer_64.pkl \
--use_201_to_test
--comment used_201_to_test

python gcn_test.py --train_split 100  --embedding_size 768 \
--data_path nasbench_101_dataset_sentance_transformer_768.pkl \
--data_path_201 nasbench_201_dataset_all_sentence_transformer.pkl \
--use_201_to_test
--comment used_201_to_test_long_embdding_weight_decay_10e-5

python gcn_test.py --train_split 100  --embedding_size 384 \
--data_path nasbench_101_dataset_sentance_transformer_all-MiniLM-L6-v2_short_embedding.pkl \
--data_path_201 nasbench_201_dataset_all_sentence_transformer_all-MiniLM-L6-v2_short_embedding.pkl \
--device cuda:0 \
--use_201_to_test \
--comment used_201_to_test_short_embdding


python gcn_test.py --train_split 100  --embedding_size 384 \
--data_path nasbench_101_dataset_sentance_transformer_all-mpnet-base-v2_short_embdding.pkl \
--data_path_201 nasbench_201_dataset_all_sentence_transformer_all-mpnet-base-v2_short_embedding.pkl \
--device cuda:0 \
--use_201_to_test \
--comment used_201_to_test_short_embdding


python gcn_test.py --train_split 100  --embedding_size 768 \
--data_path nasbench_101_dataset_sentance_transformer_768.pkl \
--data_path_201 nasbench_201_dataset_all_1.pkl \
--device cuda:1 \
--use_201_to_test \
--comment used_201_to_test_long_embdding_weight_decay


/home/yeq6/Research_project/MicroNAS/GNN_Evaluation_Result_nasbench_101_dataset_sentance_transformer_all-MiniLM-L6-v2_short_embedding_384/checkpoint128_0.001100.0_used_201_to_test_short_embdding/checkpoint.pth.tar
python gcn_verify.py --data_path_201 nasbench_201_dataset_all_sentence_transformer_all-MiniLM-L6-v2_short_embedding.pkl \
--test_201 --embedding_size 384 \
--model_path GNN_Evaluation_Result_nasbench_101_dataset_sentance_transformer_all-MiniLM-L6-v2_short_embedding_384/checkpoint128_0.001100.0_used_201_to_test_short_embdding/checkpoint.pth.tar


python gcn_verify.py --data_path_201 nasbench_201_dataset_all_sentence_transformer.pkl  --test_201 --em
bedding_size 768 --model_path GNN_Evaluation_Result_nasbench_101_dataset_sen
tance_transformer_768_768/checkpoint128_0.001100.0_/checkpoint.pth.tar \
--comment large_model_long_sentence



python nasbench_101_generation_1.py --model_path all-mpnet-base-v2 --comment long_embdding