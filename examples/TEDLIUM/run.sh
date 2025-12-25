#!/bin/bash
#SBATCH --job-name=speechllm
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # 由 torchrun 自己起多进程
#SBATCH --gres=gpu:4               # 单机4卡
#SBATCH --cpus-per-task=16         # 自行按需（每进程将均分/争用）
#SBATCH --time=480:00:00
#SBATCH --output=logs/TED/edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_1_whisper_20251203_0dropout.out

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

max_lr=1e-4
exp_dir=asr_exp/TED
config=examples/TEDLIUM/conf/whisper_large_v3_pooling_adapter_vicuna7b.json
exp_name=edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_no_context

# config=examples/TEDLIUM/conf/whisper_large_v3_pooling_adapter_vicuna7b_no_lora.json
# exp_name=edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_no_context_no_lora

pretrained_model_path="asr_exp/TED/edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_no_context_no_lora/checkpoints/epoch=1-step=19560-val-acc=0.9698.ckpt"
exp_name=edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_no_context_20251203
train_dataset=manifests/TED/ted_train.json
valid_dataset=manifests/TED/ted_dev.json

num_history=1
exp_name=edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_${num_history}_whisper_20251203_0dropout
train_dataset=manifests/TED/ted_train_${num_history}_whisper.json
valid_dataset=manifests/TED/ted_dev_${num_history}_whisper.json

no_context_prob=0

train_batch_size=6

echo "exp_dir: $exp_dir"
echo "exp_name: $exp_name"
echo "config: $config"
echo "train_dataset: $train_dataset"
echo "valid_dataset: $valid_dataset"
echo "no_context_prob: $no_context_prob"

master_port=$((30000 + RANDOM % 5000))
echo "$master_port"

torchrun --standalone --nproc_per_node=4 --master_port=$master_port examples/TEDLIUM/train.py \
    --config $config \
    --exp_dir $exp_dir \
    --exp_name $exp_name \
    --max_lr $max_lr \
    --train_dataset $train_dataset \
    --valid_dataset $valid_dataset \
    --train_batch_size $train_batch_size \
    --no_context_prob $no_context_prob \
    --pretrained_model_path $pretrained_model_path

