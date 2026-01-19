#!/bin/bash
#SBATCH --job-name=speechllm_dpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=240:00:00
#SBATCH --output=logs/TED/edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_2_whisper_0.5dropout_DPO_WER_5.out

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

max_lr=1e-4
config=examples/TEDLIUM/conf/whisper_large_v3_pooling_adapter_vicuna7b.json
exp_dir=asr_exp/TED

policy_ckpt="asr_exp/TED/edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_2_whisper_20251203_0.5dropout/checkpoints/epoch=1-step=16766-val-acc=0.9719.ckpt"

train_base_jsonl=manifests/TED/ted_train_2_whisper.json
train_hyp_txt=asr_exp/TED/edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_2_whisper_20251203_0.5dropout/train_hyp.txt
train_dpo_jsonl=manifests/TED/ted_train_2_dpo.json
dev_dpo_jsonl=manifests/TED/ted_dev_2_dpo.json

no_context_prob=0

dpo_exp_name=edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_2_whisper_20251203_0.5dropout_DPO

dpo_exp_name=edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_2_whisper_20251203_0.5dropout_DPO_WER_5
train_dpo_jsonl=manifests/TED/ted_train_2_DPO_WER_5.json
dev_dpo_jsonl=manifests/TED/ted_dev_2_DPO_WER_5.json

train_batch_size=2
val_batch_size=2
num_epochs=2
learning_rate=1e-5
beta=0.1
max_grad_norm=1.0
logging_steps=10

mkdir -p "$exp_dir"

echo "exp_dir: $exp_dir"
echo "dpo_exp_name: $dpo_exp_name"
echo "config: $config"
echo "train_dpo_jsonl: $train_dpo_jsonl"
echo "policy_ckpt: $policy_ckpt"
echo "logging_steps: $logging_steps"

master_port=$((30000 + RANDOM % 5000))
echo "$master_port"

torchrun --standalone --nproc_per_node=4 --master_port=$master_port examples/TEDLIUM/train_dpo.py \
    --config "$config" \
    --exp_dir "$exp_dir" \
    --exp_name "$dpo_exp_name" \
    --dpo_dataset "$train_dpo_jsonl" \
    --val_dataset "$dev_dpo_jsonl" \
    --policy_ckpt "$policy_ckpt" \
    --learning_rate $learning_rate \
    --train_batch_size $train_batch_size \
    --val_batch_size $val_batch_size \
    --logging_steps $logging_steps \
    --num_epochs $num_epochs \
    --beta $beta \
    --max_grad_norm $max_grad_norm \
    --no_context_prob $no_context_prob \
    --gpus 4
