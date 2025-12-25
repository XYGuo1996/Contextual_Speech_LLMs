#!/bin/bash
#SBATCH --job-name=eval_edp_speechllm
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # 由 torchrun 自己起多进程
#SBATCH --gres=gpu:1               # 单机4卡
#SBATCH --cpus-per-task=16         # 自行按需（每进程将均分/争用）
#SBATCH --time=480:00:00
#SBATCH --output=logs/TED/eval_utt_by_utt_edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_2_whisper_20251203_0dropout_attacks_from_ted.out

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

max_lr=1e-4
batch_size=1
exp_dir=asr_exp/TED
config=examples/TEDLIUM/conf/whisper_large_v3_pooling_adapter_vicuna7b.json

pre_utt_his_num=2
exp_name=edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_${pre_utt_his_num}_whisper_20251203_0dropout
ckpt="asr_exp/TED/edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_${pre_utt_his_num}_whisper_20251203_0dropout/checkpoints/epoch=1-step=19560-val-acc=0.9725.ckpt"
testsets=manifests/test_clean_${pre_utt_his_num}_gt.json
# testsets=manifests/test_other_${pre_utt_his_num}_gt.json
testsets=manifests/TED/ted_test_2_gt_attacks_from_ted.json

decode_utt_by_utt=false

echo "testing SpeechLLM with the following parameters:"
echo "ckpt: $ckpt"
echo "exp_dir: $exp_dir"
echo "exp_name: $exp_name"
echo "config: $config"
echo "testsets: $testsets"
echo "batch_size: $batch_size"
echo "decode_utt_by_utt: $decode_utt_by_utt"
echo "pre_utt_his_num: $pre_utt_his_num"

python examples/TEDLIUM/test.py \
    --config $config \
    --exp_dir $exp_dir \
    --exp_name $exp_name \
    --ckpt $ckpt \
    --testsets $testsets \
    --batch_size $batch_size \
    --decode_utt_by_utt $decode_utt_by_utt \
    --pre_utt_his_num $pre_utt_his_num
