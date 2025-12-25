#!/bin/bash
#SBATCH --job-name=speechllm_dpo_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/TED/eval_utt_by_utt_test_other_edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_2_whisper_20251203_0.5dropout_DPO_WER_15_decode_lora_alpha_20.out

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


config=examples/TEDLIUM/conf/whisper_large_v3_pooling_adapter_vicuna7b.json
exp_dir=asr_exp/TED

sft_ckpt="asr_exp/TED/edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_2_whisper_20251203_0.5dropout/checkpoints/epoch=1-step=16766-val-acc=0.9719.ckpt"

dpo_adapter_path="asr_exp/TED/edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_2_whisper_20251203_0.5dropout_DPO/checkpoints_dpo/epoch=1-step=686-val_loss=0.6506-val_pi_diff=1.4377.ckpt"

pre_utt_his_num=2
decode_utt_by_utt=true
decode_lora_alpha=20

exp_name=edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_2_whisper_0.5dropout_DPO_WER_15
# test_dataset=manifests/TED/ted_test_2_gt.json
# test_dataset=manifests/TED/ted_dev_2_gt.json
# test_dataset=manifests/TED/ted_test_2_gt_attacks_from_ted.json
# test_dataset=manifests/test_clean_2_gt.json
test_dataset=manifests/test_other_2_gt.json
# test_basename=test_clean
test_basename=test_other
# test_basename=TED_test_attacks_from_ted

echo "Starting DPO Inference..."
echo "SFT Checkpoint: $sft_ckpt"
echo "DPO Adapter: $dpo_adapter_path"
echo "Test Dataset: $test_dataset"

python examples/TEDLIUM/test_dpo.py \
    --config $config \--exp_dir $exp_dir \
    --exp_name $exp_name \
    --sft_ckpt "$sft_ckpt" \
    --dpo_adapter_path "$dpo_adapter_path" \
    --test_dataset $test_dataset \
    --test_basename $test_basename \
    --pre_utt_his_num $pre_utt_his_num \
    --decode_utt_by_utt $decode_utt_by_utt \
    --decode_lora_alpha $decode_lora_alpha

echo "Inference finished."
