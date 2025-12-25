
import argparse
import json
import os
import torch
import editdistance
from torch.utils.data import DataLoader
from peft import PeftModel

from model.trainer import SpeechLLMLightning
from model.dataset import InstructionalAudioDataset, MyCollator

def compute_wer(hyps, refs):
    if len(hyps) + len(refs) <= 0:
        return 0
    error_total = 0
    length_total = 0
    for hyp, ref in zip(hyps, refs):
        hyp_words = hyp.split()
        ref_words = ref.split()
        error = editdistance.eval(hyp_words, ref_words)
        error_total += error
        length_total += len(ref_words)
    wer = error_total * 100.0 / max(1, length_total)
    return wer

def test_dpo(args):
    with open(args.config, "r") as f:
        model_config = json.load(f)

    model_config["exp_dir"] = args.exp_dir
    model_config["exp_name"] = args.exp_name
    model_config["test_basename"] = args.test_basename
    model_config["decode_utt_by_utt"] = args.decode_utt_by_utt
    model_config["pre_utt_his_num"] = args.pre_utt_his_num

    if args.decode_utt_by_utt:
        print("DPO decoding in utt_by_utt mode is enabled.")

    from peft import PeftModel, LoraConfig, set_peft_model_state_dict

    print(f"Loading Base + SFT model from {args.sft_ckpt}...")
    model = SpeechLLMLightning(model_config, num_validation_samples=0)
    model.load_pretrained_model(args.sft_ckpt)
    
    if isinstance(model.llm_model, PeftModel):
        model.llm_model = model.llm_model.merge_and_unload()

    print(f"Loading DPO adapter from {args.dpo_adapter_path}...")

    if args.dpo_adapter_path.endswith(".ckpt"):
        ADAPTER_NAME = "dpo_adapter" 
        
        dpo_config = LoraConfig(
            r=model_config.get("lora_r", 8),
            lora_alpha=args.decode_lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=True
        )

        model.llm_model = PeftModel(model.llm_model, dpo_config, adapter_name=ADAPTER_NAME)

        checkpoint = torch.load(args.dpo_adapter_path, map_location="cpu")
        sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        if "dpo_adapter" in sd:
            print(f"[INFO] Found nested 'dpo_adapter' key in checkpoint. Unpacking...")
            sd = sd["dpo_adapter"]
        
        adapter_weights = {}
        print(f"[DEBUG] Inspecting keys from checkpoint (First 5): {list(sd.keys())[:5]}")

        for k, v in sd.items():
            new_k = k.replace("llm_model.", "")
            adapter_weights[new_k] = v

        if len(adapter_weights) == 0:
            raise ValueError("!!! No adapter weights were extracted! Checkpoint structure might be wrong.")

        incompatible = set_peft_model_state_dict(model.llm_model, adapter_weights, adapter_name=ADAPTER_NAME)
        print(f"[DEBUG] Load results: {incompatible}")
        
        model.llm_model.set_adapter(ADAPTER_NAME)
        
        print("Verifying loaded weights...")
        has_non_zero = False
        for name, param in model.llm_model.named_parameters():
            if "lora_B" in name and ADAPTER_NAME in name:
                if torch.all(param == 0):
                    print(f"!!! WARNING: Parameter {name} is ALL ZEROS. Load might have failed!")
                else:
                    print(f"SUCCESS: Parameter {name} has non-zero values. Mean: {param.abs().mean().item()}")
                    has_non_zero = True
                    break
        
        if not has_non_zero:
             print("!!! CRITICAL WARNING: No non-zero LoRA B weights found. Inference will behave like Base Model.")

    else:
        print("Detected HuggingFace adapter folder.")
        if isinstance(model.llm_model, PeftModel):
            model.llm_model.load_adapter(args.dpo_adapter_path, adapter_name="dpo_adapter")
        else:
            model.llm_model = PeftModel.from_pretrained(model.llm_model, args.dpo_adapter_path, adapter_name="dpo_adapter")
        model.llm_model.set_adapter("dpo_adapter")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = InstructionalAudioDataset(
        json_file=args.test_dataset,
        mode="test",
        no_context_prob=0.0
    )
    collator = MyCollator(model.llm_tokenizer)

    batch_size = 1
    if args.decode_utt_by_utt and batch_size != 1:
        raise ValueError("When decode_utt_by_utt is True, batch_size must be 1.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=4)

    out_dir = os.path.join(args.exp_dir, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    
    with open(f"{out_dir}/{args.test_basename}_infer_ref.txt", "w") as f:
        pass
    with open(f"{out_dir}/{args.test_basename}_infer_hyp.txt", "w") as f:
        pass

    print("Starting Inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch_on_device = []
            for item in batch:
                if isinstance(item, torch.Tensor):
                    batch_on_device.append(item.to(device))
                else:
                    batch_on_device.append(item)
            
            model.test_step(batch_on_device, batch_idx)
            
            if batch_idx % 100 == 0:
                print(f"Processed {batch_idx} samples...")

    print(f"Inference done. Results saved to {out_dir}")


def parse_args():
    import distutils.util

    def str2bool(v):
        return bool(distutils.util.strtobool(v))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--sft_ckpt", required=True, help="Path to SFT checkpoint (base + sft lora)")
    parser.add_argument("--dpo_adapter_path", required=True, help="Path to DPO adapter folder (saved by train_dpo.py)")
    parser.add_argument("--test_dataset", required=True)
    parser.add_argument("--test_basename", required=True)
    parser.add_argument("--decode_utt_by_utt", type=str2bool, default=False, help="Whether to decode utterance by utterance")
    parser.add_argument("--pre_utt_his_num", type=int, default=0, help="Number of previous utterances to use as history context")
    parser.add_argument("--decode_lora_alpha", type=int, default=8, help="Lora alpha for decoding")
    return parser.parse_args()


if __name__ == "__main__":
    test_dpo(parse_args())
