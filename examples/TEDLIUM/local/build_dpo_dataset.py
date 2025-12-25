import argparse
import json
from typing import Dict, Any
import random

def load_base_dataset(path: str) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            audio_path = obj.get("audio_path")
            if not audio_path:
                continue
            if audio_path in index:
                pass
            index[audio_path] = obj
    return index


def load_hypotheses(path: str):
    hyps = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            audio_path = parts[0]
            hyp = " ".join(parts[1:])
            hyps[audio_path] = hyp
    return hyps


def build_dpo_dataset(base_jsonl: str, hyp_txt: str, output_jsonl: str, WER: int) -> None:
    base_index = load_base_dataset(base_jsonl)
    hyp_index = load_hypotheses(hyp_txt)

    num_written = 0
    num_missing = 0

    num_good = 0
    num_bad = 0
    good_data = []
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for audio_path, base_obj in base_index.items():
            if audio_path not in hyp_index:
                num_missing += 1
                continue
            obj = dict(base_obj)
            gt_text = obj.get("text", "")
            hyp_text = hyp_index[audio_path]

            obj.setdefault("chosen", gt_text)
            obj["rejected"] = hyp_text
            if obj["wer"] < WER:
                good_data.append(obj)
                num_good += 1
                continue
            num_bad += 1
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            num_written += 1

    print(f"Wrote {num_written} DPO examples to {output_jsonl}, bad: {num_bad}")
    if num_missing > 0:
        print(
            f"Warning: {num_missing} samples from base dataset did not have hypotheses "
            f"in {hyp_txt} and were skipped."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a JSONL dataset for DPO training by pairing the original "
            "JSONL (with ground-truth `text`) and a *_infer_hyp.txt file."
        )
    )
    parser.add_argument(
        "--base_jsonl",
        type=str,
        default="manifests/TED/ted_train_2_whisper.json",
        help="Original JSONL dataset used for SFT (e.g. manifests/TED/ted_train_*.json).",
    )
    parser.add_argument(
        "--hyp_txt",
        type=str,
        default="asr_exp/TED/edp_whisper_large_v3_pooling_adapter_vicuna7b_TED_2_whisper_20251203_0.5dropout/ted_train_2_gt.json_infer_hyp.txt",
        help="Path to *_infer_hyp.txt produced by evaluation (lines: '<audio_path> <hyp ...>').",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="SpeechLLM/manifests/TED/ted_train_2_DPO.json",
        help="Output JSONL path for DPO training data.",
    )
    parser.add_argument(
        "--WER",
        type=int,
        default=10,
        help="WER threshold for filtering data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dpo_dataset(
        base_jsonl=args.base_jsonl,
        hyp_txt=args.hyp_txt,
        output_jsonl=args.output_jsonl,
        WER=args.WER,
    )
