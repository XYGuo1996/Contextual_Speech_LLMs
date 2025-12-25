import os
import json
import re
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

INPUT_ROOT = "data/tedlium/TEDLIUM_release3/legacy" 
OUTPUT_ROOT = "data/tedlium/TEDLIUM_release3/legacy_data"
TARGET_SAMPLE_RATE = 16000

def normalize_text(text):
    """
    文本清洗函数
    """
    text = re.sub(r'<[^>]+>', '', text)
    
    text = text.lower()
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_stm_file(stm_path, wav_output_dir):
    stm_path = Path(stm_path)
    sph_path = stm_path.with_suffix('.sph')
    file_id = stm_path.stem
    
    file_segments = []

    if not sph_path.exists():
        print(f"[Warning] Audio file missing for {stm_path.name}, skipping.")
        return []

    try:
        waveform, sr = torchaudio.load(str(sph_path))
        
        if sr != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
            sr = TARGET_SAMPLE_RATE
        
        total_samples = waveform.size(1)

        with open(stm_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        valid_seg_count = 0

        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            
            if len(parts) < 7:
                continue

            transcript_raw = " ".join(parts[6:])
            
            if "ignore_time_segment_in_scoring" in line or "ignore_time_segment_in_scoring" in transcript_raw:
                continue

            try:
                start_time = float(parts[3])
                end_time = float(parts[4])
            except ValueError:
                print(f"[Warning] Invalid timestamp in {stm_path.name} line {line_idx+1}")
                continue
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            if start_sample < 0: start_sample = 0
            if end_sample > total_samples: end_sample = total_samples
            if start_sample >= end_sample:
                continue

            clean_text = normalize_text(transcript_raw)
            if not clean_text:
                continue

            audio_segment = waveform[:, start_sample:end_sample]
            
            valid_seg_count += 1
            seg_filename = f"{file_id}_{valid_seg_count:05d}.wav"
            seg_save_path = wav_output_dir / seg_filename

            torchaudio.save(str(seg_save_path), audio_segment, sr, encoding="PCM_S", bits_per_sample=16)

            duration = (end_sample - start_sample) / sr
            file_segments.append({
                "audio_path": str(seg_save_path.absolute()),
                "duration": round(duration, 4),
                "text": clean_text
            })

    except Exception as e:
        print(f"[Error] Failed to process {stm_path.name}: {e}")
        return []

    return file_segments

def main():
    input_root = Path(INPUT_ROOT)
    output_root = Path(OUTPUT_ROOT)
    
    splits = ['dev', 'test', 'train']
    
    print("Start processing (Single-thread Sequential Mode)...")
    print(f"Target Sample Rate: {TARGET_SAMPLE_RATE}")
    
    for split in splits:
        stm_dir = input_root / split
        if not stm_dir.exists():
            print(f"Skipping split '{split}' (path not found: {stm_dir})")
            continue
            
        stm_files = sorted(list(stm_dir.glob('*.stm')))
        print(f"\nProcessing split: {split} ({len(stm_files)} files)")
        
        wav_output_dir = output_root / split / "wav"
        wav_output_dir.mkdir(parents=True, exist_ok=True)
        
        split_metadata = []
        
        pbar = tqdm(stm_files, desc=f"Converting {split}")
        
        for stm_file in pbar:
            segments = process_stm_file(stm_file, wav_output_dir)
            split_metadata.extend(segments)
            
            pbar.set_postfix(file=stm_file.stem)
        
        json_path = output_root / f"{split}.json"
        print(f"Saving metadata to {json_path} (Total segments: {len(split_metadata)})")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            for data_ in split_metadata:
                f.write(json.dumps(data_, ensure_ascii=False) + "\n")

    print("\nAll tasks finished successfully.")

if __name__ == "__main__":
    main()