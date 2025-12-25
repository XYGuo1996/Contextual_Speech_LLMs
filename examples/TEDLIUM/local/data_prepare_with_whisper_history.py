import json
import os
from collections import defaultdict
import copy

source_files = [
    "manifests/TED/split_decode/ted_train/data_with_wer.json", 
    "manifests/TED/split_decode/ted_dev/data_with_wer.json", 
    "manifests/TED/split_decode/ted_test/data_with_wer.json"
]

context_lengths = [1, 2, 3, 4, 5]

def get_output_path(input_path, n):
    base, ext = os.path.splitext(input_path)
    if "train" in base:
        return f"manifests/TED/ted_train_{n}_whisper{ext}"
    if "dev" in base:
        return f"manifests/TED/ted_dev_{n}_whisper{ext}"
    if "test" in base:
        return f"manifests/TED/ted_test_{n}_whisper{ext}"


def process_file_fixed_contexts(file_path):
    if not os.path.exists(file_path):
        print(f"Skipping: {file_path} (File not found)")
        return

    print(f"Reading source: {file_path} ...")
    
    raw_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    grouped_data = defaultdict(list)
    for item in raw_data:
        filename = os.path.basename(item['audio_path'])
        name_no_ext = os.path.splitext(filename)[0]
        
        try:
            talk_id, seg_index_str = name_no_ext.rsplit('_', 1)
            item['_temp_sort_idx'] = int(seg_index_str)
            grouped_data[talk_id].append(item)
        except ValueError:
            print(f"Warning: Skipping file with unexpected format: {filename}")
            continue

    for talk_id in grouped_data:
        grouped_data[talk_id].sort(key=lambda x: x['_temp_sort_idx'])

    for n in context_lengths:
        output_data = []
        
        for talk_id, segments in grouped_data.items():
            all_texts = [s.get('asr_txt', "") for s in segments]
            
            for i, original_item in enumerate(segments):
                item = copy.deepcopy(original_item)
                
                if '_temp_sort_idx' in item:
                    del item['_temp_sort_idx']
                
                if i > 0:
                    start_idx = max(0, i - n)
                    context_sentences = all_texts[start_idx : i]
                    
                    if context_sentences:
                        context_str = "\n".join(context_sentences)
                        new_prompt = (
                            f"<context_text>\n{context_str}\n</context_text>\n"
                            f"Transcribe speech to text based on the context_text."
                        )
                        item['task_prompt'] = new_prompt
                
                output_data.append(item)
        
        output_path = get_output_path(file_path, n)
        try:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for item in output_data:
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"  -> Generated: {output_path} (Context={n})")
        except Exception as e:
            print(f"Error writing {output_path}: {e}")

if __name__ == "__main__":
    for f in source_files:
        process_file_fixed_contexts(f)
    print("\nAll tasks completed.")
