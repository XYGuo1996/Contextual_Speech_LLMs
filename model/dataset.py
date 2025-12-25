import torch
from transformers import AutoProcessor, AutoFeatureExtractor
from transformers.trainer_pt_utils import LabelSmoother
import transformers

import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import random
import numpy as np
import json
import torch.nn.functional as F
import os
import subprocess
import io
import re

def replace_context(b: str, a: str) -> str:
    return re.sub(r"<context_text>.*?</context_text>", f"<context_text>\n{a}</context_text>", b, flags=re.DOTALL)


class MyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        waveforms, wavform_normalizes, texts, task_prompts, prompt_templates, audio_paths = [], [], [], [], [], []
        
        audio_lengths = []
        for sample in batch:
            waveform, wavform_normalize, text, task_prompt, prompt_template, audio_path = sample
            waveforms.append(waveform.squeeze(0))
            wavform_normalizes.append(wavform_normalize.squeeze(0))
            texts.append(text)
            task_prompts.append(task_prompt)
            prompt_templates.append(prompt_template)
            audio_paths.append(audio_path)

            audio_lengths.append(waveform.squeeze(0).shape[0])

        audio_mask = torch.zeros(len(batch), max(audio_lengths))
        for i, length in enumerate(audio_lengths):
            audio_mask[i, :length] = 1

        from torch.nn.utils.rnn import pad_sequence
        batch_waveform = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
        batch_waveform_normalize = pad_sequence(wavform_normalizes, batch_first=True, padding_value=0.0)
        return batch_waveform, batch_waveform_normalize, audio_mask, texts, task_prompts, prompt_templates, audio_paths, torch.tensor(audio_lengths)
    
class InstructionalAudioDataset():
    def __init__(self, json_file, mode='train', no_context_prob=0.5):
        self.data_frame = []
        with open(json_file, 'r') as f:
            for line in f:
                data_ = json.loads(line.strip())
                self.data_frame.append(data_)

        self.mode = mode
        self.no_context_prob = no_context_prob

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):

        audio_row = self.data_frame[idx]
        audio_path = audio_row['audio_path']
        text = audio_row['text']
        task_prompt = audio_row['task_prompt']
        prompt_template = audio_row['prompt_template']

        if "context_text" in audio_row:
            if random.random() < self.no_context_prob:
                context_text_selected = ""
            else:
                context_text = audio_row['context_text'][0]
                if len(context_text) > 0:
                    context_text_selected = context_text[random.randint(0, len(context_text)-1)]
                else:
                    context_text_selected = ""
            task_prompt = task_prompt.replace("this_need_replace_to_context_text", context_text_selected)
        # print(f"audio_row: {audio_row}")
        # print(f"task_prompt: {task_prompt}")
        else:
            if self.mode is not "test":
                if random.random() < self.no_context_prob:
                    task_prompt = replace_context(task_prompt, "")


        if "offset" in audio_row and "duration" in audio_row:
            offset = audio_row['offset']
            duration = audio_row['duration']
            waveform, sample_rate = self.load_segment_fast(audio_path, offset, duration)
        else:
            waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            sample_rate = 16000

        wavform_normalize = waveform.squeeze(0)
        wavform_normalize = torch.nn.functional.layer_norm(wavform_normalize, wavform_normalize.shape)
        wavform_normalize = wavform_normalize.unsqueeze(0)
        
        return waveform, wavform_normalize, text, task_prompt, prompt_template, audio_path

    def load_segment_fast(self, audio_file: str, start_s: float, duration_s: float):
        """
        Efficiently read audio from a specified time period using ffmpeg.
        """
        cmd = [
            "ffmpeg",
            "-ss", f"{start_s:.6f}",
            "-t", f"{duration_s:.6f}",
            "-i", audio_file,
            "-f", "wav",
            "-ac", "1",
            "-ar", "16000",
            "pipe:1"
        ]
        
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {proc.stderr.decode()}")

        waveform, sr = torchaudio.load(io.BytesIO(proc.stdout))

        return waveform, sr