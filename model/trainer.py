
import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup
import lightning.pytorch as pl
import numpy as np
from jiwer import wer
import torchmetrics
import random
import editdistance
import re
import math
import json
from torch.nn.utils.rnn import pad_sequence
import os
from collections import defaultdict

from model.encoder.encoder import get_audio_encoder, TransformerAudioEnoder
from model.projector.connector import get_connector, LinearConnector, LinearPoolConnector, CNNConnector
from model.llm import get_llm
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def compute_wer(hyps, refs):
    if len(hyps) + len(refs) <= 0:
        return 0
    error_total = 0
    length_total = 0
    for hyp, ref in zip(hyps, refs):
        # print(f"pre: {hyp}, ref: {ref}")
        hyp_words = hyp.split()
        ref_words = ref.split()
        error = editdistance.eval(hyp_words, ref_words)
        error_total += error
        length_total += len(ref_words)
    wer = error_total * 100.0 / length_total
    return wer

class SpeechLLMLightning(pl.LightningModule):
    def __init__(self, model_config=None, num_validation_samples=0):
        super().__init__()
        
        torch.manual_seed(model_config.get('seed'))

        self.save_hyperparameters()
        self.lang_name = model_config.get('lang_name')
        self.exp_name = model_config.get('exp_name')
        self.llm_name = model_config.get('llm_name',None)
        self.finetune_encoder = model_config.get('finetune_encoder')
        self.use_lora = model_config.get('use_lora')
        self.adapter_name = model_config.get('adapter_name')
        self.audio_encoder_name = model_config.get('audio_encoder_name')
        self.audio_encoder_type = model_config.get('audio_encoder_type')
        self.audio_encoder = get_audio_encoder(model_config,model_config.get('audio_encoder_name'), model_config.get('finetune_encoder'))
        
        self.connector = get_connector(model_config.get('adapter_name'), model_config.get('adapter_conf'))
        
        self.llm_tokenizer, self.llm_model = get_llm(model_config.get('llm_name'), model_config.get('use_lora'), model_config.get('lora_r'), model_config.get('lora_alpha'))
        self.max_lr = model_config.get('max_lr')
        self.warmup_steps = model_config.get('warmup_steps')
        self.max_epochs = model_config.get('max_epochs')
        self.weight_decay = model_config.get('weight_decay', 0)
        self.exp_dir = model_config.get('exp_dir', None)
        self.exp_name = model_config.get('exp_name', None)
        self.test_basename = model_config.get('test_basename', None)
        self.audio_normalize = model_config.get('audio_normalize', False)

        self.num_validation_samples = num_validation_samples
        self.llm_model.config.pad_token_id = self.llm_tokenizer.pad_token_id
        self.pad_token_id = self.llm_tokenizer.pad_token_id
        self.pre_utt_file_name = None
        self.pre_utt_pred_history = []
        self.decode_utt_by_utt = model_config.get('decode_utt_by_utt', False)
        self.pre_utt_his_num = model_config.get('pre_utt_his_num', 1)
        self.pretrained_model_path = model_config.get('pretrained_model_path', None)
        if self.pretrained_model_path is not None:
            self.load_pretrained_model()

    def load_pretrained_model(self, ckpt=None):
        if ckpt is not None:
            state_dict = torch.load(ckpt, map_location='cpu')
        else:
            state_dict = torch.load(self.pretrained_model_path, map_location='cpu')
        state_dict = state_dict["state_dict"]
        if "connector" in state_dict:
            self.connector.load_state_dict(state_dict["connector"])
            print(f"DEBUG: Loaded connector state dict: {state_dict['connector'].keys()} from {self.pretrained_model_path if ckpt is None else ckpt}")
        if "llm_lora" in state_dict:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.llm_model, state_dict["llm_lora"])
            print(f"DEBUG: Loaded llm_lora state dict: {state_dict['llm_lora'].keys()} from {self.pretrained_model_path if ckpt is None else ckpt}")

    def on_save_checkpoint(self, checkpoint) -> None:
        state_dict = {}
        state_dict["connector"] = self.connector.state_dict()
        if self.use_lora:
            from peft import get_peft_model_state_dict
            state_dict["llm_lora"] = get_peft_model_state_dict(self.llm_model)
        else:
            state_dict["llm"] = self.llm_model.state_dict()
        checkpoint["state_dict"] = state_dict

    def on_load_checkpoint(self, checkpoint) -> None:
        state_dict = checkpoint["state_dict"]
        if "connector" in state_dict:
            self.connector.load_state_dict(state_dict["connector"])
        if self.use_lora:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.llm_model, state_dict["llm_lora"])
        else:
            self.llm_model.load_state_dict(state_dict["llm"], strict=False)

    def configure_optimizers(self):
        
        opt = [
            {"params": self.connector.parameters(), "lr": self.max_lr},
            {"params": self.llm_model.parameters(), "lr": self.max_lr},
        ]
        optimizer = AdamW(opt, weight_decay=0.0)
        
        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda)

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def merge_input_ids_with_speech_features_train(self, batch_wavform, batch_wavform_normalize, audio_mask, texts, task_prompts, prompt_templates, audio_lengths, return_embedding_loss=False):
        batch_size = batch_wavform.shape[0]

        if hasattr(self.llm_model.model, "embed_tokens"):
            embedder = self.llm_model.model.embed_tokens
        elif hasattr(self.llm_model.model.model, "embed_tokens"):
            embedder = self.llm_model.model.model.embed_tokens
        else:
            embedder = self.llm_model.model.model.model.embed_tokens
        
        with torch.no_grad():
            if self.audio_normalize:
                speech_encoder_input = batch_wavform_normalize
            else:
                speech_encoder_input = batch_wavform
            
            if self.audio_encoder_name == "hubert":
                if self.audio_encoder_type == "pretrain":
                    results = self.audio_encoder(source=speech_encoder_input, padding_mask = 1-audio_mask, mask=False, features_only=True)
                    speech_embeds, audio_mel_post_mask = results["x"], results["padding_mask"]
                if self.audio_encoder_type == "finetune":
                    results = self.audio_encoder(source=speech_encoder_input, padding_mask = 1-audio_mask)
                    speech_embeds, audio_mel_post_mask = results["encoder_out"], results["padding_mask"]
                    speech_embeds = speech_embeds.transpose(0, 1)
            elif self.audio_encoder_name == "whisper":
                speech_embeds, speech_embeds_lens = self.audio_encoder(speech_encoder_input, audio_lengths)

        speech_embeds = self.connector(speech_embeds)
        
        pre_prompts = []
        post_prompts = []
        # print(f"train speech_embeds: {speech_embeds.shape}")
        # print(f"train audio_mask: {audio_mask.shape}")

        texts = [t + self.llm_tokenizer.eos_token for t in texts]

        for index in range(batch_size):
            pre_prompt, post_prompt = prompt_templates[index].split("<S> <P>")
            post_prompt = post_prompt.replace("<T>", "")
            pre_prompts.append(pre_prompt)
            post_prompts.append(post_prompt)

        pre_prompt_tokens_att = self.llm_tokenizer(pre_prompts, padding=True, return_tensors='pt')
        task_prompt_tokens_att = self.llm_tokenizer(task_prompts, padding=True, return_tensors='pt')
        post_prompt_tokens_att = self.llm_tokenizer(post_prompts, padding=True, return_tensors='pt')
        text_tokens_att = self.llm_tokenizer(texts, padding=True, return_tensors='pt')

        pre_prompt_tokens = pre_prompt_tokens_att["input_ids"]
        task_prompt_tokens = task_prompt_tokens_att["input_ids"]
        post_prompt_tokens = post_prompt_tokens_att["input_ids"]
        text_tokens = text_tokens_att["input_ids"]
        
        pre_prompt_att = pre_prompt_tokens_att["attention_mask"]
        task_prompt_att = task_prompt_tokens_att["attention_mask"]
        post_prompt_att = post_prompt_tokens_att["attention_mask"]
        text_att = text_tokens_att["attention_mask"]
        speech_embed_att = torch.ones((batch_size, speech_embeds.shape[1]), dtype=torch.long).to(speech_embeds.device)
        
        del pre_prompt_tokens_att, task_prompt_tokens_att, post_prompt_tokens_att, text_tokens_att
     
        pre_prompt_embed = embedder(pre_prompt_tokens.to(self.device))
        task_prompt_embed = embedder(task_prompt_tokens.to(self.device))
        post_prompt_embed = embedder(post_prompt_tokens.to(self.device))
        text_embed = embedder(text_tokens.to(self.device))

        combined_embeds = torch.cat((pre_prompt_embed, speech_embeds, task_prompt_embed, post_prompt_embed, text_embed), dim=1)
        combined_att = torch.cat((pre_prompt_att.to(self.device), speech_embed_att.to(self.device), task_prompt_att.to(self.device), post_prompt_att.to(self.device), text_att.to(self.device)), dim=1)

        pre_prompt_att_for_label_ids = torch.full_like(pre_prompt_tokens, -100)
        speech_embed_att_for_label_ids = torch.full_like(speech_embed_att, -100)
        task_prompt_att_for_label_ids = torch.full_like(task_prompt_att, -100)
        post_prompt_att_for_label_ids = torch.full_like(post_prompt_att, -100)
        text_att_for_label_ids = torch.where(text_att == 1, text_tokens, -100)
        
        combined_label_ids = torch.cat([pre_prompt_att_for_label_ids.to(self.device), speech_embed_att_for_label_ids.to(self.device), task_prompt_att_for_label_ids.to(self.device), post_prompt_att_for_label_ids.to(self.device), text_att_for_label_ids.to(self.device)], dim=1)
        
        del pre_prompt_embed, speech_embeds, task_prompt_embed, post_prompt_embed, text_embed
        del pre_prompt_att, speech_embed_att, task_prompt_att, post_prompt_att, text_att
        del pre_prompt_att_for_label_ids, speech_embed_att_for_label_ids, task_prompt_att_for_label_ids, post_prompt_att_for_label_ids, text_att_for_label_ids
        
        return combined_embeds, combined_att, combined_label_ids
        
    def merge_input_ids_with_speech_features_test(self, batch_wavform, batch_wavform_normalize, audio_mask, task_prompts, prompt_templates, audio_lengths, return_embedding_loss=False):
        batch_size = batch_wavform.shape[0]

        if hasattr(self.llm_model.model, "embed_tokens"):
            embedder = self.llm_model.model.embed_tokens
        elif hasattr(self.llm_model.model.model, "embed_tokens"):
            embedder = self.llm_model.model.model.embed_tokens
        else:
            embedder = self.llm_model.model.model.model.embed_tokens

        with torch.no_grad():
            if self.audio_normalize:
                speech_encoder_input = batch_wavform_normalize
            else:
                speech_encoder_input = batch_wavform
            
            if self.audio_encoder_name == "hubert":                
                if self.audio_encoder_type == "pretrain":
                    results = self.audio_encoder(source=speech_encoder_input, padding_mask = 1-audio_mask, mask=False, features_only=True)
                    speech_embeds, audio_mel_post_mask = results["x"], results["padding_mask"]
                if self.audio_encoder_type == "finetune":
                    results = self.audio_encoder(source=speech_encoder_input, padding_mask = 1-audio_mask)
                    speech_embeds, audio_mel_post_mask = results["encoder_out"], results["padding_mask"]
                    speech_embeds = speech_embeds.transpose(0, 1)
            elif self.audio_encoder_name == "whisper":
                speech_embeds, speech_embeds_lens = self.audio_encoder(speech_encoder_input, audio_lengths)

        speech_embeds = self.connector(speech_embeds)

        pre_prompts = []
        post_prompts = []

        # print(f"speech_embeds: {speech_embeds.shape}")
        # print(f"audio_mask: {audio_mask.shape}")

        for index in range(batch_size):
            pre_prompt, post_prompt = prompt_templates[index].split("<S> <P>")
            post_prompt = post_prompt.replace("<T>", "")
            pre_prompts.append(pre_prompt)
            post_prompts.append(post_prompt)

        pre_prompt_tokens_att = self.llm_tokenizer(pre_prompts, padding=True, return_tensors='pt')
        task_prompt_tokens_att = self.llm_tokenizer(task_prompts, padding=True, return_tensors='pt')
        post_prompt_tokens_att = self.llm_tokenizer(post_prompts, padding=True, return_tensors='pt')
        
        pre_prompt_tokens = pre_prompt_tokens_att["input_ids"]
        task_prompt_tokens = task_prompt_tokens_att["input_ids"]
        post_prompt_tokens = post_prompt_tokens_att["input_ids"]
        
        pre_prompt_att = pre_prompt_tokens_att["attention_mask"]
        task_prompt_att = task_prompt_tokens_att["attention_mask"]
        post_prompt_att = post_prompt_tokens_att["attention_mask"]
        speech_embed_att = torch.ones((batch_size, speech_embeds.shape[1]), dtype=torch.long).to(speech_embeds.device)
        
        del pre_prompt_tokens_att, task_prompt_tokens_att, post_prompt_tokens_att

        pre_prompt_embed = embedder(pre_prompt_tokens.to(self.device))
        task_prompt_embed = embedder(task_prompt_tokens.to(self.device))
        post_prompt_embed = embedder(post_prompt_tokens.to(self.device))

        combined_embeds = torch.cat((pre_prompt_embed, speech_embeds, task_prompt_embed, post_prompt_embed), dim=1)
        combined_att = torch.cat((pre_prompt_att.to(self.device), speech_embed_att.to(self.device), task_prompt_att.to(self.device), post_prompt_att.to(self.device)), dim=1)
        
        del pre_prompt_embed, speech_embeds, task_prompt_embed, post_prompt_embed
        del pre_prompt_att, speech_embed_att, task_prompt_att, post_prompt_att
        
        return combined_embeds, combined_att
    
    def training_step(self, batch, batch_idx):
        batch_wavform, batch_wavform_normalize, audio_mask, texts, task_prompts, prompt_templates, audio_paths, audio_lengths = batch
        
        embeds, atts, label_ids = self.merge_input_ids_with_speech_features_train(batch_wavform, batch_wavform_normalize, audio_mask, texts, task_prompts, prompt_templates, audio_lengths)
            
        with torch.set_grad_enabled(True):
            model_outputs = self.llm_model(
                    inputs_embeds=embeds,
                    attention_mask=atts,
                    labels=label_ids,
                )
            loss = model_outputs.loss
        assert loss.requires_grad == True

        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc = compute_accuracy(
                preds.detach()[:, :-1],
                label_ids.detach()[:, 1:],
                ignore_label=-100,
            )

        lr_connector = self.trainer.optimizers[0].param_groups[0]['lr']
        lr_llm = self.trainer.optimizers[0].param_groups[1]['lr']

        self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("acc", acc, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("lr_connector", lr_connector, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("lr_llm", lr_llm, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        del embeds, atts, label_ids, model_outputs, preds
        return loss
        
    def validation_step(self, batch, batch_idx):

        batch_wavform, batch_wavform_normalize, audio_mask, texts, task_prompts, prompt_templates, audio_paths, audio_lengths = batch
        # original_target_texts = [self.llm_tokenizer.decode(ids, skip_special_tokens=True) for ids in texts]

        embeds, atts, label_ids = self.merge_input_ids_with_speech_features_train(batch_wavform, batch_wavform_normalize, audio_mask, texts, task_prompts, prompt_templates, audio_lengths) # This list is modified in-place
        
        with torch.set_grad_enabled(False):
            model_outputs = self.llm_model(
                    inputs_embeds=embeds,
                    attention_mask=atts,
                    labels=label_ids,
                )
            loss = model_outputs.loss
        assert loss.requires_grad == False

        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc = compute_accuracy(
                preds.detach()[:, :-1],
                label_ids.detach()[:, 1:],
                ignore_label=-100,
            )

        self.log("val-acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val-loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        del embeds, atts, label_ids, model_outputs, preds

    def test_step(self, batch, batch_idx):
        
        batch_wavform, batch_wavform_normalize, audio_mask, texts, task_prompts, prompt_templates, audio_paths, audio_lengths = batch
        # print(f"\ntasks_prompts before: {task_prompts}")
        # print(f"audio_paths: {audio_paths}")
        if self.decode_utt_by_utt:
            this_utt_file = audio_paths[0].rsplit("/")[-1].rsplit("_", 1)[0]
            assert len(audio_paths) == 1, "When decode_utt_by_utt is True, the batch size must be 1."
            if self.pre_utt_file_name != this_utt_file:
                self.pre_utt_file_name = this_utt_file
                self.pre_utt_pred_history = []
            elif self.pre_utt_file_name == this_utt_file:
                while len(self.pre_utt_pred_history) > self.pre_utt_his_num:
                    self.pre_utt_pred_history.pop(0)
                # task_prompts = [f"{task_prompts[0]} Previous conversation history: {' '.join(self.pre_utt_pred_history)}."]
            history_prompt = "\n".join(self.pre_utt_pred_history)
            task_prompts[0] = f"<context_text>\n{history_prompt}</context_text>\nTranscribe speech to text based on the context_text."
        # print(f"tasks_prompts after: {task_prompts}")
        embeds, atts = self.merge_input_ids_with_speech_features_test(batch_wavform, batch_wavform_normalize, audio_mask, task_prompts, prompt_templates, audio_lengths)

        with torch.no_grad():
            generated_ids = self.llm_model.generate(
                inputs_embeds=embeds,
                attention_mask=atts,
                max_new_tokens=200, 
                num_beams=4,
                do_sample=False,
                min_length=1,
                top_p=1.0,
                repetition_penalty=1.0,
                length_penalty=1.0,
                temperature=1.0,
                eos_token_id=self.llm_tokenizer.eos_token_id, 
                pad_token_id=self.llm_tokenizer.pad_token_id, 
                bos_token_id=self.llm_tokenizer.bos_token_id,
            )
        
        generated_texts = self.llm_tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=True,
            add_special_tokens=False,
        )
        if self.decode_utt_by_utt:
            self.pre_utt_pred_history.append(generated_texts[0])
        target_texts = texts

        with open(f"{self.exp_dir}/{self.exp_name}/{self.test_basename}_infer_ref.txt", "a", encoding="utf-8") as ref_file, \
            open(f"{self.exp_dir}/{self.exp_name}/{self.test_basename}_infer_hyp.txt", "a", encoding="utf-8") as hyp_file:
            
            for i, (ref, hyp, key) in enumerate(zip(target_texts, generated_texts, audio_paths)):
                sentence_id = key
                ref_file.write(f"{sentence_id} {ref}\n")
                hyp_file.write(f"{sentence_id} {hyp}\n")
        
        print(f"\ngenera_texts: {generated_texts}\ntarget_texts: {target_texts}\n")
        this_wer = compute_wer(generated_texts, target_texts)
        print(f"\nBatch {batch_idx} WER: {this_wer:.2f}%\n")
        wer_metric = wer(target_texts, generated_texts)
        
        self.log("test/wer", wer_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        del embeds, atts, generated_ids
        return {"test_wer": wer_metric}

    def on_validation_epoch_start(self):
        """Select two random validation samples to log for each epoch."""
        self.selected_samples_for_logging = random.sample(range(self.num_validation_samples), 2)


def compute_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.
    Copied from https://github.com/X-LANCE/SLAM-LLM/blob/main/src/slam_llm/utils/metric.py
    Args:
        pad_outputs (LongTensor): Prediction tensors (B, Lmax).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return numerator.float() / denominator.float()