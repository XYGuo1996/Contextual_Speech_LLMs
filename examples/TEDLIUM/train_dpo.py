
import argparse
import copy
import json
import os
from typing import List, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from model.trainer import SpeechLLMLightning
from model.dataset import InstructionalAudioDataset


class DPOAudioDataset(Dataset):
    "Audio dataset for DPO training."

    def __init__(self, json_file: str, no_context_prob: float = 0.0):
        self.base_dataset = InstructionalAudioDataset(
            json_file=json_file,
            mode="train",
            no_context_prob=no_context_prob,
        )
        self.data_frame = self.base_dataset.data_frame

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        waveform, wavform_normalize, _text, task_prompt, prompt_template, audio_path = self.base_dataset[idx]
        row = self.data_frame[idx]
        chosen = row.get("chosen", row.get("text", ""))
        rejected = row.get("rejected", "")
        return waveform, wavform_normalize, task_prompt, prompt_template, audio_path, chosen, rejected


class DPOCollator:
    "Collator for DPO batches."

    def __call__(self, batch: List[Tuple]):
        import torch
        from torch.nn.utils.rnn import pad_sequence

        waveforms, wavform_normalizes = [], []
        task_prompts, prompt_templates, audio_paths = [], [], []
        chosen_texts, rejected_texts = [], []
        audio_lengths: List[int] = []

        for (
            waveform,
            wavform_normalize,
            task_prompt,
            prompt_template,
            audio_path,
            chosen,
            rejected,
        ) in batch:
            waveforms.append(waveform.squeeze(0))
            wavform_normalizes.append(wavform_normalize.squeeze(0))
            task_prompts.append(task_prompt)
            prompt_templates.append(prompt_template)
            audio_paths.append(audio_path)
            chosen_texts.append(chosen)
            rejected_texts.append(rejected)
            audio_lengths.append(waveform.squeeze(0).shape[0])

        audio_mask = torch.zeros(len(batch), max(audio_lengths))
        for i, length in enumerate(audio_lengths):
            audio_mask[i, :length] = 1

        batch_waveform = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
        batch_waveform_normalize = pad_sequence(wavform_normalizes, batch_first=True, padding_value=0.0)

        return (
            batch_waveform,
            batch_waveform_normalize,
            audio_mask,
            task_prompts,
            prompt_templates,
            audio_paths,
            torch.tensor(audio_lengths),
            chosen_texts,
            rejected_texts,
        )


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    "Compute average log-probability of target tokens for each sequence."
    log_probs = torch.log_softmax(logits, dim=-1)
    log_probs = log_probs[:, :-1, :]
    labels = labels[:, 1:]

    mask = labels != ignore_index
    safe_labels = labels.masked_fill(~mask, 0).unsqueeze(-1)

    token_logps = log_probs.gather(dim=-1, index=safe_labels).squeeze(-1)
    token_logps = token_logps * mask

    lengths = mask.sum(dim=1).clamp_min(1)
    seq_logps = token_logps.sum(dim=1) / lengths
    return seq_logps


def dpo_loss(
    logp_policy_chosen: torch.Tensor,
    logp_policy_rejected: torch.Tensor,
    logp_ref_chosen: torch.Tensor,
    logp_ref_rejected: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    "Direct Preference Optimization loss."
    pi_diff = logp_policy_chosen - logp_policy_rejected
    ref_diff = logp_ref_chosen - logp_ref_rejected
    logits = beta * (pi_diff - ref_diff)
    return -F.logsigmoid(logits).mean()


class SpeechLLMDPO(SpeechLLMLightning):
    def __init__(self, model_config, policy_ckpt_path, beta=0.1, learning_rate=5e-6):
        # Initialize parent with num_validation_samples=0
        super().__init__(model_config, num_validation_samples=0)
        self.save_hyperparameters()
        self.beta = beta
        self.policy_ckpt_path = policy_ckpt_path
        self.learning_rate = learning_rate
        
        print(f"Loading base policy model from {self.policy_ckpt_path}...")
        self.load_pretrained_model(self.policy_ckpt_path)
        if isinstance(self.llm_model, PeftModel):
            self.llm_model = self.llm_model.merge_and_unload()
            
        for p in self.llm_model.parameters():
            p.requires_grad = False
        for p in self.connector.parameters():
            p.requires_grad = False
        
        if hasattr(self.llm_model, "gradient_checkpointing_enable"):
            self.llm_model.gradient_checkpointing_enable()
        
        dpo_config = LoraConfig(
            r=model_config.get("lora_r", 8),
            lora_alpha=model_config.get("lora_alpha", 32),
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if not isinstance(self.llm_model, PeftModel):
            print("Model is not PEFT. Converting to PEFT with 'dpo_adapter'...")
            self.llm_model = get_peft_model(self.llm_model, dpo_config, adapter_name="dpo_adapter")
        else:
            print("Model is PEFT. Adding 'dpo_adapter'...")
            self.llm_model.add_adapter("dpo_adapter", dpo_config)
            
        trainable_params = [n for n, p in self.llm_model.named_parameters() if p.requires_grad]
        print(f"Trainable parameters in Policy LLM: {len(trainable_params)}")

    def configure_optimizers(self):
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.llm_model.parameters()),
            lr=self.learning_rate,
            weight_decay=0.0,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        (batch_waveform, batch_waveform_normalize, audio_mask, task_prompts, 
         prompt_templates, audio_paths, audio_lengths, chosen_texts, rejected_texts) = batch

        embeds_c, atts_c, labels_c = self.merge_input_ids_with_speech_features_train(
            batch_waveform, batch_waveform_normalize, audio_mask, chosen_texts, 
            task_prompts, prompt_templates, audio_lengths
        )
        
        embeds_r, atts_r, labels_r = self.merge_input_ids_with_speech_features_train(
            batch_waveform, batch_waveform_normalize, audio_mask, rejected_texts, 
            task_prompts, prompt_templates, audio_lengths
        )

        out_c_pol = self.llm_model(inputs_embeds=embeds_c, attention_mask=atts_c)
        logp_c_pol = sequence_logprob(out_c_pol.logits, labels_c)

        out_r_pol = self.llm_model(inputs_embeds=embeds_r, attention_mask=atts_r)
        logp_r_pol = sequence_logprob(out_r_pol.logits, labels_r)

        with torch.no_grad():
            with self.llm_model.disable_adapter():
                out_c_ref = self.llm_model(inputs_embeds=embeds_c, attention_mask=atts_c)
                logp_c_ref = sequence_logprob(out_c_ref.logits, labels_c)

                out_r_ref = self.llm_model(inputs_embeds=embeds_r, attention_mask=atts_r)
                logp_r_ref = sequence_logprob(out_r_ref.logits, labels_r)

        loss = dpo_loss(logp_c_pol, logp_r_pol, logp_c_ref, logp_r_ref, self.beta)

        with torch.no_grad():
            avg_pi_diff = (logp_c_pol - logp_r_pol).mean()
            avg_ref_diff = (logp_c_ref - logp_r_ref).mean()
            
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("pi_diff", avg_pi_diff, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("ref_diff", avg_ref_diff, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def on_validation_epoch_start(self):
        """Override parent method to prevent sampling from empty validation set."""
        pass

    def validation_step(self, batch, batch_idx):
        (batch_waveform, batch_waveform_normalize, audio_mask, task_prompts, 
         prompt_templates, audio_paths, audio_lengths, chosen_texts, rejected_texts) = batch

        embeds_c, atts_c, labels_c = self.merge_input_ids_with_speech_features_train(
            batch_waveform, batch_waveform_normalize, audio_mask, chosen_texts, 
            task_prompts, prompt_templates, audio_lengths
        )
        embeds_r, atts_r, labels_r = self.merge_input_ids_with_speech_features_train(
            batch_waveform, batch_waveform_normalize, audio_mask, rejected_texts, 
            task_prompts, prompt_templates, audio_lengths
        )

        out_c_pol = self.llm_model(inputs_embeds=embeds_c, attention_mask=atts_c)
        logp_c_pol = sequence_logprob(out_c_pol.logits, labels_c)

        out_r_pol = self.llm_model(inputs_embeds=embeds_r, attention_mask=atts_r)
        logp_r_pol = sequence_logprob(out_r_pol.logits, labels_r)

        with torch.no_grad():
            with self.llm_model.disable_adapter():
                out_c_ref = self.llm_model(inputs_embeds=embeds_c, attention_mask=atts_c)
                logp_c_ref = sequence_logprob(out_c_ref.logits, labels_c)

                out_r_ref = self.llm_model(inputs_embeds=embeds_r, attention_mask=atts_r)
                logp_r_ref = sequence_logprob(out_r_ref.logits, labels_r)

        loss = dpo_loss(logp_c_pol, logp_r_pol, logp_c_ref, logp_r_ref, self.beta)

        avg_pi_diff = (logp_c_pol - logp_r_pol).mean()
        avg_ref_diff = (logp_c_ref - logp_r_ref).mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_pi_diff", avg_pi_diff, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_ref_diff", avg_ref_diff, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def on_validation_epoch_end(self):
        """Override parent method to prevent sampling from empty validation set."""
        pass

    def on_save_checkpoint(self, checkpoint) -> None:
        state_dict = {}
        if isinstance(self.llm_model, PeftModel):
             from peft import get_peft_model_state_dict
             state_dict["dpo_adapter"] = get_peft_model_state_dict(self.llm_model, adapter_name="dpo_adapter")
        
        checkpoint["state_dict"] = state_dict


def parse_args():
    parser = argparse.ArgumentParser(description="DPO fine-tuning for SpeechLLM on TEDLIUM.")
    parser.add_argument("--config", type=str, required=True, help="Config file path used for SFT.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment root directory.")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name for DPO run.")
    parser.add_argument("--dpo_dataset", type=str, required=True, help="JSONL file with DPO pairs (chosen/rejected).")
    parser.add_argument("--val_dataset", type=str, required=False, help="Validation JSONL file with DPO pairs.")
    parser.add_argument("--policy_ckpt", type=str, required=True, help="Path to SFT checkpoint used to initialize policy/ref.")
    parser.add_argument("--max_lr", type=float, default=None, help="Optional override for max_lr in config.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for DPO fine-tuning.")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Per-device batch size for DPO.")
    parser.add_argument("--val_batch_size", type=int, default=2, help="Per-device batch size for DPO validation.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of DPO epochs.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log training stats every N steps.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many total optimization steps.")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--no_context_prob", type=float, default=0.0, help="no_context_prob used when loading DPO dataset.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use.")
    return parser.parse_args()


def train_dpo(args):
    with open(args.config, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    model_config["exp_dir"] = args.exp_dir
    model_config["exp_name"] = args.exp_name
    if args.max_lr:
        model_config["max_lr"] = args.max_lr
    
    model_config["train_batch_size"] = args.train_batch_size
    
    run_dir = os.path.join(model_config["exp_dir"], model_config["exp_name"])
    os.makedirs(run_dir, exist_ok=True)
    
    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=tb_dir, name=".")

    checkpoint_dir = os.path.join(run_dir, "checkpoints_dpo")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    monitor_metric1 = "val_loss" if args.val_dataset else "loss"
    monitor_metric2 = "val_pi_diff" if args.val_dataset else "pi_diff"
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{step}-{' + monitor_metric1 + ':.4f}-{' + monitor_metric2 + ':.4f}',
        save_top_k=-1,
        monitor=monitor_metric1,
        mode="min",
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    model = SpeechLLMDPO(
        model_config=model_config,
        policy_ckpt_path=args.policy_ckpt,
        beta=args.beta,
        learning_rate=args.learning_rate
    )

    dpo_dataset = DPOAudioDataset(json_file=args.dpo_dataset, no_context_prob=args.no_context_prob)
    collator = DPOCollator()
    train_loader = DataLoader(
        dpo_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
    )
    
    val_loader = None
    if args.val_dataset:
        val_dpo_dataset = DPOAudioDataset(json_file=args.val_dataset, no_context_prob=args.no_context_prob)
        val_loader = DataLoader(
            val_dpo_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=4,
            drop_last=True,
        )

    trainer = Trainer(
        accelerator="gpu",
        devices=args.gpus,
        strategy=DDPStrategy(find_unused_parameters=True) if args.gpus > 1 else "auto",
        max_epochs=args.num_epochs,
        max_steps=args.max_steps,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=args.max_grad_norm,
        val_check_interval=50 if args.val_dataset else None,
        accumulate_grad_batches=16,
        log_every_n_steps=args.logging_steps,
        precision="bf16",
    )

    print("Starting DPO training with Lightning...")
    trainer.fit(model, train_loader, val_dataloaders=val_loader)
    print("DPO training finished.")


if __name__ == "__main__":
    args = parse_args()
    train_dpo(args)
