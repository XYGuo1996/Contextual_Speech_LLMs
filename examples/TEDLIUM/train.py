from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from model.trainer import SpeechLLMLightning
from model.dataset import MyCollator, InstructionalAudioDataset
from lightning.pytorch.strategies import DDPStrategy

import torch.utils.data as data_utils
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback, DeviceStatsMonitor
import json
import argparse
import torch
import os

from model.encoder.encoder import get_audio_encoder, TransformerAudioEnoder
from model.projector.connector import get_connector, LinearConnector, LinearPoolConnector, CNNConnector
from model.llm import get_llm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# torch.set_float32_matmul_precision('high')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Config file path')
    parser.add_argument('--exp_dir', required=True, type=str, help='')
    parser.add_argument('--exp_name', required=True, type=str, help='Experiment name')
    parser.add_argument('--max_lr', type=float, required=False, help='Number of GPUs to use')
    parser.add_argument('--train_dataset', type=str, help='Path to training dataset JSON file')
    parser.add_argument('--valid_dataset', type=str, help='Path to validation dataset JSON file')
    parser.add_argument('--train_batch_size', required=False, type=int, default=-1, help='See config for all options')
    parser.add_argument('--no_context_prob', required=False, type=float, default=0)
    parser.add_argument('--pretrained_model_path', required=False, type=str, default=None, help='the pretrained connector model path')
    args, options = parser.parse_known_args()
    
    with open(args.config) as f:
        model_config = json.load(f)
    model_config["exp_dir"] = args.exp_dir
    model_config["exp_name"] = args.exp_name
    if args.max_lr:
        model_config["max_lr"] = args.max_lr
    if args.train_dataset:
        model_config["train_dataset"] = args.train_dataset
    if args.valid_dataset:
        model_config["valid_dataset"] = args.valid_dataset
    if args.train_batch_size != -1:
        model_config["train_batch_size"] = args.train_batch_size
    if args.pretrained_model_path is not None:
        model_config["pretrained_model_path"] = args.pretrained_model_path
    
    run_dir = os.path.join(model_config["exp_dir"], model_config["exp_name"])
    os.makedirs(run_dir, exist_ok=True)

    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=tb_dir, name=".")

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.manual_seed(model_config.get('seed'))
    adapter_name = model_config.get('adapter_name')
    train_dataset = InstructionalAudioDataset(
        json_file=model_config['train_dataset'],
        mode='train',
        no_context_prob=args.no_context_prob,
    )
    val_dataset = InstructionalAudioDataset(
        json_file=model_config['valid_dataset'], 
        mode='valid',
        no_context_prob=args.no_context_prob,
    )
    
    model = SpeechLLMLightning(model_config, num_validation_samples=len(val_dataset))

    my_collator = MyCollator(model.llm_tokenizer)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=model_config['train_batch_size'], shuffle=True, collate_fn=my_collator, num_workers=4)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=model_config['valid_batch_size'], shuffle=False, collate_fn=my_collator, num_workers=4)

    device_monitor = DeviceStatsMonitor()

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, filename='{epoch}-{step}-{val-acc:.4f}', save_top_k=1, monitor="val-acc", save_last=True, mode="max")
    early_stop_callback = EarlyStopping(monitor='val-loss', patience=5, mode='min')
    trainer = Trainer(
            accelerator='gpu',
            max_epochs=model_config['max_epochs'],
            devices=model_config['gpus'],
            strategy=DDPStrategy(find_unused_parameters=True),
            limit_train_batches=None,
            limit_val_batches=1.0,
            log_every_n_steps=10,
            val_check_interval=0.25,
            enable_checkpointing=True,
            callbacks=[checkpoint_callback, device_monitor, early_stop_callback],
            fast_dev_run=False,
            logger=logger,
            accumulate_grad_batches=model_config['grad_accumulate_steps'],
            precision="bf16-mixed",
    )
    trainer.fit(model, train_loader, val_loader)

