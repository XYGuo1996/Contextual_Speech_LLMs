import torch
import importlib
from torch import nn
from transformers import AutoModel
from model.encoder.whisper_encoder import OpenAIWhisperEncoder
from model.encoder.hubert_encoder import HubertEncoder

def get_audio_encoder(args,name=None,finetune=None):
    if "hubert" in args["audio_encoder_name"]:
        return HubertEncoder.load(args)
    elif args["audio_encoder_name"] == "microsoft/wavlm-large":
        return TransformerAudioEnoder(model_name=args["audio_encoder_name"], finetune=args["finetune_encoder"])
    elif "whisper" in args["audio_encoder_name"]:
        return OpenAIWhisperEncoder(whisper_model=args["audio_encoder_version"], download_dir=args["audio_encoder_path"], finetune=args["finetune_encoder"])
    else:
        raise NotImplementedError


class TransformerAudioEnoder(nn.Module):
    def __init__(self, model_name='facebook/hubert-xlarge-ll60k', model_path=None, finetune=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if model_path is not None:
            self.encoder.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        for param in self.encoder.parameters():
            param.requires_grad = finetune
            
    def forward(self, x):
        return self.encoder(x).last_hidden_state

