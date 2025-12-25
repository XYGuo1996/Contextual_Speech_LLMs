import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class HubertEncoder:

    @classmethod
    def load(cls, args):
        import fairseq
        print(f'args["audio_encoder_path"]: {args["audio_encoder_path"]}')
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args["audio_encoder_path"]])
        model = models[0]
        if args["audio_encoder_type"] == "pretrain":
            pass
        elif args["audio_encoder_type"] == "finetune":
            model.w2v_encoder.proj = None
            model.w2v_encoder.apply_mask = False
        else:
            assert args["audio_encoder_type"] in ["pretrain", "finetune"], "input_type must be one of [pretrain, finetune]" 
        return model