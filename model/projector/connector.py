import torch
from torch import nn

def get_connector(adapter_name, adapter_conf):
    if adapter_name == 'pooling-adapter':
        return PoolingAdapter(
            input_dim=adapter_conf['input_dim'], 
            hidden_dim=adapter_conf['hidden_dim'], 
            output_dim=adapter_conf['llm_dim'], 
            pooling_factor=adapter_conf['pooling_factor']
        )
    else:
        raise NotImplementedError

class ConcatPooling(nn.Module):
    """
    A module that perform pooling by concatenating the features of every pooling_factor frames.
    """

    def __init__(self, pooling_factor):
        super().__init__()
        self.pooling_factor = pooling_factor

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        if seq_len % self.pooling_factor != 0:
            x = x[:, : -(seq_len % self.pooling_factor), :]
        x = x.reshape(batch_size, seq_len // self.pooling_factor, input_dim * self.pooling_factor)
        return x

class PoolingAdapter(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers: int = 2,
        activation: str = "relu",
        pooling: str = "cat",
        pooling_factor: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim else input_dim
        self.num_layers = num_layers
        self.activation = activation
        self.pooling = pooling
        self.pooling_factor = pooling_factor

        if num_layers == 1:
            self.hidden_dim = output_dim

        if pooling == "cat":
            self.preprocess = nn.Sequential(
                ConcatPooling(pooling_factor), nn.Linear(self.input_dim * self.pooling_factor, self.hidden_dim)
            )
        else:
            self.preprocess = nn.Sequential(
                nn.AvgPool1d(pooling_factor, stride=pooling_factor), nn.Linear(input_dim, self.hidden_dim)
            )
        
        layers = []
        for _ in range(self.num_layers - 2):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.projector = nn.Sequential(*layers)
    
    def forward(self, audio_signal):
        outputs = self.preprocess(audio_signal)
        outputs = self.projector(outputs)
        return outputs
