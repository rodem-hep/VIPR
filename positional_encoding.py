"Positional encoding"
import math
import torch as T
import torch.nn as nn
import torchvision as TV
import numpy as np
from omegaconf import OmegaConf

# internal 
from tools import misc
from tools.discriminator import DenseNet

def sinusoidal_embedding(x, embedding_min_frequency,
                         embedding_max_frequency,
                         embedding_dims,
                         device="cuda"):
    frequencies = T.exp(
        T.linspace(
            np.log(embedding_min_frequency),
            np.log(embedding_max_frequency),
            embedding_dims // 2, device=device
        )
    )
    angular_speeds = 2.0 * T.pi * frequencies
    if len(x.shape)>=3:
        embeddings = T.concat(
            [T.sin(angular_speeds * x),
            T.cos(angular_speeds * x)], axis=3
        )
        embeddings = embeddings.reshape(x.shape[0],embedding_dims,1,1)
    else:
        embeddings = T.concat(
            [T.sin(angular_speeds * x),
            T.cos(angular_speeds * x)], axis=1
        )
    return embeddings

class sinusoidal:
    def __init__(self, embedding_max_frequency, embedding_dims, img_shape=None, device="cuda"):
        self.embedding_min_frequency = 1.0
        self.embedding_max_frequency=embedding_max_frequency
        self.embedding_dims=embedding_dims
        self.device=device
        self.img_shape=img_shape
        if img_shape is not None:
            self.create_embedding()
    
    def __call__(self, x):
        "iterative embedding for changing noise"
        return sinusoidal_embedding(x,
                                    self.embedding_min_frequency,
                                    self.embedding_max_frequency,
                                    self.embedding_dims,
                                    device=self.device
                                    )

class PositionalEncoding(nn.Module):
    # from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = T.arange(max_len).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = T.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = T.sin(position * div_term)
        pe[:, 0, 1::2] = T.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
