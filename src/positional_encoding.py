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
from src.utils import append_dims

class Sinusoidal(nn.Module):
    def __init__(self,embedding_dims, embedding_min_frequency=1,
                 embedding_max_frequency=1000, img_shape=None, device="cuda"):
        super().__init__()
        self.embedding_dims=embedding_dims
        self.device=device
        self.img_shape=img_shape
        if not isinstance(embedding_min_frequency, T.Tensor):
            embedding_min_frequency = T.tensor(embedding_min_frequency)
            
        if not isinstance(embedding_max_frequency, T.Tensor):
            embedding_max_frequency = T.tensor(embedding_max_frequency)

        self.register_buffer("min_value", embedding_min_frequency)
        self.register_buffer("max_value", embedding_max_frequency)

        if img_shape is not None:
            self.create_embedding()

        frequencies = T.arange(self.embedding_dims//2, device=self.device)
        frequencies = T.exp(frequencies)
        
        self.register_buffer("frequencies", frequencies)
        


    # def sinusoidal_embedding(self,x):
    #     angular_speeds = 2.0 * T.pi * self.frequencies
    #     # if len(x.shape)>=3:
    #     #     embeddings = T.concat(
    #     #         [T.sin(angular_speeds * x),
    #     #         T.cos(angular_speeds * x)],
    #     #         axis=3
    #     #     )
    #     #     embeddings = embeddings.reshape(x.shape[0],self.embedding_dims,1,1)
    #     # else:
    #     embeddings = T.concat(
    #         [T.sin(angular_speeds * x),
    #         T.cos(angular_speeds * x)], axis=-1
    #     )
    #     return embeddings
    def sinusoidal_embedding(self,x):
        # angular_speeds = 2.0 * T.pi * self.frequencies
        x = (x - self.min_value) * math.pi / (self.max_value - self.min_value)
        x = x*self.frequencies
        embeddings = T.concat(
            [x.cos(), x.sin()], axis=-1
        )
        return embeddings
    
    def forward(self, x, n_dims:int):
        "iterative embedding for changing noise"
        return append_dims(self.sinusoidal_embedding(x), n_dims)

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

# Embeddings

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', T.randn([out_features // 2, in_features]) * std)

    def forward(self, input, n_dims:int):
        angular_speeds = 2 * math.pi * input @ self.weight.T
        # angular_speeds = angular_speeds.transpose(-1, -2)
        return append_dims(T.cat([angular_speeds.cos(), angular_speeds.sin()], dim=-1),
                           n_dims)


if __name__ == "__main__":
    # %matplotlib widget
    import matplotlib.pyplot as plt
    inputs = 1/4*T.linspace(np.log(0.002), np.log(80), 10_000).view(-1,1)
    # inputs = T.linspace(1, 1000, 1024).view(-1,1)
    # inputs = T.linspace(0, 81, 1024).view(-1,1)
    # inputs = T.tensor([1.0, 1.1, 2.0, 2.1]).view(-1,1)
    n_features = 32
    if True:
        ff = FourierFeatures(1,2*n_features, std=1)
        y = ff(inputs)
    elif True:
        time_embed = sinusoidal(n_features, 1, 1000, device="cpu")
        y = time_embed(inputs)


    plt.figure()
    for i in range(n_features):
        plt.plot(inputs.numpy(), y.numpy()[:,i])
        # plt.figure()
        # plt.plot(y.numpy()[:,i], y.numpy()[:,i+n_features])
    # plt.figure()
    # plt.plot(inputs.numpy(), y.numpy().mean(1))


    # #
    # plt.figure()
    # x = np.linspace(0, 2*np.pi)
    # plt.plot(x, np.sin(1/2*x))