"Different pytorch modules"
import math
import torch as T
import torch.nn as nn
import torchvision as TV
import numpy as np
from omegaconf import OmegaConf
# from src.models.transformer import VisionTransformerLayer

# internal 
from tools import misc
from tools.discriminator import get_densenet

class FiLM(nn.Module):
    def __init__(self, ctxt_size, lst_channel, dense_config, use_on_image:bool=True, device="cuda"):
        super().__init__()
        self.ctxt_size = ctxt_size
        self.device=device
        self.lst_channel=lst_channel
        self.use_on_image=use_on_image
        network = get_densenet(input_dim=self.ctxt_size,
                                output_dim=2*(np.sum(lst_channel+lst_channel[:-1])),
                                **dense_config)
        self.network = nn.Sequential(*network)
        self.to(device)

    def forward(self,ctxt):
        # thought the network
        film_parameters = self.network(ctxt)
        
        # change shape to BxCx1x1
        #dim for downscale network
        film_parameters = film_parameters.reshape(len(film_parameters), -1, 2)
        dims = list(np.cumsum(self.lst_channel))
        dims.insert(0,0)
        
        #dim for upscale network
        dims +=list(dims[-1]+np.cumsum(self.lst_channel[:-1][::-1]))

        if self.use_on_image:
            self.film_parameters = [film_parameters[:,i:j, :].unsqueeze(3)
                                    for i,j in zip(dims[:-1], dims[1:])]
        else:
            self.film_parameters = [film_parameters.transpose(-1, -2)[:, :, i:j]
                                    for i,j in zip(dims[:-1], dims[1:])]

        # create iterator
        self._film_parameters_iter = iter(self.film_parameters)

    def __next__(self):
        return next(self._film_parameters_iter)

class Gate(nn.Module):
    def __init__(self, input_shape, gate_shape, act_func="relu",
                 device="cuda"):
        super().__init__()
        self.input_shape=input_shape
        self.gate_shape=gate_shape
        self.act_func=act_func
        self.device=device
        self.get_network()

    def get_network(self):

        # downscale res img to half size
        self.input_info_conv =  nn.Sequential(
            nn.Conv2d(self.input_shape, self.input_shape,
                      kernel_size=1),
            nn.BatchNorm2d(self.input_shape))

        # gate conv the conditional info
        self.gate_info_conv = nn.Sequential(
            nn.Conv2d(self.gate_shape, self.input_shape,
                      kernel_size=1),
            nn.BatchNorm2d(self.input_shape))

        self.gating= nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(self.input_shape,self.input_shape,kernel_size=1),
            nn.Sigmoid(),
            # nn.Upsample(scale_factor=2,mode="nearest")
                                     )
        self.to(self.device)
        
    def forward(self,input_img,__input_img, gate):
        # Do not use __input_img, just a duplicate of input_img
        x = self.input_info_conv(input_img)
        gate = self.gate_info_conv(gate)
        x = self.gating(T.add(x, gate))

        return input_img*x

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, block_depth, dropout=0.1,
                 img_dim=None, zero_init=True) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size=output_size
        self.block_depth = block_depth
        self.zero_init=zero_init
        self.dropout=dropout
        self.img_dim=img_dim
        self.get_network()
        
    def get_layer(self, input_size, output_size):
        return nn.Sequential(
                    nn.BatchNorm2d(input_size, output_size, affine=False), 
                    nn.LeakyReLU(),
                    nn.Dropout(p=self.dropout),
                    nn.Conv2d(input_size, output_size, kernel_size=3, padding="same"))

    def get_network(self):
        self.skip_connection = nn.Conv2d(self.input_size, self.output_size, kernel_size=1)
                
        self.layers = nn.ModuleList([self.get_layer(self.input_size, 
                                                    self.output_size)])
        for _ in range(self.block_depth-1):
            self.layers.append(self.get_layer(self.output_size, self.output_size))

        # if self.img_dim is not None:
        #     self.layers.append(VisionTransformerLayer(
        #         img_shape=self.img_dim,
        #         n_channels=self.output_size, n_patches=8,
        #         attn_heads=8
        #     ))
        # else:
        self.layers[-1][-1].weight.data.fill_(0.00)
        self.layers[-1][-1].bias.data.fill_(0.00)
            
    def forward(self,x, ctxt=None):
        residual_connection = self.skip_connection(x)
        for layer in self.layers:
            x = layer(x)
            if (ctxt is not None):
                # MxNxCxB times CxB
                x =  ctxt[:,:,0:1]*x+ ctxt[:,:,1:2]
        return x+residual_connection
