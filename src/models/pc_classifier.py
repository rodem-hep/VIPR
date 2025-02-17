'Transformer classifier'
import math
from typing import Union, Callable
from functools import partial
import torch as T
import torch.nn as nn
import hydra
import torchvision as TV
import numpy as np
from omegaconf import OmegaConf

# internal 
from tools.tools.transformers.transformer import DenseNetwork
from tools.tools.modules import IterativeNormLayer

class PCClassifier(nn.Module):
    def __init__(self, vkq_dims, num_layers:int =1,
                 encoder_cfg:dict=None,
                 dense_cfg:dict=None,
                 upscale_dims:int=64, 
                 device:str="cuda", **kwargs):

        super().__init__()
        self.vkq_dims = vkq_dims
        self.upscale_dims= upscale_dims
        self.encoder_cfg=encoder_cfg
        self.device = device
        self.dense_cfg=dense_cfg if dense_cfg!=None else {}
        self.num_layers = num_layers
        self.output_dims = kwargs.get('output_dims', self.vkq_dims)
        self.ctxt_dims = kwargs.get('ctxt_dims', 0)
        self.ctxt_upscale_dims = kwargs.get('ctxt_upscale_dims', 0)

        self.inpt_encoder_layers = nn.ModuleList([])

        self.get_network()

    def get_network(self) -> None:
        # init cnts
        self.pc_norm = IterativeNormLayer((1,self.vkq_dims))
        self.init_dense = nn.Linear(self.vkq_dims, self.upscale_dims)

        # ctxt scalar
        if self.ctxt_dims> 0:
            self.scalars_norm = IterativeNormLayer((1,self.ctxt_dims))
            self.init_scalars_ctxt = DenseNetwork(self.ctxt_dims, 
                                                  self.ctxt_upscale_dims,**self.dense_cfg)


        for _ in range(self.num_layers):
            self.inpt_encoder_layers.append(self.encoder_cfg())

        # self.downscale_conv  = DenseNetwork(self.upscale_dims+self.ctxt_upscale_dims,self.output_dims,
        #                                     **self.dense_cfg)
        self.downscale_conv  = nn.Linear(self.upscale_dims,self.output_dims)


    def count_trainable_parameters(self):
        sum_trainable = np.sum([i.numel() for i in self.parameters() if i.requires_grad])
        return sum_trainable
            
    def forward(self, cnts: T.Tensor, mask:T.Tensor=None, scalars:T.Tensor=None) -> T.Tensor:
        if scalars is None:
            scalars={}
            
        # add noise_timestamp to ctxt
        if scalars is not None:
            scalars = self.scalars_norm(scalars)
            ctxt_scalars = self.init_scalars_ctxt(scalars)

        # network starts
        # simple MLP
        cnts = self.pc_norm(cnts, mask=mask)
        cnts = self.init_dense(cnts)
        
        original_cnts = cnts.clone()

        # transformers
        for i in range(self.num_layers):

            # self attention for input
            cnts = self.inpt_encoder_layers[i](cnts, mask_vk=mask,
                                                        ctxt=ctxt_scalars)

        # this addition is from puppiml, and probably forces the order of the particles
        if False:
            # used for the first classifier version - PUPPIML_top_jets_pileup_jet_2024_12_07_13_28_41_399052
            cnts = cnts+original_cnts

        # downscale output to same output features
        output = self.downscale_conv(cnts)
        # output = self.downscale_conv(cnts, ctxt_scalars)

        return output.squeeze(-1)
    