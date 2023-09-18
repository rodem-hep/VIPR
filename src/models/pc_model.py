"Transformers"
import math
from typing import Union
import torch as T
import torch.nn as nn
import torchvision as TV
import numpy as np
from omegaconf import OmegaConf

# internal 
from tools import misc
import src.positional_encoding as pe
from src.models.transformer import (TransformerDecoder, Perceiver, DenseNetwork)

class PileUpRemoval(nn.Module):
    def __init__(self, vkq_dims, ctxt_dims:Union[int, dict]=None,
                 trans_cfg:dict=None, dense_cfg:dict=None,
                 upscale_dims:int=64, embedding_cfg=None,
                 device:str="cuda"):
        super().__init__()
        self.vkq_dims = vkq_dims
        self.ctxt_dims = ctxt_dims
        self.upscale_dims= upscale_dims
        self.trans_cfg=trans_cfg
        self.device = device
        self.dense_cfg=dense_cfg if dense_cfg!=None else {}
        self.embedding_cfg = embedding_cfg

        self.trans_encoder_layers = None
        self.init_conv_ctxt = None

        self.get_network()

    def get_network(self) -> None:

        if self.embedding_cfg is not None:
            # self.embedding = pe.FourierFeatures(1, self.embedding_dims)
            self.embedding = pe.Sinusoidal(**self.embedding_cfg)
            self.embedding_dims = self.embedding_cfg.get("embedding_dims", 0)

        self.init_conv = DenseNetwork(self.vkq_dims+self.upscale_dims,
                               self.upscale_dims, **self.dense_cfg)
        if "cnts" in self.ctxt_dims:
            self.init_conv_ctxt = DenseNetwork(self.ctxt_dims["cnts"][-1]+self.upscale_dims,
                                self.upscale_dims,**self.dense_cfg)
        if "scalars" in self.ctxt_dims:
            self.init_scalars_ctxt = DenseNetwork(self.ctxt_dims["scalars"]+self.embedding_dims,
                                self.upscale_dims,**self.dense_cfg)
            
        
        if self.trans_cfg is not None:
            if isinstance(self.ctxt_dims, int):
                raise TypeError("ctxt_dims has to be a dict. "+
                                "Keyname for FiLM is scalars")
            elif "film_cfg" in self.trans_cfg:
                self.trans_cfg["film_cfg"]["ctxt_size"]+=self.ctxt_dims.get("scalars", 0)
        
        if "pcivr_cfg" in self.trans_cfg:
            self.trans_encoder_layers = Perceiver(**self.trans_cfg)
        else:
            self.trans_encoder_layers = TransformerDecoder(**self.trans_cfg)
        
        self.downscale_conv  = DenseNetwork(2*self.upscale_dims,self.vkq_dims,
                                     zeroed=False, **self.dense_cfg)

    @T.no_grad()
    def ema(self, state_dict, ema_ratio):
        ema_state_dict = self.state_dict()
        for (key, weight), (em_key, ema_para) in zip(state_dict.items(),
                                                     ema_state_dict.items()):
            ema_state_dict[em_key] = ema_ratio * ema_para + (1 - ema_ratio) * weight

        self.load_state_dict(ema_state_dict)

    def count_trainable_parameters(self):
        sum_trainable = np.sum([i.numel() for i in self.parameters() if i.requires_grad])
        return sum_trainable
        
            
    def forward(self, input_vkq: T.Tensor, noise_variances:T.Tensor=None,
                mask:T.Tensor=None, ctxt:T.Tensor=None
                ) -> T.Tensor:
        input_vkq, mask = (input_vkq.to(self.device), mask.to(self.device))
        if ctxt is None:
            ctxt={}

        if noise_variances is not None:
            # calculate embedding
            ctxt_scalars = self.embedding(noise_variances.to(self.device),
                                          len(input_vkq.shape))
            
        # add noise_timestamp to ctxt
        if "scalars" in ctxt:
            ctxt_scalars = self.init_scalars_ctxt(ctxt_scalars,
                                                  ctxt["scalars"].to(self.device))

        if "cnts" in ctxt:
            ctxt_cnts = ctxt["cnts"].clone().to(self.device)
            ctxt_mask = ctxt["mask"].to(self.device)
            input_ctxt = self.init_conv_ctxt(ctxt_cnts, ctxt_scalars)
        else:
            input_ctxt = input_vkq.clone()
            ctxt_mask = mask.clone()
            
        # network starts
        # simple MLP
        input_vkq = self.init_conv(input_vkq, ctxt_scalars)
            
        # transformers
        # if (self.trans_cfg is not None) & False:
        #     input_vkq = self.trans_encoder_layers(input_ctxt, ctxt_mask,
        #                                                 input_vkq, mask,
        #                                                 ctxt=ctxt_scalars)
        # else:
        input_vkq = self.trans_encoder_layers(input_vkq, input_ctxt,
                                                mask_vk=ctxt_mask,
                                                ctxt=ctxt_scalars)

        # input_vkq = self.trans_encoder_layers(input_ctxt, input_ctxt,
        #                                       input_ctxt, mask_vk=ctxt_mask,
        #                                             #    mask_q=mask
        #                                                 )
        # downscale output to same output features
        return self.downscale_conv(input_vkq, ctxt_scalars)

