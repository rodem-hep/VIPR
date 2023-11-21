"Transformers"
import math
from typing import Union
import torch as T
import torch.nn as nn
import hydra
import torchvision as TV
import numpy as np
from omegaconf import OmegaConf

# internal 
from tools import misc
import tools
from tools.transformers.transformer import (TransformerDecoder, Perceiver, DenseNetwork)
import src.positional_encoding as pe

class PCDiffusion(nn.Module):
    def __init__(self, vkq_dims, ctxt_dims:Union[int, dict]=None, num_layers:int =1,
                 decoder_cfg:dict=None, encoder_cfg:dict=None, dense_cfg:dict=None,
                 upscale_dims:int=64, embedding_cfg=None, skip_cnt:bool =False,
                 device:str="cuda"):
        super().__init__()
        self.vkq_dims = vkq_dims
        self.ctxt_dims = ctxt_dims
        self.upscale_dims= upscale_dims
        self.decoder_cfg=decoder_cfg
        self.encoder_cfg=encoder_cfg
        self.device = device
        self.skip_cnt=skip_cnt
        self.dense_cfg=dense_cfg if dense_cfg!=None else {}
        self.embedding_cfg = embedding_cfg
        self.embedding_dims = embedding_cfg.embedding_dims if self.embedding_cfg is not None else 0
        self.num_layers = num_layers

        self.ctxt_encoder_layers = nn.ModuleList([])
        self.inpt_encoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        self.init_conv_ctxt = None

        self.get_network()

    def get_network(self) -> None:
        
        # init cnts
        self.init_dense = DenseNetwork(self.vkq_dims+self.upscale_dims,
                                       self.upscale_dims, **self.dense_cfg)

        # ctxt cnts
        if "cnts" in self.ctxt_dims:
            self.init_conv_ctxt = DenseNetwork(
                self.ctxt_dims["cnts"][-1]+self.upscale_dims,
                self.upscale_dims,**self.dense_cfg)
        
        # ctxt scalar
        if ("scalars" in self.ctxt_dims) or (self.embedding_dims>0):
            self.full_ctxt_dims = self.ctxt_dims.get("scalars", 0)+self.embedding_dims
            self.init_scalars_ctxt = DenseNetwork(self.full_ctxt_dims,
                                                  self.upscale_dims,**self.dense_cfg)

        if self.decoder_cfg is not None:
            if isinstance(self.ctxt_dims, int):
                raise TypeError("ctxt_dims has to be a dict. "+
                                "Keyname for FiLM is scalars")
            elif "film_cfg" in self.decoder_cfg.keywords:
                self.decoder_cfg["film_cfg"]["ctxt_size"]+=self.ctxt_dims.get("scalars", 0)
        
        if self.encoder_cfg is not None:

            for _ in range(self.num_layers):

                if self.decoder_cfg is not None:
                    self.decoder_layers.append(self.decoder_cfg())

                if ("cnts" in self.ctxt_dims):
                    self.ctxt_encoder_layers.append(self.encoder_cfg())

                self.inpt_encoder_layers.append(self.encoder_cfg())
                
        else:
            self.decoder_layers.append(self.decoder_cfg())
            
        self.last_encoder = self.encoder_cfg()
            
        self.downscale_conv  = DenseNetwork(2*self.upscale_dims,self.vkq_dims,
                                     zeroed=not self.skip_cnt, **self.dense_cfg)

        if self.skip_cnt:
            self.last_mlp  = DenseNetwork(self.vkq_dims+self.upscale_dims, self.vkq_dims,
                                        zeroed=self.skip_cnt, **self.dense_cfg)

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
        
            
    def forward(self, input_vkq: T.Tensor, mask:T.Tensor=None, ctxt:T.Tensor=None) -> T.Tensor:
        input_vkq= input_vkq.to(self.device)
        input_vkq_original= input_vkq.clone()
        
        if mask is not None:
            mask= mask.to(self.device)
        if ctxt is None:
            ctxt={}
            
        # add noise_timestamp to ctxt
        if "scalars" in ctxt:
            ctxt_scalars = self.init_scalars_ctxt(ctxt["scalars"].to(self.device))

        #ctxt cnts
        if "cnts" in ctxt:
            ctxt_cnts = ctxt["cnts"].clone().to(self.device)
            ctxt_mask = ctxt["mask"].to(self.device)
            input_ctxt = self.init_conv_ctxt(ctxt_cnts, ctxt_scalars)
        else:
            input_ctxt = input_vkq.clone()
            if mask is not None:
                ctxt_mask = mask.clone()
            
        # network starts
        # simple MLP
        input_vkq = self.init_dense(input_vkq, ctxt_scalars)

        # transformers
        for i in range(self.num_layers):

            # encodering input/ctxt
            if self.encoder_cfg is not None:

                # self attention for input
                input_vkq = self.inpt_encoder_layers[i](input_vkq, mask_vk=mask,
                                                         ctxt=ctxt_scalars)

                # self attention for ctxt
                if len(self.ctxt_encoder_layers)>0:
                    input_ctxt = self.ctxt_encoder_layers[i](input_ctxt,
                                                            mask_vk=ctxt_mask,
                                                            ctxt=ctxt_scalars)

            # Decode attention
            if len(self.decoder_layers)>0:
                input_vkq = self.decoder_layers[i](input_vkq, input_ctxt,
                                                    mask_vk=ctxt_mask,
                                                    ctxt=ctxt_scalars)

        # last SA
        input_vkq = self.last_encoder(input_vkq, mask_vk=mask,ctxt=ctxt_scalars)

        # downscale output to same output features
        output = self.downscale_conv(input_vkq, ctxt_scalars)

        if self.skip_cnt:
            return self.last_mlp(input_vkq_original+output, ctxt_scalars)
        else:
            return output