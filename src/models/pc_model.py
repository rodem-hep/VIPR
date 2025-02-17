"Transformers"
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
from tools.tools import diffusion_schemes as ds
from tools.tools.modules import IterativeNormLayer


class PCDiffusion(
    ds.RectifiedFlows
):
    def __init__(self, vkq_dims, time_embedding: partial, 
                 ctxt_dims:Union[int, dict]=None, num_layers:int =1,
                 decoder_cfg:dict=None, encoder_cfg:dict=None, ctxt_encoder_cfg:dict=None, dense_cfg:dict=None,
                 upscale_dims:int=64, skip_cnt:bool =False,
                 device:str="cuda", **kwargs):
        super().__init__()
        self.vkq_dims = vkq_dims
        self.ctxt_dims = ctxt_dims
        self.upscale_dims= upscale_dims
        self.decoder_cfg=decoder_cfg
        self.encoder_cfg=encoder_cfg
        self.ctxt_encoder_cfg=ctxt_encoder_cfg
        self.device = device
        self.skip_cnt=skip_cnt
        self.dense_cfg=dense_cfg if dense_cfg!=None else {}
        self.time_embedding = time_embedding
        self.num_layers = num_layers
        self.loss=kwargs.get('loss', nn.MSELoss())
        self.eval_cfg = kwargs.get('eval_cfg', {'eval_iters': 10,
                                                'n_diffusion_steps': 20})
        
        self.ctxt_encoder_layers = nn.ModuleList([])
        self.inpt_encoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        self.init_conv_ctxt = None
        

        self.get_network()

    def get_network(self) -> None:
        # input norm
        self.inpt_normaliser = IterativeNormLayer((1,self.vkq_dims), max_iters=10_000)
        
        
        # init cnts
        self.init_dense = DenseNetwork(self.vkq_dims+self.upscale_dims,
                                       self.upscale_dims, **self.dense_cfg)

        # ctxt cnts
        if "cnts" in self.ctxt_dims:
            self.init_conv_ctxt = DenseNetwork(
                self.ctxt_dims["cnts"]+self.upscale_dims,
                self.upscale_dims,**self.dense_cfg)

        self.full_ctxt_dims = self.ctxt_dims.get('scalars', 0)+self.time_embedding.embedding_dims
        
        # init time embedding
        self.time_embedding = T.nn.Sequential(
            self.time_embedding,
            DenseNetwork(in_features=self.time_embedding.embedding_dims, out_features=self.time_embedding.embedding_dims,**self.dense_cfg)
            )

        # ctxt scalar
        if self.full_ctxt_dims > 0:

            self.init_scalars_ctxt = DenseNetwork(self.full_ctxt_dims,
                                                  self.upscale_dims,**self.dense_cfg)

        if self.decoder_cfg is not None:
            if isinstance(self.ctxt_dims, int):
                raise TypeError("ctxt_dims has to be a dict. "+
                                "Keyname for FiLM is scalars")
            elif "film_cfg" in self.decoder_cfg.keywords:
                self.decoder_cfg["film_cfg"]["ctxt_size"]+=self.ctxt_dims.get("scalars", 0)
        
        if self.encoder_cfg is not None:

            for nr in range(self.num_layers):

                if self.decoder_cfg is not None:
                    self.decoder_layers.append(self.decoder_cfg())

                if ("cnts" in self.ctxt_dims) & (self.ctxt_encoder_cfg is not None):
                    self.ctxt_encoder_layers.append(self.ctxt_encoder_cfg()) # using perceivers
                    self.glob_token_out = self.ctxt_encoder_layers[nr].layers[0].decode_cfg is None
                elif ("cnts" in self.ctxt_dims):
                    self.ctxt_encoder_layers.append(self.encoder_cfg())
                    self.glob_token_out=False

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
    
    def generate(self, noise:T.Tensor, ctxt=None, mask=None, n_steps=None):
        # noise -> images -> denormalized images
        
        with T.no_grad():
            generated_images = self.reverse_diffusion(noise=noise, ctxt=ctxt, mask=mask,
                                                      n_steps=n_steps)

        generated_images = self.inpt_normaliser.reverse(generated_images, mask=mask).cpu()

        # create output dict
        gen_data = {"gen_data": generated_images}
        if ctxt is not None:
            gen_data["ctxt"] = ctxt
        if mask is not None:
            gen_data["mask"] = mask

        return gen_data
            
    def forward(self, input_vkq: T.Tensor, mask:T.Tensor, latn:T.Tensor, time:T.Tensor) -> T.Tensor:
        input_vkq= input_vkq.to(self.device)
        input_vkq_original= input_vkq.clone()
        
        mask= mask.to(self.device)

            
        # add noise_timestamp to ctxt
        if "scalars" in latn:
            ctxt_scalars = T.concat([latn["scalars"], time], -1)
            ctxt_scalars = self.init_scalars_ctxt(ctxt_scalars.to(self.device))

        #ctxt cnts
        if "cnts" in latn:
            ctxt_cnts = latn["cnts"].clone().to(self.device)
            ctxt_mask = latn["mask"].to(self.device)
            input_ctxt = self.init_conv_ctxt(ctxt_cnts, ctxt_scalars)
            # clone the cnts ctxt for perceiver
            input_ctxt_clone = input_ctxt.clone()
            ctxt_mask_clone = ctxt_mask.clone()
            
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
                    if (self.ctxt_encoder_cfg is not None) and (self.glob_token_out): # pool with perceiver
                        input_ctxt = self.ctxt_encoder_layers[i](input_ctxt_clone,
                                                                mask_vk=ctxt_mask_clone,
                                                                ctxt=ctxt_scalars)
                        ctxt_mask = T.ones(input_ctxt.shape[:2]).bool().to(self.device)
                    else:
                        input_ctxt = self.ctxt_encoder_layers[i](input_ctxt,
                                                                mask_vk=ctxt_mask,
                                                                ctxt=ctxt_scalars)
            # Decode attention
            if len(self.decoder_layers)>0:
                input_vkq = self.decoder_layers[i](input_vkq, input_ctxt,
                                                    mask_vk=ctxt_mask,
                                                    mask_q = mask,
                                                    ctxt=ctxt_scalars)

        # last SA
        input_vkq = self.last_encoder(input_vkq, mask_vk=mask,ctxt=ctxt_scalars)

        # downscale output to same output features
        output = self.downscale_conv(input_vkq, ctxt_scalars)

        if self.skip_cnt:
            return self.last_mlp(input_vkq_original+output, ctxt_scalars)
        else:
            return output
    
    def train_step(self, inpt:T.Tensor, ctxt:T.Tensor=None,
                     mask:T.Tensor=None, training:bool=True
                     ):
        return self._train_step(inpt, ctxt, mask, training)