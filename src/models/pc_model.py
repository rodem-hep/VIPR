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
from tools.discriminator import DenseNet
import src.positional_encoding as pe
from src.models.transformer import MultiHeadAttention, PerceiverBlock, PCMLP

class TransformerEncoder(nn.Module):
    def __init__(self, vkq_dims, ctxt_dims:Union[int, dict]=None,
                 pcivr_cfg:dict=None, mlp_cfg:dict=None,
                 upscale_dims:int=64, n_encoders=2,
                 embedding_dims=None, attn_heads:int = 4, device:str="cuda"):
        super().__init__()
        self.vkq_dims = vkq_dims
        self.ctxt_dims = ctxt_dims
        self.n_encoders=n_encoders
        self.upscale_dims= upscale_dims
        self.attn_heads=attn_heads
        self.pcivr_cfg=pcivr_cfg
        self.device = device
        self.mlp_cfg=mlp_cfg if mlp_cfg!=None else {}
        self.embedding_dims = 0 if embedding_dims is None else embedding_dims

        self.trans_encoder_layers = nn.ModuleList([])
        self.init_conv_ctxt = None

        self.get_network()

    def get_network(self) -> None:

        if self.embedding_dims!=0:
            self.embedding = pe.FourierFeatures(1, self.embedding_dims)
            # self.embedding = pe.sinusoidal(self.embedding_dims,
            #                                embedding_min_frequency=1e-5,
            #                                embedding_max_frequency=80)

        self.init_conv = PCMLP(self.vkq_dims+self.embedding_dims,
                               self.upscale_dims,
                               norm_args={"normalized_shape":self.vkq_dims+self.embedding_dims},
                               skip_cnt=False)
        if "cnts" in self.ctxt_dims:
            self.init_conv_ctxt = PCMLP(self.vkq_dims+self.embedding_dims,
                                self.upscale_dims,
                                norm_args={"normalized_shape":self.vkq_dims+self.embedding_dims},
                                skip_cnt=False)
            
        
        if "film_cfg" in self.pcivr_cfg:
            if isinstance(self.ctxt_dims, int):
                raise TypeError("ctxt_dims has to be a dict. "+
                                "Keyname for FiLM is scalars")
            self.pcivr_cfg["film_cfg"]["ctxt_size"]+=self.ctxt_dims.get("scalars", 0)
        
        for _ in range(self.n_encoders):
            if self.pcivr_cfg is not None:
                self.trans_encoder_layers.append(PerceiverBlock(**self.pcivr_cfg))
            else:
                self.trans_encoder_layers.append(MultiHeadAttention(depth_q=self.upscale_dims,
                                                            depth_vk=self.upscale_dims,
                                                            attn_heads=self.attn_heads))
        
        self.downscale_conv  = PCMLP(self.upscale_dims,self.vkq_dims,
                                     norm_args={"normalized_shape":self.upscale_dims},
                                     zeroed=False,
                                     **self.mlp_cfg)

        self.last_conv = PCMLP(self.vkq_dims,self.vkq_dims, n_layers=1, zeroed=True,
                               skip_cnt=False)

    @T.no_grad()
    def exponential_moving_averages(self, state_dict, ema_ratio):
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
        input_vkq, noise_variances, mask = (input_vkq.to(self.device),
                                      noise_variances.to(self.device),
                                      mask.to(self.device))
        if ctxt is None:
            ctxt={}

        original_input = input_vkq.clone()
        if noise_variances is not None:
            # calculate embedding
            noise_variances = self.embedding(noise_variances, len(input_vkq.shape))
            
            # concat noise_timestamp and inputs_vkq
            input_vkq = T.concat(
                [input_vkq, 
                 noise_variances.expand(len(noise_variances), input_vkq.shape[1], 
                                        noise_variances.shape[-1])],
                 -1
                )
            if "cnts" in ctxt:
                ctxt["cnts"] = T.concat(
                    [ctxt["cnts"].to(self.device),
                    noise_variances.expand(len(noise_variances), ctxt["cnts"].shape[1],
                                            noise_variances.shape[-1])],
                    -1
                    )
            
            noise_variances = noise_variances.squeeze(-2)

            # add noise_timestamp to ctxt 
            if "scalars" not in ctxt:
                ctxt["scalars"] = noise_variances
            else:
                ctxt["scalars"] = T.concat([noise_variances,
                                            ctxt["scalars"].to(self.device)], 1)

        # network starts
        # simple MLP
        input_vkq = self.init_conv(input_vkq)
        if self.init_conv_ctxt is not None:
            # if failed means cnts is missing in ctxt
            input_ctxt = self.init_conv_ctxt(ctxt["cnts"])
            ctxt_mask = ctxt["mask"].to(self.device)
        else:
            input_ctxt = input_vkq.clone()
            ctxt_mask = mask.clone()
            
        # transformers
        for nr in range(self.n_encoders):
            # input_vkq_attn = self.trans_encoder_layers[nr](input_vkq, input_vkq,
            #                                                input_vkq, mask)
            # if "cnts" in ctxt:
            #     output_vkq  =  self.trans_encoder_layers[nr](input_ctxt,
            #                                                 ctxt["mask"].to(self.device),
            #                                                 input_vkq, mask.to(self.device),
            #                                                 scalar_ctxt=ctxt["scalars"])
            input_vkq  =  self.trans_encoder_layers[nr](input_ctxt, ctxt_mask,
                                                         input_vkq, mask,
                                                         scalar_ctxt=ctxt["scalars"])
            # input_vkq = input_vkq_attn+self.mlp_layers[nr](input_vkq_attn)

        # output skip connection and ffc
        # return self.downscale_conv(input_vkq)+original_input
        return self.last_conv(self.downscale_conv(input_vkq)+original_input)

