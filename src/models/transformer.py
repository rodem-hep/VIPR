"Transformer"
import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import math
import torch as T
import torch.nn as nn
import torchvision as TV
import numpy as np
from typing import Optional, List, Tuple, Union, Callable, Mapping
from omegaconf import OmegaConf

# internal 
from tools import misc
from tools.discriminator import DenseNet
from tools.torch_utils import activation_functions
from src.models.modules import FiLM
import src.positional_encoding as pe

class DenseNetwork(nn.Module):
    def __init__(self, in_features:int, out_features:int, n_layers:int=1, ctxt_dim:int=0,
                 act_str:str="leakyrelu", norm:str="", norm_args=None, zeroed=False,
                #  skip_cnt:bool=True
                 ):
        super().__init__()
        # self.skip_cnt=skip_cnt
        self.norm_args=norm_args if norm_args!=None else {}
        # if in_features!=out_features:
        #     self.skip_cnt=False
        self.layers = nn.Sequential()
        
        in_features += ctxt_dim

        for _ in range(n_layers-1):
            #add norm
            if ("layer" in norm) & (norm_args is not None):
                self.layers.append(nn.LayerNorm(in_features, **self.norm_args))
            elif ("batch" in norm) & (norm_args is not None):
                self.layers.append(nn.BatchNorm1d(in_features, **self.norm_args))
                
            #add linear
            self.layers.append(nn.Linear(in_features, in_features))
            
            #add acti function
            if act_str is not None:
                self.layers.append(activation_functions(act_str.casefold()))

        # last linear
        self.layers.append(nn.Linear(in_features, out_features))
        # if act_str is not None:
        #     self.layers.append(activation_functions(act_str.casefold()))
        
        if zeroed:
            # if act_str is not None:
            #     self.layers[-2].weight.data.fill_(0.00)
            #     self.layers[-2].bias.data.fill_(0.00)
            # else:
            self.layers[-1].weight.data.fill_(0.00)
            self.layers[-1].bias.data.fill_(0.00)

    def forward(self, input: T.Tensor, ctxt: Union[T.Tensor,None] = None):
        if ctxt is not None:
            if len(input.shape) != len(ctxt.shape):
                ctxt = ctxt.view(*input.shape[:-1],ctxt.shape[-1])
            if input.shape[:-1] != ctxt.shape[:-1]:
                ctxt = ctxt.expand(*input.shape[:-1],ctxt.shape[-1])
            input = T.concat([input, ctxt], -1)
        return self.layers(input)
            
def merge_masks(
    # mask_q: Union[T.BoolTensor, None],
    mask_vk: Union[T.BoolTensor, None],
    # attn_mask: Union[T.BoolTensor, None],
    q_shape: T.Size,
    k_shape: T.Size,
    device: T.device,
) -> Union[None, T.BoolTensor]:
    """Create a full attention mask which incoporates the padding
    information."""

    # Create the full mask which combines the attention and padding masks
    merged_mask = None

    # If either pad mask exists, create
    if mask_vk is not None:
    # if mask_q is not None or mask_vk is not None:
        # if mask_q is None:
        #     mask_q = T.full((q_shape[0], q_shape[1]), True, device=device)
        if mask_vk is None:
            mask_vk = T.full((k_shape[0], k_shape[1]), True, device=device)
        merged_mask = mask_vk.unsqueeze(-2).expand(-1, q_shape[-2], -1)
        # merged_mask = mask_q.unsqueeze(-1) & mask_vk.unsqueeze(-2)

    # If attention mask exists, create
    # if attn_mask is not None:
    #     merged_mask = attn_mask if merged_mask is None else attn_mask & merged_mask

    return merged_mask
    
def attention(query, key, value):
    "Compute 'Scaled Dot Product Attention'"
    # d_k is the number of features

    d_k = query.size(-1)

    scores = T.matmul(query, key.transpose(-2, -1).contiguous() ) / math.sqrt(d_k)

    p_attn = T.nn.functional.softmax(scores, dim = -1)

    return T.matmul(p_attn, value) #, p_attn


class MultiHeadAttention(nn.Module):
    """
    Simple mulit head attention WITHOUT skip connection at the end
    """ 
    def __init__(self, depth_q, depth_vk, image_shape_q=None, image_shape_vk=None,
                pos_encode_kwargs=None, attn_heads=4, trainable_pe=False,
                device="cpu", **kwargs):
        super().__init__()
        # self.depth=depth
        self.device=device
        self.image_shape_q= image_shape_q
        self.image_shape_vk= image_shape_vk
        self.pos_encode_kwargs=pos_encode_kwargs
        self.attn_heads=attn_heads
        self.trainable_pe=trainable_pe
        
        self.depth_q = int(depth_q)
        self.depth_vk = int(depth_vk)
        self.dropout = kwargs.get("dropout", 0.1)
        self.zero_init = kwargs.get("kwargs", False)
        
        # multi head blk
        if (
            (self.attn_heads>=self.depth_q) or
            (self.attn_heads>=self.depth_vk)
            ):
            print("Warning: attn_heads changed to 1")
            self.attn_heads=1

        self.depth_q_blk = self.depth_q//self.attn_heads
        self.depth_vk_blk = self.depth_vk//self.attn_heads

        if self.attn_heads*self.depth_vk_blk != self.depth_vk:
            raise ValueError("dimension not fitting")
        
        self.get_network()

        # positional encoding
        if (self.image_shape_q is not None) & (self.image_shape_vk is not None):
            if trainable_pe: #TODO should there be one if self attention
                self.pe_q=  T.nn.Parameter(T.randn(*self.image_shape_q))
                self.pe_vk =  T.nn.Parameter(T.randn(*self.image_shape_vk))
            elif self.image_shape_q is not None:
                self.pe_q = self.positional_encoding(self.image_shape_q,
                                                    self.pos_encode_kwargs,self.device)
                self.pe_vk = self.positional_encoding(self.image_shape_vk,
                                                    self.pos_encode_kwargs, self.device)
        self.to(self.device)

    @staticmethod
    def positional_encoding(image_shape, pos_encode_kwargs, device):
        channels = image_shape[0]
        embedding = pe.sinusoidal(embedding_dims=channels,device="cpu",
                                  **pos_encode_kwargs)
        h_index = T.linspace(0, 1, image_shape[-1])
        w_index = T.linspace(0, 1, image_shape[-2])
        h_embed = embedding(h_index[:, None])
        w_embed = embedding(w_index[:, None])
        pe_encoding = T.zeros(1, channels, image_shape[-2], image_shape[-1])

        for i in range(embedding.embedding_dims//2):
            pe_encoding[0, i:i+2, :, :] = T.stack(T.meshgrid(h_embed[:,i], w_embed[:,i],
                                                             indexing='xy'),0)
        return pe_encoding.to(device)

    def get_network(self):
        ## init trainable modules
        self.attention_blks = nn.ModuleList([])

        # attention
        self.W_query = nn.Sequential(
            nn.Linear(self.depth_q, self.depth_vk),
            )

        self.W_key = nn.Sequential(
            nn.Linear(self.depth_vk, self.depth_vk),
            )

        self.W_value = nn.Sequential(
            nn.Linear(self.depth_vk, self.depth_vk),
            )
        
        self.out_proj = nn.Sequential(
            # nn.LayerNorm(self.depth_vk),
            nn.Linear(self.depth_vk, self.depth_q),
            )
        if self.zero_init:
            self.out_proj[-1].weight.data.fill_(0.00)
            self.out_proj[-1].bias.data.fill_(0.00)

        self.to(self.device)
    
    # def forward(self, v: T.Tensor, k: T.Tensor, q: T.Tensor,
    #              mask_vk: T.Tensor=None, mask_q: T.Tensor=None
    #              )->T.Tensor:
        
    #     return self._forward(v, k, q, mask_vk=mask_vk, mask_q=mask_q)
    
    def image_forward(self, v: T.Tensor, k: T.Tensor, q: T.Tensor) -> T.Tensor:
        # Get the shape of the q tensor and save the original for later
        b, c, *spatial = q.shape
        
        # Add positional encoding
        v = v+self.pe_vk
        k = k+self.pe_vk
        q = q+self.pe_q

        # Flatten each image to combine the spacial dimensions: B, C/F, HxW/point-cloud
        q = T.flatten(q, start_dim=2)
        k = T.flatten(k, start_dim=2)
        v = T.flatten(v, start_dim=2)

        a_out = self._forward(v, k, q)
    
        # Bring back spacial dimensions: B, q_dim, H, W
        a_out = a_out.view(b, -1, *spatial)

        # Return the additive connection to the original q
        return a_out

    def forward(self, q: T.Tensor, v: T.Tensor = None, k: T.Tensor = None,
                 mask_vk: T.Tensor=None, mask_q: T.Tensor=None,
                 attn_mask: T.Tensor=None
                 ) -> T.Tensor:

        # If only q is provided then we automatically apply self attention
        if k is None:
            k = q
        if v is None:
            v = k

        batch_size = q.shape[0]
        
        #attn mask
        if (mask_vk is not None) or (mask_q is not None):
            attn_mask = merge_masks(mask_vk, q.shape, k.shape, q.device)
            # B x n_heads x L x S
            attn_mask = attn_mask.unsqueeze(1).repeat(1,self.attn_heads,1,1)

        # Project using the learnable layers: B, HxW/point-cloud, model_dim/features
        q = self.W_query(q)
        k = self.W_key(k)
        v = self.W_value(v)

        # Break (model_dim or features = num_heads x channels_per_head) for the different heads
        # B, HxW/point-cloud, num_heads, channels_per_head/features
        shape = (batch_size, -1, self.attn_heads, self.depth_vk_blk)
        q = q.view(shape)
        k = k.view(shape)
        v = v.view(shape)

        # Permute for the attention to apply each head independantly
        # B, num_heads, HxW, channels_per_head/features
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # Now we can apply the attention operation
        a_out = T.nn.functional.scaled_dot_product_attention(q, k, v,
                                                             attn_mask=attn_mask,
                                                             )

        # Concatenate the all of the heads together to get back to: B, model_dim/features, HxW
        a_out = a_out.transpose(1, -2).contiguous().view(batch_size, -1, self.depth_vk)

        # Pass through the final 1x1 convolution layer: B, q_dim, HxW
        return self.out_proj(a_out)

########### Layers ###########

class TransformerEncoderLayer(nn.Module):
    "self attention"
    def __init__(
        self,
        model_dim: int,
        mha_config: Union[Mapping,None] = None,
        dense_cfg: Union[Mapping,None] = None,
        ctxt_dim: int = 0
        ):
        
        super().__init__()
        mha_config = mha_config or {}
        dense_cfg = dense_cfg or {}
        self.model_dim = model_dim
        self.ctxt_dim = ctxt_dim

        # The basic blocks
        self.self_attn = MultiHeadAttention(
            model_dim, model_dim, **mha_config
        )
        self.dense = DenseNetwork(model_dim, model_dim, **dense_cfg)

        # The pre MHA and pre FFN layer normalisations
        self.input_norm = nn.LayerNorm(model_dim)
        self.after_attn_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        x: T.Tensor,
        mask: Optional[T.BoolTensor] = None,
        ctxt: Union[T.Tensor,None] = None,
        # attn_bias: Union[T.Tensor,None] = None,
        attn_mask: Optional[T.BoolTensor] = None,
    ) -> T.Tensor:
        """Pass using residual connections and layer normalisation."""
        x = x + self.self_attn(
            self.input_norm(x),mask_vk=mask,
            mask_q=mask, attn_mask=attn_mask#, attn_bias=attn_bias
        )
        x = x + self.dense(self.after_attn_norm(x))#, ctxt)
        return x

class TransformerDecoderLayer(nn.Module):
    """A transformer dencoder layer based on the GPT-2+Normformer style arcitecture.

    It contains:
    - self-attention-block
    - cross-attention block
    - dense network

    Layer norm is applied before each layer
    Residual connections are used, bypassing each layer

    Attnention masks and biases are only applied to the self attention operation
    """

    def __init__(
        self,
        model_dim: int,
        mha_config: Union[Mapping,None] = None,
        dense_cfg: Union[Mapping,None] = None,
        ctxt_dim: int = 0,
        init_self_attn: bool=False,
    ) -> None:
        """
        Args:
            mha_config: Keyword arguments for multiheaded-attention block
            dense_cfg: Keyword arguments for feed forward network
        """
        super().__init__()
        mha_config = mha_config or {}
        dense_cfg = dense_cfg or {}
        self.model_dim = model_dim
        self.ctxt_dim = ctxt_dim
        self.init_self_attn = init_self_attn

        # The basic blocks
        if self.init_self_attn:
            self.self_attn = MultiHeadAttention(
                model_dim, model_dim, **mha_config
            )
            self.norm_preSA = nn.LayerNorm(model_dim)

        self.cross_attn = MultiHeadAttention(
            model_dim, model_dim, **mha_config
        )
        self.dense = DenseNetwork(
            model_dim, model_dim, **dense_cfg #, ctxt_dim=ctxt_dim
        )

        # The pre_operation normalisation layers (lots from Foundation Transformers)
        self.norm_preC1 = nn.LayerNorm(model_dim)
        self.norm_preC2 = nn.LayerNorm(model_dim)
        self.norm_preNN = nn.LayerNorm(model_dim)

    def forward(
        self,
        q_seq: T.Tensor,
        kv_seq: T.Tensor,
        mask_q: Optional[T.BoolTensor] = None,
        mask_vk: Optional[T.BoolTensor] = None,
        ctxt: Union[T.Tensor,None] = None,
        attn_bias: Union[T.Tensor,None] = None,
        attn_mask: Optional[T.BoolTensor] = None,
    ) -> T.Tensor:
        """Pass using residual connections and layer normalisation."""

        # Apply the self attention residual update
        if self.init_self_attn:
            q_seq = q_seq + self.self_attn(
                self.norm_preSA(q_seq),
                mask_vk=mask_q,
                attn_mask=attn_mask,
                # attn_bias=attn_bias,
            )

        # Apply the cross attention residual update
        q_seq = q_seq + self.cross_attn(
            q=self.norm_preC1(q_seq), k=self.norm_preC2(kv_seq), mask_vk=mask_vk
        )

        # Apply the dense residual update
        q_seq = q_seq + self.dense(self.norm_preNN(q_seq)) # , ctxt

        return q_seq

class PerceiverLayer(nn.Module):
    def __init__(self, latent_dim:list, encode_cfg:dict, decode_cfg:dict,
                 process_cfg:dict=None, film_cfg:dict=None, dense_cfg:dict=None,
                 n_processes:int=0, device:str="cuda"):
        super().__init__()
        self.latent_dim=latent_dim
        self.process_cfg=process_cfg
        self.n_processes=n_processes
        self.film_cfg=film_cfg
        self.encode_cfg=encode_cfg
        self.decode_cfg=decode_cfg
        self.dense_cfg=dense_cfg if dense_cfg!=None else {}
        self.device = device

        # Layers
        self.film_layer = None
        self.processing_layers = nn.ModuleList([])

        #init network
        self.get_network()
        
        self.to(self.device)

    def get_network(self) -> None:
        # trainable latent space
        self.latent_arr = T.nn.Parameter(T.randn(*self.latent_dim))
        self.latent_mask = T.full(self.latent_arr.shape, True)
        
        # self.in_query_norm = nn.LayerNorm(self.input_dim)
        # self.out_query_norm = nn.LayerNorm(self.output_dim)
        # self.post_self_attn_norm = nn.LayerNorm(self.latent_dim[-1])
        
        ### Encoder
        self.encode_layer = TransformerDecoder(**self.encode_cfg)

        self.post_init_query_mlp = DenseNetwork(self.latent_dim[-1],self.latent_dim[-1],
                                         **self.dense_cfg)
        
        ### Processor
        for _ in range(self.n_processes):
            self.processing_layers.append(TransformerEncoder(**self.process_cfg))
        
        ### decoder
        if self.decode_cfg is not None:
            self.decode_layer = TransformerDecoder(**self.decode_cfg)

            self.last_query_mlp = DenseNetwork(self.latent_dim[-1],self.latent_dim[-1],
                                            **self.dense_cfg)

        # film for context
        if (self.film_cfg is not None) & (self.n_processes>0):
            self.film_layer = FiLM(**self.film_cfg,
                                   lst_channel=[self.latent_dim[-1]]*(self.n_processes+1))

    def forward(self, input_ten:T.Tensor, ctxt_ten:T.Tensor, mask_vk:T.Tensor,
                ctxt:T.Tensor=None) -> T.Tensor:

        # init film context
        if self.film_layer is not None:
            self.film_layer(ctxt)
        
        latent_ten = self.latent_arr.expand(len(ctxt_ten),*self.latent_arr.shape)
        
        ### Encode ctxt_ten to latent_ten
        latent_ten = self.encode_layer(latent_ten, ctxt_ten, mask_vk=mask_vk)

        #ctxt from FiLM
        if self.film_layer is not None:
            FiLM_para = next(self.film_layer)
            latent_ten =  FiLM_para[:,0:1]*latent_ten+FiLM_para[:,1:2]


        ### Processor
        for nr_layer in range(self.n_processes):

            if self.film_layer is not None:
                FiLM_para = next(self.film_layer)
                latent_ten =  FiLM_para[:,0:1]*latent_ten+FiLM_para[:,1:2]

            latent_ten = self.processing_layers[nr_layer](latent_ten)

        if self.decode_cfg is not None:
            ### Decoder latent_ten to input_ten
            output = self.decode_layer(input_ten, latent_ten)

            # skip connection
            return output
            # return self.last_query_mlp(output)+output_ten
        else:
            return latent_ten


########### Blocks ###########

class TransformerEncoder(nn.Module):
    """A stack of N transformer encoder layers followed by a final normalisation step.

    Sequence -> Sequence
    """

    def __init__(
        self,
        model_dim: int = 64,
        num_layers: int = 3,
        mha_config: Union[Mapping,None] = None,
        dense_cfg: Union[Mapping,None] = None,
        ctxt_dim: int = 0,
    ) -> None:
        """
        Args:
            model_dim: Feature sieze for input, output, and all intermediate layers
            num_layers: Number of encoder layers used
            mha_config: Keyword arguments for the mha block
            dense_cfg: Keyword arguments for the dense network in each layer
            ctxt_dim: Dimension of the context inputs
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(model_dim, mha_config, dense_cfg, ctxt_dim)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, x: T.Tensor, **kwargs) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        for layer in self.layers:
            x = layer(x, **kwargs)
        return self.final_norm(x)

class TransformerDecoder(nn.Module):
    """A stack of N transformer dencoder layers followed by a final normalisation step.

    Sequence x Sequence -> Sequence
    """

    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        mha_config: Union[Mapping,None] = None,
        dense_cfg: Union[Mapping,None] = None,
        ctxt_dim: int = 0,
        init_self_attn:bool = False,
    ) -> None:
        """
        Args:
            model_dim: Feature sieze for input, output, and all intermediate layers
            num_layers: Number of encoder layers used
            mha_config: Keyword arguments for the mha block
            dense_cfg: Keyword arguments for the dense network in each layer
            ctxt_dim: Dimension of the context input
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(model_dim, mha_config, dense_cfg, ctxt_dim,
                                        init_self_attn=init_self_attn)
                for _ in range(num_layers)
            ]
        )
        self.init_self_attn=init_self_attn
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, q_seq: T.Tensor, kv_seq: T.Tensor, **kwargs) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        for layer in self.layers:
            q_seq = layer(q_seq, kv_seq, **kwargs)
        return self.final_norm(q_seq)

class Perceiver(nn.Module):
    """A stack of N transformer dencoder layers followed by a final normalisation step.

    Sequence x Sequence -> Sequence
    """

    def __init__(
        self,
        pcivr_cfg: Union[Mapping,None],
        num_layers: int = 1,
        device:str="cpu",
    ) -> None:
        """
        Args:
            pcivr_cfg: PerceiverLayer config
            num_layers: Number of encoder layers used
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PerceiverLayer(**pcivr_cfg)
                for _ in range(num_layers)
            ]
        )
        self.pcivr_cfg=pcivr_cfg
        self.num_layers=num_layers
        self.device=device

        self.to(self.device)
        # self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, input_ten: T.Tensor, ctxt_ten: T.Tensor, **kwargs) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        for layer in self.layers:
            input_ten = layer(input_ten, ctxt_ten, **kwargs)
        return input_ten
        # return self.final_norm(q_seq)

class UPerceiver(nn.Module):
    """A stack of N transformer dencoder layers followed by a final normalisation step.

    Sequence x Sequence -> Sequence
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        max_cnts: int,
        cnts_sizes: List[int],
        pcivr_cfg: Mapping,
        device:str="cpu",
    ) -> None:
        """
        Args:
            pcivr_cfg: PerceiverLayer config
            num_layers: Number of encoder layers used
        """
        super().__init__()
        self.input_dim=input_dim
        self.model_dim=model_dim
        self.cnts_sizes=cnts_sizes
        self.max_cnts=max_cnts
        self.pcivr_cfg=pcivr_cfg
        self.device=device

        self.down_layers = nn.ModuleList([])
        self.up_layers = nn.ModuleList([])
        self.get_network()
        
    def get_network(self):
        # scale features up/down
        self.pre_dense = DenseNetwork(self.input_dim,self.model_dim,
                                      **self.pcivr_cfg.dense_cfg)

        self.post_dense = DenseNetwork(self.model_dim,self.input_dim,
                                      **self.pcivr_cfg.dense_cfg)
        # downscale
        for cnts_per_layer in self.cnts_sizes:
            pcivr_cfg_up = self.pcivr_cfg.copy()
            pcivr_cfg_up["latent_dim"] = [cnts_per_layer, self.model_dim]
            self.down_layers.append(PerceiverLayer(**pcivr_cfg_up))

        # upscale
        for cnts_per_layer in self.cnts_sizes[1::-1]+[self.max_cnts]:
            pcivr_cfg_up = self.pcivr_cfg.copy()
            pcivr_cfg_up["latent_dim"] = [cnts_per_layer, self.model_dim]
            self.up_layers.append(PerceiverLayer(**pcivr_cfg_up))
        self.to(self.device)

        # self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, input_ten: T.Tensor, mask_vk: T.Tensor, **kwargs) -> T.Tensor:
        """Pass the input through all layers sequentially."""
        skip_features=[]
        
        # init dense to upscale features
        input_ten = self.pre_dense(input_ten)

        # downscale pc with cross attention with attn pooling
        skip_features.append(input_ten.clone())
        input_ten = self.down_layers[0](input_ten=kwargs.get("ctxt_ten", None),
                                        ctxt_ten=input_ten, mask_vk=mask_vk, **kwargs)
        for layer in self.down_layers[1:]:
            skip_features.append(input_ten.clone())
            input_ten = layer(input_ten=kwargs.get("ctxt_ten", None),
                              ctxt_ten=input_ten, mask_vk=T.ones(*input_ten.shape[:-1])==1,
                              **kwargs)

        # upscale pc with cross attention with skip connections
        skip_features = skip_features[::-1]
        for layer in self.up_layers:
            input_ten = layer(input_ten=skip_features.pop(),
                              ctxt_ten=input_ten,
                              mask_vk=T.ones(*input_ten.shape[:-1])==1, **kwargs)

        # downscales features
        input_ten = self.post_dense(input_ten)

        return input_ten

def test_get_data():
    config = misc.load_yaml("configs/data_cfgs/data_cfg.yaml")
    data = hydra.utils.instantiate(config.train_set)
    
    dataloader = hydra.utils.instantiate(config.loader_cfg)(data)
    dataloader = hydra.utils.instantiate(config.img_enc)(dataloader)
    #find std and mean of data
    data=[]
    data_ctxt=[]
    for nr, i in enumerate(dataloader):
        data.append(i[0] if len(i) else i)
        if i[1] is not None:
            data_ctxt.append(i[1] if len(i) else i)
        if nr==1:
            break
    data=T.concat(data)
    if data_ctxt[0] is not None:
        data_ctxt=T.concat(data_ctxt)
    return data

if __name__ == "__main__":
    import hydra
    model_cfg = misc.load_yaml("/home/malteal/local_work/diffusion/configs/model/u_perceiver_cfg.yaml")
    max_cnts =200
    UNet = UPerceiver(max_cnts=max_cnts, **model_cfg.trans_cfg)
    
    input_ten = T.randn(128, max_cnts, 3)
    # ctxt_ten = T.randn(128, 64, 3)
    
    output_ten = UNet(input_ten, mask_vk=input_ten.sum(-1)>1)
    



