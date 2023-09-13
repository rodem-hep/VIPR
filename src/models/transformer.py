"Transformer"
import math
import torch as T
import torch.nn as nn
import torchvision as TV
import numpy as np
from typing import Optional, List, Tuple, Union
from omegaconf import OmegaConf

# internal 
from tools import misc
from tools.discriminator import DenseNet
from tools.torch_utils import activation_functions
from src.models.modules import FiLM
import src.positional_encoding as pe

class PCMLP(nn.Module):
    def __init__(self, in_features:int, out_features:int, n_layers:int=1,
                 act_str:str="leakyrelu", norm:str="", norm_args=None, zeroed=False,
                 skip_cnt:bool=True):
        super().__init__()
        self.skip_cnt=skip_cnt
        self.norm_args=norm_args if norm_args!=None else {}
        if in_features!=out_features:
            self.skip_cnt=False
        self.layers = nn.Sequential()
        

        for _ in range(n_layers-1):
            #add norm
            if ("layer" in norm) & (norm_args is not None):
                self.layers.append(nn.LayerNorm(**self.norm_args))
            elif ("batch" in norm) & (norm_args is not None):
                self.layers.append(nn.BatchNorm1d(**self.norm_args))
                
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

    def forward(self, input):
        if self.skip_cnt:
            return self.layers(input)+input
        else:
            return self.layers(input)
            
def merge_masks(
    q_mask: Union[T.BoolTensor, None],
    kv_mask: Union[T.BoolTensor, None],
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
    if q_mask is not None or kv_mask is not None:
        if q_mask is None:
            q_mask = T.full((q_shape[0], q_shape[1]), True, device=device)
        if kv_mask is None:
            kv_mask = T.full((k_shape[0], k_shape[1]), True, device=device)
        merged_mask = q_mask.unsqueeze(-1) & kv_mask.unsqueeze(-2)

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
                device="cuda", **kwargs):
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
            nn.LayerNorm(self.depth_vk),
            nn.Linear(self.depth_vk, self.depth_q),
            )
        if self.zero_init:
            self.out_proj[-1].weight.data.fill_(0.00)
            self.out_proj[-1].bias.data.fill_(0.00)

        self.to(self.device)
    
    def forward(self, v: T.Tensor, k: T.Tensor, q: T.Tensor,
                 mask_vk: T.Tensor=None, mask_q: T.Tensor=None
                 )->T.Tensor:
        
        return self._forward(v, k, q, mask_vk=mask_vk, mask_q=mask_q)
    
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

    def _forward(self, v: T.Tensor, k: T.Tensor, q: T.Tensor,
                 mask_vk: T.Tensor=None, mask_q: T.Tensor=None,
                 attn_mask: T.Tensor=None
                 ) -> T.Tensor:
        # q_orig = q.clone()

        batch_size = q.shape[0]
        
        #attn mask
        if (mask_vk is not None) or (mask_q is not None):
            attn_mask = merge_masks(mask_q, mask_vk, q.shape, k.shape, q.device)
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
        q = q.transpose(1, -2).contiguous()
        k = k.transpose(1, -2).contiguous()
        v = v.transpose(1, -2).contiguous()

        # Now we can apply the attention operation
        a_out = T.nn.functional.scaled_dot_product_attention(q, k, v,
                                                             attn_mask=attn_mask,
                                                             )

        # Concatenate the all of the heads together to get back to: B, model_dim/features, HxW
        a_out = a_out.transpose(1, -2).contiguous().view(batch_size, -1, self.depth_vk)

        # Pass through the final 1x1 convolution layer: B, q_dim, HxW
        return self.out_proj(a_out)

class MultiHeadGateAttention(MultiHeadAttention):
    # similar to https://arxiv.org/pdf/2103.06104.pdf
    def __init__(self, depth_q, depth_vk, image_shape_q, image_shape_vk,
                 pos_encode_kwargs, attn_heads=4, device="cuda", **kwargs):
        if np.any(image_shape_q[0] != image_shape_vk[0]):
            raise ValueError("Image dimension between q and vk has to be the same")
        super().__init__(depth_q=depth_vk, depth_vk=depth_vk,
                         image_shape_q=image_shape_q, image_shape_vk=image_shape_vk,
                         pos_encode_kwargs=pos_encode_kwargs, attn_heads=attn_heads, device=device, **kwargs)
        self.original_depth_q=depth_q
        
        self.values_conv = nn.Sequential(nn.Conv2d(self.depth_vk, self.depth_vk, kernel_size=1),
                                  nn.BatchNorm2d(self.depth_vk),
                                  nn.SiLU())

        self.keys_conv = nn.Sequential(nn.Conv2d(self.depth_vk, self.depth_vk, kernel_size=1),
                                  nn.BatchNorm2d(self.depth_vk),
                                  nn.SiLU())

        self.queries_conv = nn.Sequential(nn.Conv2d(self.original_depth_q, self.depth_vk, kernel_size=1),
                                  nn.BatchNorm2d(self.depth_vk),
                                  nn.SiLU())

        # Upsample before or after attention??!??!
        #after results in every 2x2 is the same
        self.conv = nn.Sequential(
                                # nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.Conv2d(self.depth_vk, self.depth_vk, kernel_size=1),
                                # nn.BatchNorm2d(self.depth_vk),
                                nn.Sigmoid()
                                  )
        self.to(self.device)

    def forward(self, values, keys, queries):

        queries = self.queries_conv(queries)
        values = self.values_conv(values)
        keys = self.keys_conv(keys)

        gate_images = self.image_forward(values, keys, queries)

        return self.conv(gate_images) * values
        

class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, depth, image_shape, pos_encode_kwargs,
                 attn_heads=4, device="cuda", **kwargs):

        super().__init__(depth, depth, image_shape_q=image_shape,
                         image_shape_vk=image_shape,
                         pos_encode_kwargs=pos_encode_kwargs,
                         attn_heads=attn_heads, device=device, **kwargs)

    def forward(self, inputs):
        # return inputs+self.image_forward(inputs,inputs,inputs)
        return self.image_forward(inputs,inputs,inputs)
    
class VisionTransformerLayer(nn.Module):
    """
    implementation of ViT and T2TViT

    Need to have difference between channel features and patch features
        trainable postional encoding 3d

    First attention in patch then between patches?

    """
    def __init__(self, img_shape:np.ndarray, n_channels:int, kernel_size:tuple=None,
                 n_patches:int=None,
                 downscale_size:int=64,
                 attn_heads:int=16,
                 stride:int=None,
                dropout:float=0.1,
                trainable_pe:bool=True,
                device:str="cpu"
                    ):
        super().__init__()
        if (n_patches is None) and (kernel_size is None):
            raise ValueError("either n_patches or kernel_size has to be defined")
        
        if isinstance(img_shape, (int, np.int64)):
            self.img_shape = np.array([img_shape,img_shape])
        else:
            self.img_shape=np.array(img_shape)
        self.n_channels=n_channels
        self.kernel_size=kernel_size
        self.trainable_pe=trainable_pe
        self.dropout=dropout
        self.n_patches=n_patches
        if n_patches is None:
            if not isinstance(kernel_size, tuple):
                raise TypeError("kernel_size has to be a tuple")
            self.n_patches=self.img_shape//kernel_size
        elif kernel_size is None:
            self.kernel_size = tuple(self.img_shape//n_patches)
        self.stride=stride if stride is not None else self.kernel_size
        self.attn_heads=attn_heads
        self.downscale_size=downscale_size
        self.device=device
        # self.flatten_channels=True


        self.patch_img_dim=self.img_shape//self.n_patches
        self.patch_features=np.product(self.patch_img_dim)
        self.total_features = self.patch_features*self.n_channels

        if all(self.img_shape != self.patch_img_dim*self.n_patches):
            print(f"self.patch_dim: {self.patch_img_dim}")
            print(f"self.img_shape: {self.img_shape}")
            raise ValueError("Image not divisible")

        self.get_network()
    
    def get_network(self):
        self.downscale_nn= nn.Sequential(
            nn.BatchNorm1d(self.total_features),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(self.total_features, self.downscale_size, 1)
            )

        self.transformer_layer = MultiHeadAttention(self.downscale_size,
                                                    self.downscale_size,
                                                    attn_heads=self.attn_heads,
                                                    trainable_pe=False,
                                                    device=self.device)

        self.upscale_nn= nn.Sequential(
            nn.BatchNorm1d(self.downscale_size),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(self.downscale_size, self.total_features, 1),
            )
        
        self.upscale_nn[-1].weight.data.fill_(0.00)
        self.upscale_nn[-1].bias.data.fill_(0.00)

        if self.trainable_pe:
           self.pe = T.nn.Parameter(T.randn(self.n_channels, *self.img_shape))

        # fold/unfolding
        self.args = {"kernel_size":self.kernel_size,
                           "dilation":1,
                           "padding":0,
                           "stride":self.stride}
        self.unfold = nn.Unfold(**self.args)
        self.fold = nn.Fold(output_size=tuple(self.img_shape), **self.args)


        self.to(self.device)

    def forward(self, image):

        image_orig = image.clone()

        # add positional encoding
        if self.trainable_pe:
           image = image+self.pe

        # prepare image
        image_processed = self.unfold(image)
        # image_processed = self.img_to_patch(image)

        # downscale
        image_processed = self.downscale_nn(image_processed)

        attn_image = self.transformer_layer(image_processed,
                                            image_processed,
                                            image_processed)

        # upscale
        attn_image = self.upscale_nn(attn_image)

        # fold image back
        output_image = self.fold(attn_image)
        
        return output_image+image_orig

class PerceiverBlock(nn.Module):
    def __init__(self, input_query_dims:int, output_query_dims:int,
                 latent_dims:list, n_processes=2, film_cfg:dict=None,
                 mlp_cfg:dict=None, attn_heads:int = 4, device:str="cuda"):
        super().__init__()
        self.input_query_dims=input_query_dims
        self.output_query_dims=output_query_dims
        self.latent_dims=latent_dims
        self.n_processes=n_processes
        self.attn_heads=attn_heads
        self.device = device
        self.film_cfg=film_cfg
        self.mlp_cfg=mlp_cfg if mlp_cfg!=None else {}
        # Layers
        self.film_layer = None
        self.processing_layers = nn.ModuleList([])
        self.mlp_processing_layers = nn.ModuleList([])
        self.post_mlp_norms = nn.ModuleList([])
        self.pre_mlp_norms = nn.ModuleList([])
        # self.mlp_processing_layers = nn.ModuleList([])

        #init network
        self.get_network()
        
        self.to(self.device)

    def get_network(self) -> None:
        # trainable latent space
        self.latent_arr = T.nn.Parameter(T.randn(*self.latent_dims))
        self.latent_mask = T.full(self.latent_arr.shape, True)
        
        self.in_query_norm = nn.LayerNorm(self.input_query_dims)
        self.out_query_norm = nn.LayerNorm(self.output_query_dims)
        self.post_self_attn_norm = nn.LayerNorm(self.latent_dims[-1])
        
        ### Encoder
        self.encode_layer = MultiHeadAttention(depth_q=self.latent_dims[-1],
                                               depth_vk=self.input_query_dims,
                                               attn_heads=self.attn_heads,
                                               zero_init=False)

        self.post_init_query_mlp = PCMLP(self.latent_dims[-1],self.latent_dims[-1],
                                        #  norm_args={"normalized_shape":self.latent_dims[-1]},
                                         **self.mlp_cfg)
        
        ### Processor
        for _ in range(self.n_processes):
            self.mlp_processing_layers.append(PCMLP(
                self.latent_dims[-1],self.latent_dims[-1],
                norm_args={"normalized_shape":self.latent_dims[-1]}, **self.mlp_cfg)
                )
            self.pre_mlp_norms.append(
                nn.LayerNorm(self.latent_dims[-1]),
            )
            self.processing_layers.append(MultiHeadAttention(depth_q=self.latent_dims[-1],
                                                             depth_vk=self.latent_dims[-1],
                                                             attn_heads=self.attn_heads,
                                                             zero_init=False))

        # film for context
        if self.film_cfg is not None:
            self.film_layer = FiLM(**self.film_cfg,
                                   lst_channel=[self.latent_dims[-1]]*(self.n_processes+1))

        ### Decoder
        self.last_query_mlp = PCMLP(
                self.output_query_dims,self.output_query_dims,
                norm_args={"normalized_shape":self.output_query_dims},
                skip_cnt=False, **self.mlp_cfg
                )

        self.decode_layer = MultiHeadAttention(depth_q=self.output_query_dims,
                                               depth_vk=self.latent_dims[-1],
                                               attn_heads=self.attn_heads,
                                               zero_init=False)
   
    def forward(self, input_arr:T.Tensor, input_mask:T.Tensor,
                output_arr:T.Tensor, output_mask:T.Tensor,
                scalar_ctxt:T.Tensor=None) -> T.Tensor:
        
        # init film context
        if self.film_layer is not None:
            self.film_layer(scalar_ctxt)
        
        latent_ten = self.latent_arr.expand(len(input_arr),*self.latent_arr.shape)
        
        ### Encode input_arr to latent_ten size
        norm_input_arr = self.in_query_norm(input_arr)
        latent_ten = self.encode_layer(norm_input_arr, norm_input_arr,
                                        latent_ten,
                                        mask_vk=input_mask,
                                        )#+latent_ten

        #ctxt from FiLM
        if self.film_layer is not None:
            FiLM_para = next(self.film_layer)
            latent_ten =  FiLM_para[:,0:1]*latent_ten+FiLM_para[:,1:2]

        # l-norm + MLP + residual connection
        latent_ten = self.post_init_query_mlp(latent_ten)
        
        ### Processor (self attn)
        for nr_layer in range(self.n_processes):

            if self.film_layer is not None:
                FiLM_para = next(self.film_layer)
                latent_vals =  FiLM_para[:,0:1]*latent_ten+FiLM_para[:,1:2]

            latent_vals = self.pre_mlp_norms[nr_layer](latent_vals)
            latent_ten = self.processing_layers[nr_layer](latent_vals,
                                                           latent_vals,
                                                           latent_vals)+latent_ten

            latent_ten = self.mlp_processing_layers[nr_layer](latent_ten)
        latent_ten = self.post_self_attn_norm(latent_ten)
            
        ### Decoder cross attn
        # norm both pc
        norm_out_arr = self.out_query_norm(output_arr)

        # TODO should this use mask_q=output_mask ?
        output = self.decode_layer(latent_ten, latent_ten, norm_out_arr,
                                #    mask_q=output_mask
                                   )

        # skip connection
        return self.last_query_mlp(output)+output_arr


def test_get_data():
    import hydra
    config = misc.load_yaml("configs/data_cfg.yaml")
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
    device="cuda"
    data = test_get_data()
    data = data[:4]


    # data = T.randn((4,3, 16,16))
    img_shape = list(data.shape[1:])
    ViT = VisionTransformerLayer(img_shape=img_shape[1:],
                                n_channels=img_shape[0],
                                kernel_size=8,
                                stride=4,
                                device=device)
    # ViT = T2TVisionTransformerLayer(in_channels=data.shape[1],
    #                                 out_channels=1,
    #                                 kernel_size=8,
    #                                 stride=4,
    #                                 device=device)
    
    output = ViT(data.to(device))

    import matplotlib.pyplot as plt
    style = {"vmax":1, "vmin":0}

    index = 0
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(data[index].permute(1, 2, 0).cpu().detach().numpy(), **style)
    ax[1].imshow(output[index].permute(1, 2, 0).cpu().detach().numpy(), **style)
    plt.axis("off")

