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
import positional_encoding as pe
    
def attention(query, key, value):
    "Compute 'Scaled Dot Product Attention'"
    # d_k is the number of features

    d_k = query.size(-1)

    scores = T.matmul(query, key.transpose(-2, -1).contiguous() ) / math.sqrt(d_k)

    p_attn = T.nn.functional.softmax(scores, dim = -1)

    return T.matmul(p_attn, value) #, p_attn


class MultiHeadAttention(nn.Module):
    """
    TODO should not be for images

    should have another for images
    """ 
    def __init__(self, depth_q, depth_vk, image_shape_q=None, image_shape_vk=None,
                pos_encode_kwargs=None,attn_heads=4, trainable_pe=False,
                device="cuda", **kwargs):
        super().__init__()
        # self.depth=depth
        self.device=device
        self.image_shape_q= image_shape_q
        self.image_shape_vk= image_shape_vk
        self.pos_encode_kwargs=pos_encode_kwargs
        self.attn_heads=attn_heads
        self.trainable_pe=trainable_pe
        
        self.depth_q = depth_q
        self.depth_vk = depth_vk
        
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
            nn.BatchNorm1d(self.depth_q),
            nn.Conv1d(self.depth_q, self.depth_vk, 1),
            )

        self.W_key = nn.Sequential(
            nn.BatchNorm1d(self.depth_vk),
            nn.Conv1d(self.depth_vk, self.depth_vk, 1),
            )

        self.W_value = nn.Sequential(
            nn.BatchNorm1d(self.depth_vk),
            nn.Conv1d(self.depth_vk, self.depth_vk, 1),
            )
        
        self.out_proj = nn.Sequential(
            nn.BatchNorm1d(self.depth_vk),
            nn.Conv1d(self.depth_vk, self.depth_q, 1),
            )

        self.out_proj[-1].weight.data.fill_(0.00)
        self.out_proj[-1].bias.data.fill_(0.00)

        self.to(self.device)
    
    def forward(self, v:T.Tensor, k:T.Tensor, q:T.Tensor)->T.Tensor:
        
        return self._forward(v, k, q)
    
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

    def _forward(self, v: T.Tensor, k: T.Tensor, q: T.Tensor) -> T.Tensor:
        q_orig = q.clone()

        batch_size = q.shape[0]

        # Project using the learnable layers: B, model_dim/features, HxW/point-cloud
        q = self.W_query(q)
        k = self.W_key(k)
        v = self.W_value(v)

        # Break (model_dim or features = num_heads x channels_per_head) for the different heads
        shape = (batch_size, self.attn_heads, self.depth_vk_blk, -1)
        q = q.view(shape)
        k = k.view(shape) # B, num_heads, channels_per_head/features, HxW/point-cloud
        v = v.view(shape)

        # Permute for the attention to apply each head independantly
        # B, num_heads, HxW, channels_per_head/features
        q = q.transpose(-1, -2).contiguous()
        k = k.transpose(-1, -2).contiguous()
        v = v.transpose(-1, -2).contiguous()

        # Now we can apply the attention operation
        a_out = attention(q, k, v)

        # Concatenate the all of the heads together to get back to: B, model_dim/features, HxW
        a_out = a_out.transpose(-1, -2).contiguous().view(batch_size, self.depth_vk, -1)

        # skip connection
        a_out = a_out+q_orig

        # Pass through the final 1x1 convolution layer: B, q_dim, HxW
        # with additionl skip connection
        return self.out_proj(a_out)+a_out

class MultiHeadGateAttention(MultiHeadAttention):
    # similar to https://arxiv.org/pdf/2103.06104.pdf
    def __init__(self, depth_q, depth_vk, image_shape_q, image_shape_vk,
                 pos_encode_kwargs, attn_heads=4, device="cuda", **kwargs):
        if np.any(image_shape_q[0] != image_shape_vk[0]):
            raise ValueError("Image dimension between q and vk has to be the same")
        self.original_depth_q=depth_q
        super().__init__(depth_vk, depth_vk, image_shape_q, image_shape_vk,
                         pos_encode_kwargs, attn_heads, device, **kwargs)
        
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
            nn.GELU(),
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
            nn.GELU(),
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

    # def img_to_patch(self, image, reverse=False, image_shape=None):
    #     """
    #     Inputs:
    #         x - Tensor representing the image of shape [B, C, H, W]
    #         patch_size - Number of pixels per dimension of the patches (integer)
    #         flatten_channels - If True, the patches will be returned in a flattened format
    #                         as a feature vector instead of a image grid.
    #     from https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html
    #     """
    #     if image_shape is None:
    #         B, C, H, W = image.shape
    #     else:
    #         B, C, H, W = image_shape
    #     if reverse:
    #         image = image.permute(0, 2,1) #[B, H'*W', C*p_H*p_W]
    #         if self.flatten_channels:
    #             image = image.unflatten(1, (H//self.patch_img_dim[0], W//self.patch_img_dim[1]))  # [B, H', W', C*p_H*p_W]
    #             image = image.unflatten(-1, (self.n_channels, self.patch_img_dim[0], self.patch_img_dim[1]))  # [B, H', W', C, p_H, p_W]
    #         image = image.permute(0, 3, 1, 4, 2, 5)  # [B, C, H', W', p_H, p_W]
    #         image = image.reshape(B, C, H, W) # [B, C, H, W]
    #         return image
    #     else:
    #         image = image.reshape(B, C, H // self.patch_img_dim[0],
    #                             self.patch_img_dim[0],
    #                             W // self.patch_img_dim[1],
    #                             self.patch_img_dim[1])
    #         image = image.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    #         image = image.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    #         if self.flatten_channels:
    #             image = image.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]

    #         # return [B, C*p_H*p_W, H'*W']
    #         # same as [B, features, nodes]
    #         return image.permute(0, 2, 1)
    
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

         # restore image
        # output_image = self.img_to_patch(attn_image, reverse=True,
        #                                   image_shape=image_orig.shape)
        
        return output_image+image_orig

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

