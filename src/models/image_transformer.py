"Transformer setup for images"
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
from src.models.transformer import MultiHeadAttention



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
