"UNet"
import math
import torch as T
import torch.nn as nn
import torchvision as TV
import numpy as np
from omegaconf import OmegaConf
import torchvision

# internal 
from tools import misc
from tools.discriminator import DenseNet
from tools.modules import FiLM, Gate, ResidualBlock
import src.positional_encoding as pe
from src.models.image_transformer import (MultiHeadSelfAttention, MultiHeadGateAttention,
                         VisionTransformerLayer)


class UNet(nn.Module):
    def __init__(self, input_shape, channels, block_depth, min_size,
                 diffusion:bool=False, embedding_max_frequency=None, embedding_dims=0,
                 ctxt_dims=0, use_gate=True, img_enc=1, dropout=0, film_config=None, self_attention_cfg=None,
                 cross_attention_cfg=None, device="cuda"):
        super().__init__()
        self.input_shape = input_shape
        self.ctxt_dims = ctxt_dims
        self.embedding_max_frequency=embedding_max_frequency
        self.embedding_dims=embedding_dims
        self.channels=channels
        self.img_enc=img_enc
        self.device=device
        self.block_depth=block_depth
        self.diffusion=diffusion
        self.use_gate=use_gate
        self.self_attention_cfg=self_attention_cfg
        self._use_film=film_config is not None
        self.film_config=film_config
        self.cross_attention_cfg=cross_attention_cfg
        self.dropout=dropout

        # image dimensions after pooling
        self.img_dims = self.input_shape[-1]//2**(np.arange(len(self.channels)-1))

        if self.img_dims[-1] < min_size:
            raise ValueError("Reduce the number of channels or decrease min_size")

        self.get_network()
        self.to(self.device)

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
        
    def get_network(self):
        self.down_blocks = nn.ModuleList([])
        self.residual_block = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.init_end_conv2 = nn.ModuleList([])
        # self.gates = nn.ModuleList([]) if self.use_gate else None
        self.film = nn.ModuleList([]) if self._use_film else None
        self.cross_attention = nn.ModuleList([])
        
        # permute layer because torch is dumb
        self.in_permute = torchvision.ops.Permute([0,3,1,2])
        self.out_permute = torchvision.ops.Permute([0,2,3,1])
        
        # positional embedding
        # self.embedding = pe.sinusoidal(self.embedding_max_frequency,
        #                                self.embedding_dims,
        #                                device = self.device)
        self.embedding = pe.FourierFeatures(1, self.embedding_dims)

        # Upscale/pooling
        self.noise_upscale = nn.Upsample(size=self.input_shape[1:], mode="nearest")
        self.upscale_embedding = nn.Upsample(scale_factor=2, mode="nearest")
        self.downscale_embedding = nn.AvgPool2d(2)

        ## downscale network
        start_channels = self.input_shape[0]*self.img_enc
        if not self._use_film:
            start_channels +=self.embedding_dims

        # init conv
        self.start_conv = nn.Conv2d(start_channels,self.channels[0], kernel_size=1)
        self.end_conv = nn.Conv2d(self.input_shape[0],self.input_shape[0],kernel_size=1)
        self.end_conv.weight.data.fill_(0.00)
        self.end_conv.bias.data.fill_(0.00)
        
        ## FiLM context
        if self._use_film:
            self.film = FiLM(self.embedding_dims,#+self.ctxt_dims.get("ctxt_images", [0])[-1],
                             self.channels[1:], dense_config=self.film_config,
                             device=self.device)
        else: # dummy film
            self.film = iter([None for i in range(1000)])

        ## downscale network
        for input_ch, output_ch, img_dim in zip(self.channels[:-1], self.channels[1:],
                                                self.img_dims):
            self.down_blocks.append(ResidualBlock(input_ch, output_ch,
                                                  self.block_depth,
                                                  dropout=self.dropout,
                                                #   img_dim=img_dim if img_dim>=16 else None
                                                  ))

        ## Upscale network
        for input_ch, output_ch, in_img_dim, out_img_dim in zip(
            self.channels[::-1][:-1],
            self.channels[::-1][1:],
            self.img_dims[::-1][:-1],
            self.img_dims[::-1][1:]
            ):
            #gated cross attention
            # when img too big, use simple gating
            if ((self.cross_attention_cfg is not None)):
                # (out_img_dim <= self.cross_attention_cfg["attn_below"])): 
                self.cross_attention.append(
                    MultiHeadGateAttention(input_ch, output_ch,
                                            image_shape_vk= [output_ch, out_img_dim, out_img_dim], 
                                            image_shape_q= [output_ch, out_img_dim, out_img_dim],
                                            #[input_ch, in_img_dim, in_img_dim], 
                                            permute_layers=[self.out_permute, self.in_permute],
                                            **self.cross_attention_cfg))
            else:
                if self.use_gate:
                    self.cross_attention.append(Gate(output_ch, input_ch)) # TODO add identity so no if in forward
                else:
                    self.cross_attention.append(None)

            self.up_blocks.append(ResidualBlock(input_ch+output_ch, output_ch,
                                                self.block_depth,
                                                img_dim=out_img_dim if out_img_dim>=16 else None))
        self.up_blocks.append(ResidualBlock(self.channels[1], self.input_shape[0],
                                            self.block_depth,
                                            img_dim=self.img_dims[0] if self.img_dims[0]>=16 else None))
        
        #self attention at low resolution
        self.down_self_attention = nn.ModuleList([])
        for img_dim, n_channels in zip(self.img_dims,np.array(self.channels)[1:]):
            if img_dim <= self.self_attention_cfg["attn_below"]:
                self.down_self_attention.append(
                    MultiHeadSelfAttention(n_channels,
                                           image_shape= [img_dim, img_dim, n_channels],
                                           permute_layers=[self.out_permute, self.in_permute],
                                            **self.self_attention_cfg))
            else:
                self.down_self_attention.append(None)

        #self attention at low resolution
        self.up_self_attention = nn.ModuleList([])
        for img_dim, n_channels in zip(self.img_dims[::-1],np.array(self.channels)[1:][::-1]):
            if img_dim <= self.self_attention_cfg["attn_below"]:
                self.up_self_attention.append(
                    MultiHeadSelfAttention(n_channels,
                                           image_shape= [img_dim, img_dim, n_channels], 
                                           permute_layers=[self.out_permute, self.in_permute],
                                           **self.self_attention_cfg))
            else:
                self.up_self_attention.append(None)

    def forward(self, noisy_images, noise_variances=None, ctxt=None, **kwargs):
        if ctxt is None:
            ctxt = {}
        
        noisy_images=noisy_images.to(self.device)
        noise_variances=noise_variances.to(self.device)

        # noise positional encoding
        if (noise_variances is not None) and (not self._use_film):
            e = self.embedding(noise_variances)
            e = self.noise_upscale(e)
            x = T.concat([x, e],1)
        elif self._use_film:
            e = self.embedding(noise_variances, len(noisy_images.shape))
            self.film(e[:,0,0,:])
            # self.film(e[:,:,0,0])
        
        if "images" in ctxt:
            noisy_images = T.concat([noisy_images, ctxt["images"].to(self.device)],-1)

        # init conv
        x = self.start_conv(self.in_permute(noisy_images))

        # downscale part
        skips = []
        for down_blk, attention in zip(self.down_blocks[:-1], self.down_self_attention):
            x = down_blk(x, next(self.film))
            if attention is not None:
                x = attention(x)
            skips.append(x)
            x = self.downscale_embedding(x)
            
        # middel part
        x = self.down_blocks[-1](x, next(self.film))

        if self.self_attention_cfg is not None:
            x = self.down_self_attention[-1](x)

        #upscale part
        for up_blk, cross_attention, attention in zip(self.up_blocks, self.cross_attention,
                                           self.up_self_attention):
            skip = skips.pop()

            if attention is not None:
                x = attention(x)

            x = self.upscale_embedding(x)

            if cross_attention is not None:
                skip = cross_attention(skip, skip, x)

            x = T.concat([x, skip],1)
            x = up_blk(x, next(self.film))

        x = self.up_blocks[-1](x) 

        x = self.end_conv(x) # TODO add original input?

        return self.out_permute(x)

    

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
    config = misc.load_yaml("configs/configs.yaml")
    input_shape = (config.image_size, config.image_size, 3)
    model = UNet(input_shape, config.channels, config.block_depth)
    inputs = T.randn((1,3, 32,32))
    outputs = model(inputs)
    print(model.trainable_parameters())
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