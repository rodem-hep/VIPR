"UNet"
import math
import torch as T
import torch.nn as nn
import torchvision as TV
import numpy as np

# internal 
from tools import misc

def sinusoidal_embedding(x, embedding_min_frequency,
                         embedding_max_frequency,
                         embedding_dims,
                         device="cuda"):
    frequencies = T.exp(
        T.linspace(
            np.log(embedding_min_frequency),
            np.log(embedding_max_frequency),
            embedding_dims // 2, device=device
        )
    )
    angular_speeds = 2.0 * T.pi * frequencies
    embeddings = T.concat(
        [T.sin(angular_speeds * x), T.cos(angular_speeds * x)], axis=3
    )
    embeddings = embeddings.reshape(x.shape[0],embedding_dims,1,1)
    return embeddings

class sinusoidal:
    def __init__(self, embedding_max_frequency, embedding_dims, device="cuda"):
        self.embedding_min_frequency = 1.0
        self.embedding_max_frequency=embedding_max_frequency
        self.embedding_dims=embedding_dims
        self.device=device

    def __call__(self, x):
        return sinusoidal_embedding(x,
                                    self.embedding_min_frequency,
                                    self.embedding_max_frequency,
                                    self.embedding_dims,
                                    self.device
                                    )

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, block_depth) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size=output_size
        self.block_depth = block_depth
        self.get_network()

    def get_network(self):
        self.residual = nn.Conv2d(self.input_size, self.output_size, kernel_size=1)
        self.norm = nn.ModuleList([
            nn.BatchNorm2d(self.input_size, self.output_size, affine=False)])
        self.convs = nn.ModuleList([nn.Conv2d(self.input_size, self.output_size,
                                              kernel_size=3, padding="same")])
        for _ in range(self.block_depth-1):
            self.norm.append(nn.BatchNorm2d(self.output_size, self.output_size, affine=False))
            self.convs.append(nn.Conv2d(self.output_size, self.output_size,
                                        kernel_size=3, padding="same"))
        self.act_func = nn.SiLU()
            
    def forward(self,x):
        residual = self.residual(x) # TODO norm before here?
        for nr in range(self.block_depth):
            x = self.norm[nr](x)
            x = self.convs[nr](x)
            x = self.act_func(x)
        x = T.add(x, residual)
        return x

class UNet(nn.Module):
    def __init__(self, input_shape, channels, block_depth,
                 diffusion:bool=False, embedding_max_frequency=None, embedding_dims=0, device="cuda"):
        super().__init__()
        self.input_shape = input_shape
        self.embedding_max_frequency=embedding_max_frequency
        self.embedding_dims=embedding_dims
        self.channels=channels
        self.device=device
        self.block_depth=block_depth
        self.diffusion=diffusion
        self.down_blocks = nn.ModuleList([])
        self.residual_block = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.init_end_conv2 = nn.ModuleList([])
        self.get_network()
        self.to(self.device)
        
    def exponential_moving_averages(self, state_dict, ema_ratio):
        ema_state_dict = self.state_dict()
        for (key, weight), (em_key, ema_para) in zip(state_dict.items(),
                                                     ema_state_dict.items()):
            # if em_key!=key:
            #     raise ValueError("Differences between state_dict ema network and network")
            ema_state_dict[em_key] = ema_ratio * ema_para + (1 - ema_ratio) * weight

        self.load_state_dict(ema_state_dict)

    def trainable_parameters(self):
        for parameter in self.parameters():
            if parameter.requires_grad:
                yield parameter
        
    def count_trainable_parameters(self):
        sum_trainable=0
        for paras in self.trainable_parameters():
            sum_trainable+=np.sum(paras.detach().numpy().shape)
        return sum_trainable
        
    def get_network(self):

        # self.embedding = TV.transforms.Lambda(sinusoidal_embedding)
        self.embedding = sinusoidal(self.embedding_max_frequency,
                                    self.embedding_dims)

        self.noise_upscale = nn.Upsample(size=self.input_shape[1:], mode="nearest")
        self.upscale_embedding = nn.Upsample(scale_factor=2,mode="nearest")
        self.downscale_embedding = nn.AvgPool2d(2)

        self.init_end_conv2.append(nn.Conv2d(
            self.input_shape[0],
            self.channels[0],
            kernel_size=1))

        #downscale network
        self.down_blocks.append(ResidualBlock(self.channels[0]+self.embedding_dims,
                                              self.channels[1], self.block_depth))
        self.residual_block.append(ResidualBlock(self.channels[1],
                                                 self.channels[1]//2, 1))

        for input_ch, output_ch in zip(self.channels[1:-1], self.channels[2:]):
            self.down_blocks.append(ResidualBlock(input_ch, output_ch,
                                                  self.block_depth))
            self.residual_block.append(ResidualBlock(output_ch,output_ch//2, 1))

        # upscale network
        self.up_blocks.append(ResidualBlock(self.channels[::-1][0],
                                            self.channels[::-1][1],
                                            self.block_depth))

        for input_ch, output_ch in zip(self.channels[::-1][1:],
                                       self.channels[::-1][2:]):
            self.up_blocks.append(ResidualBlock(2*input_ch, output_ch,
                                                self.block_depth))

        self.init_end_conv2.append(nn.Conv2d(2*self.input_shape[0]+1,
                                             self.input_shape[0], kernel_size=1)) # TODO zero inits
    
    # def 

    def forward(self, noisy_images, noise_variances=None):

        x = self.init_end_conv2[0](noisy_images)

        if noise_variances is not None:
            e = self.embedding(noise_variances)
            e = self.noise_upscale(e)
            x = T.concat([x, e],1)

        skips = []
        for down_blk, res_blk in zip(self.down_blocks, self.residual_block):
            x = down_blk(x)
            skips.append(res_blk(x))
            x = self.downscale_embedding(x)

        for up_blk, skip_conn in zip(self.up_blocks, skips[::-1]):
            # print(up_blk)
            x = up_blk(x)
            x = self.upscale_embedding(x)
            x = T.concat([x, skip_conn],1)

        x = self.init_end_conv2[-1](x)

        return x

    

if __name__ == "__main__":
    config = misc.load_yaml("configs/configs.yaml")
    input_shape = (config.image_size, config.image_size, 3)
    model = UNet(input_shape, config.channels, config.block_depth)
    inputs = T.randn((1,3, 32,32))
    outputs = model(inputs)
    print(model.trainable_parameters())