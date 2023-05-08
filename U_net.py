
import math
import torch as T
import torch.nn as nn
import torchvision as TV

# internal 
from tools import misc

def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = T.exp(
        T.linspace(
            T.math.log(embedding_min_frequency),
            T.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = T.concat(
        [T.sin(angular_speeds * x), T.cos(angular_speeds * x)], axis=3
    )
    return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, input_size, block_depth) -> None:
        super().__init__()
        self.input_size = input_size
        self.block_depth = self.block_depth
        self.get_network()

    def get_network(self):
        self.residual = nn.ModuleList(
            [nn.Conv2d(self.input_size, self.input_size, kernel_size=1)])
        self.norm = nn.BatchNorm(center=False, scale=False)
        self.convs = nn.ModuleList(
            [nn.Conv2d(self.input_size, self.input_size, kernel_size=3, padding="same")
             for _ in range( self.block_depth)
             ])
        self.act_func = nn.SiLU()
            
    def forward(self,x):
        residual = self.residual(x)
        for nr in range(self.block_depth):
            x = self.norm(x)
            x = self.convs[nr](x)
            x = self.act_func(x)
        x = T.Add()([x, residual])
        return x



class UNet(nn.Module):
    def __init__(self, input_shape, channels, block_depth):
        super().__init__()
        self.input_shape = input_shape
        self.channels=channels
        self.block_depth=block_depth
        self.down_blocks = nn.ModuleList([])
        self.residual_block = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.init_end_conv2 = nn.ModuleList([])
        self.get_network()
        
    def get_network(self):

        self.embedding = TV.transforms.Lambda(sinusoidal_embedding)
        self.upscale_embedding = nn.Upsample(size=self.input_shape[0],
                                             mode="nearest")

        self.init_end_conv2.append(nn.Conv2d(input_shape[-1], self.channels[0],
                                             kernel_size=1))

        for input_ch, output_ch in zip(self.channels[:-1], self.channels[1:]):
            self.down_blocks.append(ResidualBlock(input_ch, output_ch,
                                                  self.block_depth))

        for _ in range(self.block_depth):
            self.residual_block.append(ResidualBlock(self.channels[-1]))

        for width in reversed(self.channels[:-1]):
            self.up_blocks.append(ResidualBlock(width, self.block_depth))

        self.init_end_conv2.append(nn.Conv2d(3, kernel_size=1,
                                             kernel_initializer="zeros"))


    def DownBlock(self, x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = nn.AvgPool2d(pool_size=2)(x)
        return x

    def UpBlock(self, x):
        x, skips = x
        x = nn.Upsample(size=2, mode="bilinear")(x)
        for _ in range(block_depth):
            x = T.concat()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x


    def forward(self, noisy_images, noise_variances):
        # noisy_images = keras.Input(shape=(image_size, image_size, 3))
        # noise_variances = keras.Input(shape=(1, 1, 1))

        e = self.embedding(noise_variances)
        e = self.upscale_embedding(e)
        x = self.init_end_conv2[0](noisy_images)

        x = T.concat([x, e])

        skips = []
        for down_blk in self.down_blocks:
            x = down_blk([x, skips])

        for res_blk in self.residual_block:
            x = res_blk(x)

        for up_blk in self.up_blocks:
            x = up_blk([x, skips])

        x = self.init_end_conv2[-1](x)

        return x

    

if __name__ == "__main__":
    config = misc.load_yaml("configs/configs.yaml")
    input_shape = (config.image_size, config.image_size, 3)
    model = UNet(input_shape, config.channels, config.block_depth)
    