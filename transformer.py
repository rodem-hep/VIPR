"Transformer"
import math
import torch as T
import torch.nn as nn
import torchvision as TV
import numpy as np
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
    # TODO should not be for images
    def __init__(self, depth_q, depth_vk, image_shape_q, image_shape_vk, pos_encode_kwargs,
                 attn_heads=4, trainable_pe=True, device="cuda", **kwargs):
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
        if trainable_pe: #TODO should there be one if self attention
           self.pe_q=  T.nn.Parameter(T.randn(*self.image_shape_q))
           self.pe_vk =  T.nn.Parameter(T.randn(*self.image_shape_vk))
        else:
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
            nn.Conv1d(self.depth_q, self.depth_vk, 1),
            nn.BatchNorm1d(self.depth_vk))

        self.W_key = nn.Sequential(
            nn.Conv1d(self.depth_vk, self.depth_vk, 1),
            nn.BatchNorm1d(self.depth_vk))

        self.W_value = nn.Sequential(
            nn.Conv1d(self.depth_vk, self.depth_vk, 1),
            nn.BatchNorm1d(self.depth_vk))
        
        self.out_proj = nn.Conv1d(self.depth_vk, self.depth_q, 1)
        self.out_proj.weight.data.fill_(0.00)
        self.out_proj.bias.data.fill_(0.00)

        self.to(self.device)
    
    def forward(self, v:T.Tensor, k:T.Tensor, q:T.Tensor)->T.Tensor:
        
        return self._forward(v, k, q)

    def _forward(self, v: T.Tensor, k: T.Tensor, q: T.Tensor) -> T.Tensor:


        # Get the shape of the q tensor and save the original for later
        q_orig = q.clone()
        b, c, *spatial = q.shape
        
        # Add positional encoding
        v = v+self.pe_vk
        k = k+self.pe_vk
        q = q+self.pe_q

        # Flatten each image to combine the spacial dimensions: B, C/F, HxW/point-cloud
        q = T.flatten(q, start_dim=2)
        k = T.flatten(k, start_dim=2)
        v = T.flatten(v, start_dim=2)

        # Project using the learnable layers: B, model_dim/features, HxW/point-cloud
        q = self.W_query(q)
        k = self.W_key(k)
        v = self.W_value(v)

        # Break (model_dim or features = num_heads x channels_per_head) for the different heads
        shape = (b, self.attn_heads, self.depth_vk_blk, -1)
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
        a_out = a_out.transpose(-1, -2).contiguous().view(b, self.depth_vk, -1)

        # Pass through the final 1x1 convolution layer: B, q_dim, HxW
        a_out = self.out_proj(a_out)

        # Bring back spacial dimensions: B, q_dim, H, W
        a_out = a_out.view(b, -1, *spatial)

        # Return the additive connection to the original q
        return q_orig+a_out

    # def forward(self, values, keys, queries):

    #     #B,C,H,W
    #     nbatches = queries.size(0)
    #     query_ori = queries.clone()
        
    #     pe_q = self.pe_q.repeat_interleave(nbatches, 0)
    #     pe_vk = self.pe_vk.repeat_interleave(nbatches, 0)
    #     values = values.to(self.device)+pe_vk
    #     keys = keys.to(self.device)+pe_vk
    #     queries = queries.to(self.device)+pe_q
        
    #     #B,C,HW
    #     values_flatten = self.W_value(values.flatten(2))
    #     queries_flatten = self.W_query(queries.flatten(2))
    #     keys_flatten = self.W_key(keys.flatten(2))

    #     #B,n_head, h_dim, HW
    #     values_head_dim = values_flatten.view(nbatches, self.attn_heads,
    #                                          self.depth_vk_blk, -1)
    #     queries_head_dim = queries_flatten.view(nbatches, self.attn_heads,
    #                                          self.depth_vk_blk, -1)
    #     keys_head_dim = keys_flatten.view(nbatches, self.attn_heads,
    #                                          self.depth_vk_blk, -1)

    #     attn_mat, p_attn = attention(queries_head_dim, keys_head_dim, values_head_dim)
        
    #     attn_mat = self.out_proj(attn_mat.flatten(1,2))

    #     return query_ori+attn_mat.view(*queries.shape)

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

        gate_images = self._forward(values, keys, queries)

        return self.conv(gate_images) * values
        

class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, depth, image_shape, pos_encode_kwargs,
                 attn_heads=4, device="cuda", **kwargs):
        super().__init__(depth, depth, image_shape_q=image_shape,
                         image_shape_vk=image_shape,
                         pos_encode_kwargs=pos_encode_kwargs,
                         attn_heads=attn_heads, device=device, **kwargs)

    def forward(self, inputs):
        return inputs+self._forward(inputs,inputs,inputs)

if __name__ == "__main__":
    config = misc.load_yaml("configs/diffusion_cfg.yaml")
    depth = 32
    model = MultiHeadSelfAttention(depth,[depth, 3,3], 
                                   **config.unet_config.self_attention_cfg)
    inputs = T.randn((1,depth, 3,3))
    outputs  = model(inputs)
    # conv = nn.Conv1d(depth, depth, 1)
    # out = conv(inputs)
    print(outputs)
    # print(model.trainable_parameters())