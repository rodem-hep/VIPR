# diffusion schemes
import torch as T
import torch.nn as nn
import numpy as np
from tqdm import tqdm 
# internal
import src.utils as utils
from typing import Union
from torch.utils.data import DataLoader

from tools.datamodule.pipeline import Loader


def generate_gaussian_noise(shape:dict, datatype:str, 
                            eval_ctxt:dict,
                            n_constituents: Union[tuple, int]=None,
                            size:int=None, **kwargs):
    if size is None:
        size = 1e10
    # move ctxt tensor to array
    for i in eval_ctxt:
        if isinstance(eval_ctxt[i], T.Tensor):
            eval_ctxt[i] = eval_ctxt[i].cpu().numpy()
    "generate noisy images for diffusion"
    if "image" in datatype:
        if (len(eval_ctxt)) and (size > len(eval_ctxt.get("images", []))):
            size = len(eval_ctxt["images"])
        mask=None
        gaussian_noise = T.randn(tuple([size]+shape["images"])).numpy()

    elif "pc" in datatype:
        if n_constituents is None:
            raise ValueError("n_constituents has to be defined")
        
        # calculate the n constituents
        if isinstance(n_constituents, tuple):
            n_constituents = np.random.randint(*n_constituents, size)
        elif isinstance(n_constituents, int):
            n_constituents = np.random.randint(1, n_constituents, size=size)
        
        if size>len(n_constituents):
            size = len(n_constituents)
            
        if not isinstance(n_constituents, np.ndarray):
            raise TypeError("n_constituents has to be a np.array")

        mask = np.zeros([size]+shape["images"][:1])==1
        for nr,i in enumerate(n_constituents[:size]):
            mask[nr, :i] = True
        gaussian_noise = T.randn(tuple([size]+shape["images"])).numpy()
        
        #reduce eval_ctxt size
        for i in eval_ctxt:
            eval_ctxt[i] =eval_ctxt[i][:size]

    return DataLoader(Loader(gaussian_noise,mask=mask,ctxt=eval_ctxt),
                      **kwargs.get("loader_kwargs",
                                   {"batch_size": 512, "num_workers": 8})
                      )

    
class Solvers(nn.Module):
    def __init__(self, solver_name, verbose=True):
        super().__init__()
        if "heun2d" in solver_name:
            self.do_heun_step=True
        else:
            self.do_heun_step=False
        self.solver = self.heun2d
        self.verbose=verbose

    @T.no_grad()
    def heun2d(self, initial_noise:Union[T.Tensor, tuple], diffusion_steps:np.ndarray,
               ctxt:dict=None, mask:T.Tensor=None)->T.Tensor:
        if ctxt is None:
            ctxt = {}

        #heuns 2nd solver
        # scale to correct std
        x = initial_noise*diffusion_steps[0]
        for i in tqdm(range(len(diffusion_steps)-1),
                      disable=not self.verbose):

            # left tangent
            dx = 1/diffusion_steps[i] * (x-self.denoise(x, diffusion_steps[i],
                                                        training=False, ctxt=ctxt.copy(),
                                                        mask=mask))
            dt = (diffusion_steps[i+1]-diffusion_steps[i])
            # solve euler
            x_1 = x+dt*dx

            if all(diffusion_steps[i+1]!=0) & self.do_heun_step: # solver heun 2nd
                # right tangent
                dx_ = 1/diffusion_steps[i+1] * (x_1 -self.denoise(x_1, diffusion_steps[i+1],
                                                         training=False,
                                                         ctxt=ctxt.copy(),
                                                         mask=mask))

                x = (x+dt*(dx+dx_)*0.5)
            else:
                x = x_1
        return x

class UniformDiffusion(nn.Module):
    "from https://keras.io/examples/generative/ddim/"
    
    def uniform_diffusion_time(self, diffusion_times):
        
        # diffusion times -> angles
        start_angle = T.acos(T.tensor(self.diffusion_config.max_signal_rate,
                                      device=self.device))
        end_angle = T.acos(T.tensor(self.diffusion_config.min_signal_rate,
                                    device=self.device))

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = T.cos(diffusion_angles)
        noise_rates = T.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def reverse_diffusion(self, images, ctxt=None, mask=None):
        # reverse diffusion = sampling
        num_images = images.shape[0]
        step_size = 1.0 / self.eval_cfg.n_diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = images.to(self.device)
        for step in range(self.eval_cfg.n_diffusion_steps):
            noisy_images = next_noisy_images.detach()

            # separate the current noisy image to its components
            diffusion_times = T.ones([num_images]+[1]*(len(images.shape)-1),
                                     device=self.device) - step * step_size

            noise_rates, signal_rates = self.uniform_diffusion_time(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False,
                ctxt=ctxt.copy(), mask=mask
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.uniform_diffusion_time(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images.cpu()
    
    def denoise(self, noisy_images, noise_rates, signal_rates, training, ctxt=None,
                mask=None):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network
            network.eval()
        
        # if (ctxt is not None) & (self.normalizer is not None):
        #     ctxt = self.normalizer(ctxt)

        # predict noise component and calculate the image component using it
        pred_noises = network(noisy_images,noise_rates**2, ctxt=ctxt, mask=mask)
        
        # remove noise from image
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images
        
    def _shared_step(self, images:T.Tensor, ctxt:T.Tensor=None,
                     mask:T.Tensor=None, training:bool=True):
        
        # Move tensors to device 
        images= images.to(self.device)
        if mask is not None:
            mask=mask.to(self.device)
            
        noises = T.randn_like(images)

        # sample uniform random diffusion times
        diffusion_times = T.rand(
            size=[len(images)]+[1]*(len(images.shape)-1),
            device=self.device
        )

        # sample time
        noise_rates, signal_rates = self.uniform_diffusion_time(diffusion_times)
        
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=training,
            ctxt=ctxt, mask=mask
        )
        pred_images = pred_images.detach()

        noise_loss = self.loss(noises[mask], pred_noises[mask])  # used for training
        image_loss = self.loss(images[mask], pred_images[mask])  # only used as metric
        
        return {"image_loss": image_loss, "noise_loss":noise_loss}


class ElucidatingDiffusion(Solvers):
    def __init__(self, rho=7, s_min=0.002, s_max=80, s_data=1):
        super().__init__("heun2d")
        # super().__init__()
        self.rho=rho
        self.s_min=s_min
        self.s_max=s_max
        self.s_data=s_data
        self.P_mean=-1.2
        self.P_std=1.2
        self.inv_rho = (1/self.rho)

    def sample_sigma(self, i:int, N:int=20, n_imgs:int=1):
        "sample sigma during generation"
        time_step = [(
            self.s_max**self.inv_rho
            +i/(N-1)*(self.s_min**self.inv_rho-self.s_max**self.inv_rho)
                      )**self.rho]*n_imgs
        time_step = T.tensor(np.stack(time_step, 0))
        return utils.append_dims(time_step, target_dims=4)

    
    def precondition(self, sigma):
        
        norm=T.sqrt(sigma**2+self.s_data**2)
        
        c_skip= self.s_data**2/norm**2
        
        c_out = sigma*self.s_data/norm
        
        c_input = 1/norm
        
        # c_noise = 1/4*T.log(sigma)
        
        return c_skip, c_out, c_input

    def sample_time(self,size, target_dims):
        noise = T.randn(size, requires_grad=True)
        noise = T.exp(noise*self.P_std+self.P_mean)
        noise = T.clip(noise, self.s_min, self.s_max)
        return utils.append_dims(noise, target_dims=target_dims)
    
    def denoise(self, images, sigma, training, ctxt=None, mask=None):
        # the exponential moving average weights are used at evaluation 
        if ctxt is None:
            ctxt={}

        if training:
            network = self.network
        else:
            network = self.ema_network
            network.eval()
        
        # norm conditions
        if ("cnts" in ctxt) & (self.ctxt_normaliser is not None):
            ctxt["cnts"] = self.ctxt_normaliser(ctxt["cnts"],
                                                mask=ctxt["mask"])
        
        if ("scalars" in ctxt) & (self.ctxt_scalar_normaliser is not None):
            ctxt["scalars"] = self.ctxt_scalar_normaliser(ctxt["scalars"])

        # preconditions
        c_skip, c_out, c_input = self.precondition(sigma)
        
        # embedding 
        sigma = self.embedding(sigma.to(self.device), len(images.shape))

        # add noise embedding to ctxt
        if "scalars" in ctxt:
            ctxt["scalars"] = T.concat([sigma.squeeze(1), ctxt["scalars"]],1)
        else:
            ctxt["scalars"] = sigma.squeeze(1)

        # predict noise component and calculate the image component using it
        network_output = network(c_input*images, ctxt=ctxt, mask=mask).cpu()
        
        return c_skip*images+c_out*network_output
    
    def _shared_step(self, images:T.Tensor, ctxt:T.Tensor=None,
                     mask:T.Tensor=None, training:bool=True
                     )->T.Tensor:
        ctxt={} if ctxt is None else ctxt
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network
            network.eval()

        # Pass through the normalisers
        images = self.normaliser(images, mask, training=training)
        if "cnts" in ctxt:
            ctxt["cnts"] = self.ctxt_normaliser(ctxt["cnts"],
                                                mask = ctxt["mask"],
                                                training=training)
        if "scalars" in ctxt:
            ctxt["scalars"] = self.ctxt_scalar_normaliser(ctxt["scalars"],
                                                          training=training)


        # sample noise distribution
        sigma_noise = self.sample_time(len(images), len(images.shape)).to(images.device)

        # calculate preconditions
        c_skip, c_out, c_input = self.precondition(sigma_noise)

        #generate noise
        noises = T.randn_like(images) * sigma_noise
        
        # mix the images with noises accordingly
        noisy_images = images + noises

        # scaled target
        scaled_target = (images-c_skip*noisy_images)/c_out
        
        # embedding
        sigma_noise = self.embedding(sigma_noise.to(self.device), len(images.shape))
        
        # add noise embedding to ctxt
        if "scalars" in ctxt:
            ctxt["scalars"] = T.concat([sigma_noise.squeeze(1), ctxt["scalars"]],1)
        else:
            ctxt["scalars"] = sigma_noise.squeeze(1)
            

        with T.autocast(device_type=self.device, dtype=T.float16):
            # predict noise component and calculate the image component using it
            pred_images = network(c_input*noisy_images, ctxt=ctxt, mask=mask)
            # assert pred_images.dtype is T.float16

            # loss function
            loss = self.loss(pred_images[mask],scaled_target[mask].to(self.device))
            # assert loss.dtype is T.float32

        return {"noise_loss":loss}

    def reverse_diffusion(self, images, ctxt=None, mask=None):
        #heuns solver
        sigma_steps = self.sample_sigma(np.arange(self.eval_cfg.n_diffusion_steps),
                                        N=self.eval_cfg.n_diffusion_steps,
                                       n_imgs=len(images)
                                       ).permute(1,0,2,3).float()

        if len(images.shape)+1!=len(sigma_steps.shape): # images might need addtional dimensions
            sigma_steps = sigma_steps.unsqueeze(-1)

        return self.solver(initial_noise=images,
                            diffusion_steps=sigma_steps,
                            ctxt=ctxt, mask=mask)
