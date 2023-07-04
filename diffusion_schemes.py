# diffusion schemes
import torch as T
import torch.nn as nn
import numpy as np
# internal
import utils

    
class Solvers(nn.Module):
    def __init__(self, solver_name):
        super().__init__()
        if "heun2d" in solver_name:
            self.solver = self.heun2d
        else:
            self.solver = self.ddim

    def heun2d(self, initial_noise:T.Tensor, sigma_steps:np.ndarray, ctxt=None)->:
        #heuns 2nd solver

        # scale to correct std
        x = initial_noise.detach()*sigma_steps[0]
        for i in range(len(sigma_steps)-1):

            # left tangent
            dx = (x -self.denoise(x, sigma_steps[i],training=False, ctxt=ctxt)
                  )/sigma_steps[i]
            dt = (sigma_steps[i+1]-sigma_steps[i])
            # solve euler
            x_1 = x+dt*dx

            if all(sigma_steps[i+1]!=0): # solver heun 2nd
                # right tangent
                dx_ = 1/sigma_steps[i+1] * (x_1 -self.denoise(x_1, sigma_steps[i+1],
                                                         training=False,
                                                         ctxt=ctxt))

                x = (x+dt*(1/2*dx+1/2*dx_)).detach()
            else:
                x = x_1.detach()
        return x

    def ddim(self, initial_noise, diffusion_steps, ctxt=None):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images.detach()

            # separate the current noisy image to its components
            diffusion_times = T.ones((num_images, 1, 1, 1), device=self.device) - step * step_size
            noise_rates, signal_rates = self.uniform_diffusion_time(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False,
                ctxt=ctxt
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

        return pred_images

class UniformDiffusion(nn.Module):
    "from https://keras.io/examples/generative/ddim/"
    # def __init__(self):
    
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

    def reverse_diffusion(self, initial_noise, diffusion_steps, ctxt=None):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images.detach()

            # separate the current noisy image to its components
            diffusion_times = T.ones((num_images, 1, 1, 1), device=self.device) - step * step_size
            noise_rates, signal_rates = self.uniform_diffusion_time(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False,
                ctxt=ctxt
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

        return pred_images
    
    def denoise(self, noisy_images, noise_rates, signal_rates, training, ctxt=None):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network
            network.eval()
        
        if ctxt is not None:
            ctxt = self.normalizer(ctxt)

        # predict noise component and calculate the image component using it
        pred_noises = network(noisy_images,noise_rates**2, ctxt=ctxt)
        
        # remove noise from image
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def train_step(self, images, ctxt=None):
        # normalize images to have standard deviation of 1, like the noises
        log={"image_loss": None, "noise_loss":None}
        self.optimizer.zero_grad(set_to_none=True)
        images = self.normalizer(images)
            
        noises = T.randn_like(images)

        # sample uniform random diffusion times
        diffusion_times = T.rand(
            size=(len(images), 1, 1, 1),
            device=self.device
        )

        # sample time
        noise_rates, signal_rates = self.uniform_diffusion_time(diffusion_times)
        
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=True,
            ctxt=ctxt
        )

        pred_images = pred_images.detach()
        noise_loss = self.loss(noises, pred_noises)  # used for training
        image_loss = self.loss(images, pred_images)  # only used as metric

        # apply gradients
        noise_loss.backward()
        self.optimizer.step()

        log["noise_loss"] = noise_loss.cpu().detach().numpy()
        log["image_loss"] = image_loss.cpu().detach().numpy()

        # track the exponential moving averages of weights
        with T.no_grad():
            self.ema_network.exponential_moving_averages(self.network.state_dict(),
                                                        self.diffusion_config.ema)

        # T.cuda.empty_cache()
        return log
    
class ElucidatingDiffusion(Solvers):
    def __init__(self, rho=7, s_min=0.002, s_max=80, s_data=1):
        super().__init__("heun2d")
        self.rho=rho
        self.s_min=s_min
        self.s_max=s_max
        self.s_data=s_data
        self.P_mean=-1.2
        self.P_std=1.2
        self.inv_rho = (1/self.rho)
        # self.solvers = Solvers("heun2d")

    def sample_sigma(self, i:int, N:int=20, n_imgs:int=1):
        "sample sigma during generation"
        time_step = [(
            self.s_max**self.inv_rho
            +i/(N-1)*(self.s_min**self.inv_rho-self.s_max**self.inv_rho)
                      )**self.rho]*n_imgs
        time_step = T.tensor(np.stack(time_step, 0), device=self.device)
        return utils.append_dims(time_step, target_dims=4)

    
    def precondition(self, sigma):
        
        norm=T.sqrt(sigma**2+self.s_data**2)
        
        c_skip= self.s_data**2/norm**2
        
        c_out = sigma*self.s_data/norm
        
        c_input = 1/norm
        
        c_noise = 1/4*T.log(sigma)
        
        return c_skip, c_out, c_input, c_noise

    def sample_time(self,size):
        noise = T.randn(size, device=self.device, requires_grad=True)
        noise = T.exp(noise*self.P_std+self.P_mean)
        noise = T.clip(noise, self.s_min, self.s_max)
        return utils.append_dims(noise, target_dims=4)
    
    def denoise(self, images, sigma, training, ctxt=None):
        # the exponential moving average weights are used at evaluation 
        if training:
            network = self.network
        else:
            network = self.ema_network
            network.eval()
        
        if ctxt is not None:
            ctxt = self.normalizer(ctxt)

        # preconditions
        c_skip, c_out, c_input, c_noise = self.precondition(sigma)

        # make c_noise correct size
        # c_noise_img = T.reshape(c_noise, (len(images), 1, 1, 1))

        # predict noise component and calculate the image component using it
        network_output = network(c_input*images,c_noise, ctxt=ctxt)
        
        return c_skip*images+c_out*network_output
    
    def _shared_step(self, images:T.tensor, ctxt:T.tensor=None,
                     training:bool=True)->T.tensor:

        # the exponential moving average weights are used at evaluation 
        if training:
            network = self.network
        else:
            network = self.ema_network
            network.eval()

        # normalising images
        images = self.normalizer(images)
        if ctxt is not None:
            ctxt = self.normalizer(ctxt)

        # sample noise distribution
        sigma_noise = self.sample_time(len(images))

        # calculate preconditions
        c_skip, c_out, c_input, c_noise = self.precondition(sigma_noise)

        #generate noise
        noises = T.randn_like(images) * sigma_noise
        
        # mix the images with noises accordingly
        noisy_images = images + noises

        # predict noise component and calculate the image component using it
        pred_images = network(c_input*noisy_images,c_noise, ctxt=ctxt)

        # scaled target 
        scaled_target = (images-c_skip*noisy_images)/c_out
        
        # loss function
        loss = self.loss(pred_images,scaled_target)

        return loss

    def train_step(self, images, ctxt=None):
        # normalize images to have standard deviation of 1, like the noises
        log={"image_loss_train": 0, "noise_loss_train":0}
        self.optimizer.zero_grad(set_to_none=True)
            
        # loss function
        loss = self._shared_step(images, ctxt)

        # apply gradients
        loss.backward()
        self.optimizer.step()

        log["noise_loss_train"] = loss.cpu().detach().numpy()

        # track the exponential moving averages of weights
        with T.no_grad():
            self.ema_network.exponential_moving_averages(self.network.state_dict(),
                                                        self.diffusion_config.ema)

        # T.cuda.empty_cache()
        return log

    def reverse_diffusion(self, initial_noise, diffusion_steps, ctxt=None):
        #heuns solver
        sigma_steps = self.sample_sigma(np.arange(diffusion_steps), N=diffusion_steps,
                                       n_imgs=len(initial_noise)
                                       )
        sigma_steps = sigma_steps.permute(1,0,2,3).unsqueeze(-1).float()

        return self.solver(initial_noise=initial_noise,
                            sigma_steps=sigma_steps,
                            ctxt=ctxt)
