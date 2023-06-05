
import sys
import os
from datetime import datetime
import copy

import torch as T
import torch.nn as nn
import torchvision 
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import omegaconf
import wandb
import hydra

# internal
from UNet import UNet
import pipeline as pl 
from tools import misc

class DiffusionModel(nn.Module):
    def __init__(self, diffusion_config, unet_config,save_path,
                 mean, std, wandb=None, device="cuda", **kwargs):
        super().__init__()
        self.wandb=wandb
        self.device=device
        self.diffusion_config=diffusion_config
        self.save_path=save_path
        self.unet_config=unet_config
        self.normalizer = transforms.Normalize(mean=mean, std=std)
        
        self.network = UNet(device=self.device, **unet_config)
        self.optimizer = T.optim.AdamW(self.network.parameters(),
                                       lr=diffusion_config.learning_rate)
        self.ema_network = copy.deepcopy(self.network)

        self.log_columns=["epoch", "denoise_images","noise_loss", "image_loss"]
        self.log={i:[] for i in self.log_columns}

        self.loss = nn.MSELoss()
        
        # eval with same noise
        self.initial_noise = T.randn(
            tuple(kwargs.get("num_images", [9])+self.unet_config.input_shape),
            device=self.device
            )
        
        
        # save best
        self.noise_loss_best=999

        # save configs
        os.makedirs(save_path, exist_ok=True)
        for i in ["figures", "states"]:
            os.makedirs(f"{save_path}/{i}", exist_ok=True)

        misc.save_yaml(diffusion_config, f"{save_path}/diffusion_cfg.yaml")
        misc.save_yaml(unet_config, f"{save_path}/unet_cfg.yaml")
        
    def save(self, path, additional_info={}):
        states_to_save = {
            'model': self.ema_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            }
        states_to_save.update(additional_info)
        T.save(states_to_save, path)

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.std
        return T.clip(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
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

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network
            network.eval()

        # predict noise component and calculate the image component using it
        pred_noises = network(noisy_images,noise_rates**2)
        
        # remove noise from image
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
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
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, diffusion_steps, num_images=None):
        # noise -> images -> denormalized images
        if num_images is None:
            initial_noise = self.initial_noise
        else:
            initial_noise = T.randn(tuple([num_images]+self.unet_config.input_shape),
                                device=self.device)
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images.permute(0, 2, 3, 1)


    def train_step(self, images):
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
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=True
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
    
    def wandb_log(self):
        self.wandb.log(self.log)
        self.log={i:[] for i in self.log_columns}
    
    def run_evaluate(self, epoch_nr, images=None):
        generated_images, fig = self.plot_images(test_images=images, plot_diffusion_steps=20, epoch=epoch_nr)
        if self.wandb is not None:
            self.log["denoise_images"] = fig 
            plt.close(fig)
    
    def run_training(self, train_dataloader, n_epochs, run_eval=True):
        pbar = tqdm(range(n_epochs))
        for ep in pbar:
            log = {i:[] for i in self.log_columns}
            for i, _ in train_dataloader:
                log_ep = self.train_step(i.to(self.device))
                
                for key, items in log_ep.items():
                    log[key].append(items)
                    
            # input logging
            for key, items in log_ep.items():
                self.log[key] = np.mean(items)
                
            if self.log["noise_loss"]<self.noise_loss_best:
                self.save(f"{self.save_path}/states/diffusion_{ep}.pth")

            self.log["epoch"] = ep
            
            if run_eval:
                # run evaluation
                self.run_evaluate(ep)

            if self.wandb is not None:
                self.wandb_log()

    def plot_images(self, test_images=None, plot_diffusion_steps=20, epoch=None,
                    logs=None, num_rows=3, num_cols=3):
        # plot random generated images for visual evaluation of generation quality
        if test_images is None:
            test_images = self.generate(
                # num_images=num_rows * num_cols,
                diffusion_steps=plot_diffusion_steps,
            )
            test_images = test_images.cpu().detach().numpy()
        else:
            test_images= test_images.permute(0, 2, 3, 1)
        fig = plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                style = {"vmax":1, "vmin":0}
                if test_images[index].shape[-1]:
                    style["cmap"] = "gray"
                    
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(test_images[index], **style)
                plt.axis("off")
        plt.tight_layout()
        plt.title(epoch)
        return test_images, fig

@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(config):
    data = hydra.utils.instantiate(config.train_set)
    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    wandb.init(config=config,**config.wandb)
    
    dataloader = hydra.utils.instantiate(config.loader_cfg)(data)
    
    data=[]
    for nr, i in tqdm(enumerate(dataloader)):
        data.append(i[0])
        if nr==10:
            break
    data=T.concat(data)
    
    # run diffusion
    model = DiffusionModel(diffusion_config=config.diffusion_config,
                           unet_config=config.unet_config,
                           device=config.device,
                           mean=data.mean(),
                           std=data.std(),
                           save_path=config.save_path,
                           wandb=wandb
                           )
    print(f"Trainable parameters: {model.network.count_trainable_parameters()}")
    wandb.config.update({"Model Parameters": model.network.count_trainable_parameters()})
    
    model.run_training(dataloader, config.num_epochs)

        
if __name__ == "__main__":
    main()