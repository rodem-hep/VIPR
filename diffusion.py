
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
import diffusion_schemes as ds

def sigma(i, rho, N=20, s_min=0.002, s_max=80):
    return (s_max**(1/rho)+i/(N-1)*(s_min**(1/rho)-s_max**(1/rho)))**rho


class DiffusionModel(
    # ds.UniformDiffusion
    ds.ElucidatingDiffusion
    ):
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

        self.log_columns=["epoch", "denoise_images"]
        self.log_columns += [f"{j}_{i}" for i in ["train", "valid"]
                                for j in ["noise_loss", "image_loss"]]
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

    def generate(self, diffusion_steps, num_images=None, ctxt=None):
        # noise -> images -> denormalized images
        if num_images is None:
            initial_noise = self.initial_noise
        else:
            initial_noise = T.randn(tuple([num_images]+self.unet_config.input_shape),
                                    device=self.device)
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps,
                                                  ctxt=ctxt)
        generated_images = self.denormalize(generated_images)
        return generated_images.permute(0, 2, 3, 1)
    
    def wandb_log(self):
        self.wandb.log(self.log)
        self.log={i:[] for i in self.log_columns}
    
    def run_evaluate(self, test_loader=None, epoch_nr=0, images=None, eval_ctxt=None):

        if eval_ctxt is not None:
            eval_ctxt = eval_ctxt.to(self.device)

        _, fig = self.plot_images(test_images=images, plot_diffusion_steps=20,
                                  epoch=epoch_nr, eval_ctxt=eval_ctxt)
        if self.wandb is not None:
            self.log["denoise_images"] = fig 
            plt.close(fig)
        
        if test_loader is not None:
            noise_loss = []
            # run over training samples
            for i, ctxt in test_loader:
                log_ep = self._shared_step(i.to(self.device),
                                           ctxt=ctxt.to(self.device),
                                           training=False)
                noise_loss.append(log_ep.cpu().detach().numpy())
            self.log["noise_loss_valid"] = np.mean(noise_loss)
                
    def run_training(self, train_dataloader, n_epochs, test_loader=None,
                     run_eval=True, eval_ctxt=None):

        #progress bar for training
        pbar = tqdm(range(n_epochs))

        # create initial log            
        log = {i:[] for i in self.log_columns}

        for ep in pbar:

            if run_eval:
                # run evaluation
                self.run_evaluate(test_loader, ep, eval_ctxt=eval_ctxt)


            # run over training samples
            for i, ctxt in train_dataloader:
                log_ep = self.train_step(i.to(self.device), ctxt=ctxt.to(self.device))
                
                for key, items in log_ep.items():
                    log[key].append(items)
                    
            # input logging
            for key, items in log_ep.items():
                self.log[key] = np.mean(items)
                
            if self.log["noise_loss_valid"]<self.noise_loss_best:
                self.save(f"{self.save_path}/states/diffusion_{ep}.pth")
                self.noise_loss_best = self.log["noise_loss_valid"]

            self.log["epoch"] = ep

            if self.wandb is not None:
                self.wandb_log()
            

    def plot_images(self, test_images=None, plot_diffusion_steps=20, epoch=None,
                    logs=None, num_rows=3, num_cols=3, eval_ctxt=None):
        # plot random generated images for visual evaluation of generation quality
        if test_images is None:
            test_images = self.generate(
                # num_images=num_rows * num_cols,
                diffusion_steps=plot_diffusion_steps,
                ctxt=eval_ctxt,
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
    dataloader = hydra.utils.instantiate(config.data_cfgs)
    # test_sample = hydra.utils.instantiate(config.test_set)
    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    wandb.init(config=config, **config.wandb)

    train_loader = dataloader.train_dataloader()
    test_loader = dataloader.test_dataloader()

    #find std and mean of data
    data=[]
    data_ctxt=[]
    for nr, i in tqdm(enumerate(test_loader)):
        data.append(i[0] if len(i) else i)
        if i[1] is not None:
            data_ctxt.append(i[1] if len(i) else i)
        if nr==10:
            break
    data=T.concat(data)
    if data_ctxt[0] is not None:
        data_ctxt=T.concat(data_ctxt)
    
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
    
    if (data_ctxt[0] is not None) & (model.wandb is not None):
        _, fig_ctxt = model.plot_images(test_images=data_ctxt[:9])
        _, fig_truth = model.plot_images(test_images=data[:9])
        model.log["context_image"] = fig_ctxt
        model.log["true_image"] = fig_truth
        plt.close(fig_ctxt)
        plt.close(fig_truth)
    
    model.run_training(train_loader, config.num_epochs,
                       test_loader=test_loader,
                       eval_ctxt=data_ctxt[:9] if data_ctxt[0] is not None else None,
                       run_eval=True
                       )

        
if __name__ == "__main__":
    main()
