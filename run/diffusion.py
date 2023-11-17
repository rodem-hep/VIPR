import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import sys
import os
import copy

import torch as T
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import omegaconf
import wandb
import hydra

# internal
from tools import misc
from tools import schedulers
import src.diffusion_schemes as ds
# from src.models.image_model import UNet # used on hydra
from src.utils import fig2img
import tools.datamodule.pipeline as pl
import src.positional_encoding as pe

# from src.models.modules import IterativeNormLayer
from tools.modules import IterativeNormLayer
# jet_2023_11_13_12_14_10_126343


class DiffusionModel(
    # ds.UniformDiffusion
    ds.ElucidatingDiffusion
    ):
    def __init__(self, init_noise, embedding_cfg, train_cfg, eval_cfg, 
                 network, save_path=None, train_loader=None, test_loader=None,
                 wandb=None, device="cuda", **kwargs):
        super().__init__()
        self.init_noise = init_noise
        self.embedding_cfg=embedding_cfg
        self.train_cfg=train_cfg
        self.eval_cfg=eval_cfg
        self.wandb=wandb
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.device=device
        self.save_path=save_path
        self.run_eval = self.eval_cfg.eval_iters>0
        # normalisers
        self.ctxt_normaliser = None
        self.ctxt_scalar_normaliser = None
        

        # mixed precision
        self.loss_scaler = None
        self.loss_scaler = T.cuda.amp.GradScaler()
        self.eval_ctxt =None
        inpt_shape = kwargs.get("inpt_shape")
        if inpt_shape is None:
            inpt_shape = self.train_loader.dataset._shape()

        self.eval_fw = kwargs.get("eval_fw", self.test_loader)
        self.out_trans = kwargs.get("out_trans", None)

        # init normalization layers
        self.normaliser = IterativeNormLayer(inpt_shape["images"][-1]).to(self.device)
        if "ctxt_images" in inpt_shape:
            self.ctxt_normaliser = IterativeNormLayer(inpt_shape["ctxt_images"][-1]).to(self.device)
        if "ctxt_scalars" in inpt_shape:
            self.ctxt_scalar_normaliser = IterativeNormLayer(inpt_shape["ctxt_scalars"][-1]).to(self.device)
        
        # push network to device        
        self.network = network.to(self.device)

        # copy network for ema
        self.ema_network = copy.deepcopy(self.network)
        self.ema_network.eval()

        self.optimizer = T.optim.AdamW(self.network.parameters(),
                                       lr=train_cfg.learning_rate)
        if "lr_scheduler" in train_cfg:
            self.lr_scheduler = schedulers.get_scheduler(optimizer=self.optimizer,
                                                         **train_cfg.lr_scheduler)


        # init log
        self.log_columns=["epoch", "denoise_images", "lr"]
        self.log_columns += [f"{j}_{i}" for i in ["train", "valid"]
                                for j in ["noise_loss", "image_loss","clip"]]
        self.log={i:[] for i in self.log_columns}
        self.n_train_size=0

        # init loss function
        self.loss = nn.MSELoss()

        # eval with same noise
        if (self.test_loader is not None):
            self.eval_ctxt = self.test_loader.dataset.get_normed_ctxt()

            n_cnts=None
            if "pc" in self.init_noise.datatype:
                n_cnts = self.eval_ctxt.pop("true_n_cnts",self.test_loader.dataset.max_cnstits)
            self.initial_noise = ds.generate_gaussian_noise(eval_ctxt=self.eval_ctxt,
                                                            n_constituents=n_cnts,
                                                            **self.init_noise)
            if (("images" in self.eval_ctxt)
                & ("image" in self.init_noise.datatype)):
                #find std and mean of data
                log = self.eval_fw(self.eval_ctxt["images"][:9], name="context_image")
                self.log.update(log)

        train_cfg["ctxt_dims"] = self.network.ctxt_dims
        if "embedding" not in train_cfg:
            train_cfg["embedding"] = "sinusoidal"
        
        # init embedding
        if "sinusoidal" in train_cfg["embedding"]:
            self.embedding = pe.Sinusoidal(device=self.device, **self.embedding_cfg)
        else:
            self.embedding = pe.FourierFeatures(1, device=self.device, **self.embedding_cfg)
        
        
        # save best
        self.noise_loss_best=999
        
        # save configs and make folder
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            for i in ["figures", "states"]:
                os.makedirs(f"{save_path}/{i}", exist_ok=True)
                
            misc.save_yaml(train_cfg, f"{save_path}/diffusion_cfg.yaml")
        


    def save(self, path, additional_info={}):
        # save complet diffusion setup
        T.save(self.state_dict(), path)
        
        # save individual optimizer/network with additional info
        states_to_save = {
            'model': self.ema_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            }
        states_to_save.update(additional_info)
        T.save(states_to_save, path.replace("diffusion_", "model_"))


    def generate(self, images,  ctxt=None, mask=None):
        # noise -> images -> denormalized images
        with T.no_grad():
            generated_images = self.reverse_diffusion(images=images, ctxt=ctxt, mask=mask)

        generated_images = self.normaliser.reverse(generated_images, mask=mask).cpu()

        # clip between 0 and 1 for images
        if "image" in self.init_noise.datatype:
            generated_images = T.clip(generated_images, 0, 1)

        # create output dict
        gen_data = {"gen_data": generated_images}
        if ctxt is not None:
            gen_data["ctxt"] = ctxt
        if mask is not None:
            gen_data["mask"] = mask

        return gen_data

    
    def wandb_log(self):
        self.wandb.log(self.log)
        self.log={i:[] for i in self.log_columns}
    
    def generate_samples(self, initial_noise, disable_bar=True):

        generated_data={}

        for sample in tqdm(initial_noise, total=len(initial_noise),
                            disable=disable_bar # len(self.initial_noise)==1
                            ):

            # sample = {i:j.to(self.device) for i,j in sample.items()}

            _generated = self.generate(**sample)

            # concat to generated_data
            for i,j in _generated.items():
                if isinstance(j, dict): # for dict nested ctxt
                    if i not in generated_data:
                        generated_data[i]={}
                    for k,l in j.items():
                        if k not in generated_data[i]:
                            generated_data[i][k]=T.tensor([])
                        
                        generated_data[i][k] = T.concat([generated_data[i][k], l],0)
                else:
                    if T.isnan(_generated[i]).any():
                        print("why")
                    if i not in generated_data:
                        generated_data[i]=T.tensor([])
                    generated_data[i] = T.concat([generated_data[i], _generated[i]],0)

        return generated_data

    def run_evaluate(self, test_loader, epoch_nr=0, disable_bar=True):

        # plot random generated images for visual evaluation of generation quality
        if (not epoch_nr%self.eval_cfg.eval_iters):# & (epoch_nr>0):

            # generate sample
            generated_data = self.generate_samples(self.initial_noise,
                                                   disable_bar=disable_bar)

            # evaluate in framework and log
            if self.eval_fw is not None:
                log = self.eval_fw(**generated_data, name="generated_images", n_epoch=epoch_nr)
                self.log.update(log)

        # validate training scores
        if (test_loader is not None) & (self.wandb is not None):
            noise_loss = {i.replace("_valid", ""):[] for i in self.log.keys()
                          if "valid" in i}
            # run over training samples
            for sample in tqdm(test_loader, disable=True):

                # sample = {i: j.to(self.device) for i,j in sample.items()}
                with T.no_grad():
                    log_ep = self._shared_step(**sample, training=False)
                for i,j in log_ep.items():
                    noise_loss[i].append(j.cpu().detach().numpy())

            for i,j in noise_loss.items():
                if len(j)>0:
                    self.log[f"{i}_valid"] = np.mean(j)

    def train_step(self, images, ctxt=None, mask=None):
        # normalize images to have standard deviation of 1, like the noises
        self.optimizer.zero_grad(set_to_none=True)
            
        # loss function
        log = self._shared_step(images, ctxt, mask)

        # apply gradients
        # apply gradients w/wo mp
        if self.loss_scaler is not None:
            self.loss_scaler.scale(log["noise_loss"]).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            self.loss_scaler.unscale_(self.optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            clip_gradient = T.nn.utils.clip_grad_norm_(self.network.parameters(), 10, error_if_nonfinite=False)
            if not clip_gradient.isnan():
                log["clip"] = clip_gradient
            # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
            self.loss_scaler.step(self.optimizer)
            self.loss_scaler.update()
        else:
            log["noise_loss"].backward()
            self.optimizer.step()
            
        self.lr_scheduler.step()

        # track the exponential moving averages of weights
        with T.no_grad():
            self.ema_network.ema(self.network.state_dict(),self.train_cfg.ema)

        return {i:j.cpu().detach().numpy() for i,j in log.items()}
                
    def run_training(self):
        
        #progress bar for training
        pbar = tqdm(range(self.train_cfg.num_epochs))


        for ep in pbar:

            if self.run_eval:
                # run evaluation
                self.run_evaluate(self.test_loader, ep)

            # run over training samples
            for nr, sample in enumerate(self.train_loader):

                log_ep = self.train_step(**sample)
                self.n_train_size+=len(sample["images"])
                
                for key, items in log_ep.items():
                    # if np.isnan(items):
                    #     raise ValueError("is NaN")
                    self.log[key+"_train"].append(items)

            self.log["n_samples"] = self.n_train_size
            
            # input logging
            for key in log_ep:
                self.log[key+"_train"] = np.mean(self.log[key+"_train"])

            self.log["lr"] = self.optimizer.state_dict()["param_groups"][0]["lr"]
            
            if self.run_eval & True: # TODO not using the save model yet
                if self.log["noise_loss_valid"]<self.noise_loss_best:
                    self.save(f"{self.save_path}/states/diffusion_{ep}.pth")
                    self.noise_loss_best = self.log["noise_loss_valid"]

            self.log["epoch"] = ep

            if self.wandb is not None:
                self.wandb_log()



@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="config")
def main(config):
    #dataloader
    train_loader = hydra.utils.instantiate(config.data.train,
                                           loader_config=config.data.loader_config)
    if "jet" in config.data.valid._target_.lower():
        test_loader = hydra.utils.instantiate(config.data.valid,
                                            loader_config=config.data.loader_config,
                                            max_cnstits=train_loader.dataset.max_cnstits,
                                            datatype=train_loader.dataset.datatype,
                                            )
    else:
        test_loader = hydra.utils.instantiate(config.data.valid,
                                            loader_config=config.data.loader_config,
                                            )
    
    # init W&B
    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )

    wandb.init(config=config, **config.wandb)

    # init network
    network=hydra.utils.instantiate(config.model,ctxt_dims=train_loader.get_ctxt_shape())

    # init diffusion
    diffusion = hydra.utils.instantiate(
        config.trainer,
        network=network,
        train_loader=train_loader.train_dataloader(),
        test_loader=test_loader.test_dataloader(),
        device=config.device,
        save_path=config.save_path,
        eval_fw=test_loader,
        wandb=wandb,
        )

    print(f"Trainable parameters: {diffusion.network.count_trainable_parameters()}")
    wandb.config.update({"Model Parameters": diffusion.network.count_trainable_parameters()})

    diffusion.run_training()

        
if __name__ == "__main__":
    main()
